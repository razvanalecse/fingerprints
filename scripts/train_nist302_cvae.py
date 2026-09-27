#!/usr/bin/env python3
"""Train and evaluate uncertainty-aware CVAEs on registered SD302 pairs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.losses.cvae import RegisteredApproximateCVAELoss
from fingerprint_reconstruction.metrics import (
    empirical_interval_coverage,
    paired_ridge_frequency_error,
    pairwise_diversity_mae,
    region_image_metrics,
    uncertainty_error_spearman,
)
from fingerprint_reconstruction.models.cvae import ConditionalVAE, SpatialConditionalVAE
from fingerprint_reconstruction.models.support import (
    VISIBLE_LATENT_SUPPORT,
    FingerprintSupportPredictor,
    apply_support_constraint,
)
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_orientation_field,
    orientation_error,
)
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help=(
            "Skip training and evaluate this checkpoint on the complete validation "
            "split by default. Pass --split test to deliberately evaluate on the "
            "sealed test split instead; test is otherwise never instantiated."
        ),
    )
    parser.add_argument("--run-epochs", type=int)
    parser.add_argument(
        "--support-mode", choices=("soft", "hard", "feathered"),
        help="Evaluation-time override of the configured support composition.",
    )
    parser.add_argument("--support-threshold", type=float)
    parser.add_argument("--support-feather-sigma", type=float)
    parser.add_argument(
        "--decoder-upsampling-mode",
        choices=("nearest", "bilinear"),
        help="Evaluation/training override for decoder interpolation.",
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Split to evaluate against when --evaluate-checkpoint is given; ignored otherwise "
             "(training always uses train/validation). Test is never instantiated implicitly.",
    )
    return parser.parse_args()


def code_provenance() -> dict[str, object]:
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ("git", *arguments), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = {}
    for package in ("numpy", "scipy", "torch", "torchvision", "scikit-image"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "packages": packages,
    }


def load_support_predictor(path: Path | None, device: torch.device):
    if path is None:
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("target_semantics") != VISIBLE_LATENT_SUPPORT:
        raise ValueError("support checkpoint must predict official visible latent support")
    model = FingerprintSupportPredictor(
        channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def support_probability(
    model: torch.nn.Module | None,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if model is None:
        return None
    return torch.maximum(model(torch.cat((observed, mask), dim=1)), mask)


def run_epoch(
    *,
    model: torch.nn.Module,
    support_model: torch.nn.Module | None,
    loader: DataLoader,
    objective: RegisteredApproximateCVAELoss,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if support_model is not None:
        support_model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        roi = batch["evaluation_roi"].to(device)
        confidence = batch["geometric_confidence"].to(device)
        predicted_support = support_probability(support_model, observed, mask)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, mu, logvar = model(target, observed, mask)
            output = objective(
                reconstruction,
                target,
                mask,
                mu,
                logvar,
                beta=beta,
                evaluation_roi=roi,
                geometric_confidence=confidence,
                observed=observed,
                support_probability=predicted_support,
            )
            if not torch.isfinite(output.total):
                raise FloatingPointError(f"non-finite loss at batch {batch_index}")
            if training:
                output.total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                totals["gradient_norm_before_clip"] += float(gradient_norm) * len(target)
        batch_size = int(target.shape[0])
        examples += batch_size
        totals["loss"] += float(output.total.detach()) * batch_size
        for name, value in output.components.items():
            totals[name] += float(value) * batch_size
    if examples == 0:
        raise RuntimeError("epoch contained no examples")
    return {name: value / examples for name, value in totals.items()} | {
        "examples": float(examples)
    }


@torch.no_grad()
def sample_in_chunks(
    model: torch.nn.Module,
    observed: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_samples: int,
    chunk_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("sample chunk size must be positive")
    chunks = []
    for start in range(0, num_samples, chunk_size):
        count = min(chunk_size, num_samples - start)
        chunks.append(
            model.sample(
                observed, mask, num_samples=count, generator=generator
            )
        )
    return torch.cat(chunks, dim=1)


def constrain_sample_batch(
    samples: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    predicted_support: torch.Tensor | None,
    *,
    mode: str,
    threshold: float,
    feather_sigma: float = 1.5,
) -> torch.Tensor:
    if predicted_support is None:
        return samples
    batch, count, _, height, width = samples.shape
    constrained = apply_support_constraint(
        samples.reshape(batch * count, 1, height, width),
        observed.repeat_interleave(count, dim=0),
        mask.repeat_interleave(count, dim=0),
        predicted_support.repeat_interleave(count, dim=0),
        mode=mode,
        threshold=threshold,
        feather_sigma=feather_sigma,
    )
    return constrained.reshape(batch, count, 1, height, width)


@torch.no_grad()
def probabilistic_evaluation(
    model: torch.nn.Module,
    support_model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    *,
    k_values: tuple[int, ...],
    nominal_coverages: tuple[float, ...],
    sample_chunk_size: int,
    evaluation_seed: int,
    support_mode: str,
    support_threshold: float,
    support_feather_sigma: float = 1.5,
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    """Report expected, mean, best-of-K, diversity and calibration separately."""

    if not k_values or min(k_values) < 2 or tuple(sorted(set(k_values))) != k_values:
        raise ValueError("k_values must be sorted unique integers >=2")
    model.eval()
    if support_model is not None:
        support_model.eval()
    generator = torch.Generator().manual_seed(int(evaluation_seed))
    maximum_k = max(k_values)
    records: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        latent = batch["latent_image"].to(device)
        quality = batch["quality"].to(device)
        predicted_support = support_probability(support_model, observed, mask)
        samples = sample_in_chunks(
            model,
            observed,
            mask,
            num_samples=maximum_k,
            chunk_size=sample_chunk_size,
            generator=generator,
        )
        samples = constrain_sample_batch(
            samples,
            observed,
            mask,
            predicted_support,
            mode=support_mode,
            threshold=support_threshold,
            feather_sigma=support_feather_sigma,
        )
        for index in range(len(observed)):
            heldout = (quality[index, 0] == 1).cpu().numpy()
            background = (quality[index, 0] == 0).cpu().numpy()
            if not heldout.any():
                continue
            reference = latent[index, 0].cpu().numpy()
            per_sample = samples[index, :, 0].cpu().numpy()
            record: dict[str, object] = {
                "sample_id": batch["sample_id"][index],
                "subject_id": batch["subject_id"][index],
            }
            single_metrics = region_image_metrics(reference, per_sample[0], heldout)
            record.update(
                {f"single_heldout_q1_{name}": value for name, value in single_metrics.items()}
            )
            record["single_background_darkness"] = float(
                np.mean(1.0 - per_sample[0][background])
            )
            for count in k_values:
                selected = per_sample[:count]
                mean_estimate = selected.mean(axis=0)
                mean_metrics = region_image_metrics(reference, mean_estimate, heldout)
                record.update(
                    {
                        f"mean_k{count}_heldout_q1_{name}": value
                        for name, value in mean_metrics.items()
                    }
                )
                sample_mae = np.mean(
                    np.abs(selected[:, heldout] - reference[heldout][None]), axis=1
                )
                record[f"best_k{count}_heldout_q1_mae"] = float(sample_mae.min())
                record[f"diversity_k{count}_heldout_q1_mae"] = pairwise_diversity_mae(
                    selected, heldout
                )
                try:
                    record[f"uncertainty_error_spearman_k{count}"] = (
                        uncertainty_error_spearman(selected, reference, heldout)
                    )
                except ValueError:
                    record[f"uncertainty_error_spearman_k{count}"] = float("nan")
                for nominal in nominal_coverages:
                    coverage, width = empirical_interval_coverage(
                        selected,
                        reference,
                        heldout,
                        nominal_coverage=nominal,
                    )
                    label = int(round(100 * nominal))
                    record[f"coverage_k{count}_{label}"] = coverage
                    record[f"coverage_error_k{count}_{label}"] = abs(coverage - nominal)
                    record[f"interval_width_k{count}_{label}"] = width
                uncertainty = selected.std(axis=0, ddof=1)
                record[f"mean_uncertainty_k{count}_heldout_q1"] = float(
                    uncertainty[heldout].mean()
                )
                record[f"mean_uncertainty_k{count}_background"] = float(
                    uncertainty[background].mean()
                )
                record[f"mean_k{count}_background_darkness"] = float(
                    np.mean(1.0 - mean_estimate[background])
                )

            mean_estimate = per_sample.mean(axis=0)
            reference_field = estimate_orientation_field(
                reference, use_foreground_mask=False
            )
            mean_field = estimate_orientation_field(
                mean_estimate, use_foreground_mask=False
            )
            try:
                record["mean_heldout_q1_orientation_error"] = orientation_error(
                    reference_field, mean_field, region_mask=heldout
                )
            except ValueError:
                record["mean_heldout_q1_orientation_error"] = float("nan")
            frequency = paired_ridge_frequency_error(reference, mean_estimate, heldout)
            record.update(
                {f"mean_heldout_q1_{name}": value for name, value in frequency.items()}
            )
            # Stable aliases refer explicitly to the maximum-K estimate.
            for name, value in region_image_metrics(
                reference, mean_estimate, heldout
            ).items():
                record[f"mean_heldout_q1_{name}"] = value
            record["pairwise_diversity_mae"] = record[
                f"diversity_k{maximum_k}_heldout_q1_mae"
            ]
            record["uncertainty_error_spearman"] = record[
                f"uncertainty_error_spearman_k{maximum_k}"
            ]
            if predicted_support is not None:
                predicted_binary = (
                    predicted_support[index, 0] >= support_threshold
                ).cpu().numpy()
                true_support = (quality[index, 0] >= 1).cpu().numpy()
                intersection = np.logical_and(predicted_binary, true_support).sum()
                union = np.logical_or(predicted_binary, true_support).sum()
                record["predicted_support_iou"] = float(intersection / max(union, 1))
            records.append(record)
    if not records:
        raise ValueError("no validation image had held-out quality-1 pixels")
    return records


@torch.no_grad()
def save_preview(
    model: torch.nn.Module,
    support_model: torch.nn.Module | None,
    batch: dict[str, object],
    device: torch.device,
    output: Path,
    *,
    samples_k: int,
    sample_chunk_size: int,
    evaluation_seed: int,
    support_mode: str,
    support_threshold: float,
    support_feather_sigma: float = 1.5,
    title: str,
) -> None:
    model.eval()
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    target = batch["target"][:1].to(device)
    latent = batch["latent_image"][:1].to(device)
    predicted_support = support_probability(support_model, observed, mask)
    generator = torch.Generator().manual_seed(int(evaluation_seed))
    samples = sample_in_chunks(
        model,
        observed,
        mask,
        num_samples=samples_k,
        chunk_size=sample_chunk_size,
        generator=generator,
    )
    samples = constrain_sample_batch(
        samples,
        observed,
        mask,
        predicted_support,
        mode=support_mode,
        threshold=support_threshold,
        feather_sigma=support_feather_sigma,
    )[0]
    mean_sample, std_sample = samples.mean(dim=0), samples.std(dim=0)
    display_observed = torch.where(
        mask[0].bool(), observed[0], torch.full_like(observed[0], 0.72)
    )
    panels = [
        latent[0, 0].cpu().numpy(),
        display_observed[0].cpu().numpy(),
        target[0, 0].cpu().numpy(),
        mean_sample[0].cpu().numpy(),
        samples[0, 0].cpu().numpy(),
        samples[min(1, samples_k - 1), 0].cpu().numpy(),
        std_sample[0].cpu().numpy(),
    ]
    titles = [
        "Real latent",
        "Y,M",
        "X_pseudo",
        f"Predictive mean K={samples_k}",
        "Single sample 1",
        "Single sample 2",
        "Predictive std",
    ]
    if predicted_support is not None:
        panels.append(predicted_support[0, 0].cpu().numpy())
        titles.append("P(visible support|Y,M)")
    figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for axis, panel, panel_title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="magma" if panel_title == "Predictive std" else "gray")
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    if confidence_report["source_split"] != "validation":
        raise ValueError("geometric confidence must be calibrated on validation")
    confidence_weights = tuple(float(value) for value in confidence_report["ordered_weights"])
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    shape = tuple(int(value) for value in config["data"]["image_size"])
    requested_split = getattr(args, "split", "validation")
    if requested_split != "validation" and args.evaluate_checkpoint is None:
        raise ValueError("--split test/train is only meaningful with --evaluate-checkpoint")
    # The "validation" dict/loader key is reused for whichever split was
    # requested for evaluation, so every downstream reference to
    # datasets["validation"]/loaders["validation"] needs no further changes;
    # "train" is always loaded too (used for e.g. dataset sizes reported below).
    datasets = {
        "train": Nist302RegisteredDataset(
            manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
            exemplar_roots=roots, split="train", output_shape=shape, geometric_confidence_weights=confidence_weights,
        ),
        "validation": Nist302RegisteredDataset(
            manifest_path=args.manifest, latent_root=args.latent_root, annotation_root=args.annotation_root,
            exemplar_roots=roots, split=requested_split, output_shape=shape, geometric_confidence_weights=confidence_weights,
        ),
    }
    training = config["training"]
    model_type = str(config["model"].get("type", "global"))
    if model_type not in {"global", "spatial"}:
        raise ValueError(f"unknown model type: {model_type!r}")
    evaluation_checkpoint = None
    if args.evaluate_checkpoint is not None:
        evaluation_checkpoint = torch.load(
            args.evaluate_checkpoint, map_location=device, weights_only=False
        )
        checkpoint_model_type = str(evaluation_checkpoint.get("model_type", "global"))
        if checkpoint_model_type != model_type:
            raise ValueError(
                "evaluation checkpoint model type differs from the configuration: "
                f"{checkpoint_model_type!r} != {model_type!r}"
            )
        channels = tuple(int(value) for value in evaluation_checkpoint["channels_used"])
        latent_size = int(evaluation_checkpoint["latent_dim"])
        epochs, batch_size = 0, int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "evaluation_only"
    elif args.smoke_test:
        channels, latent_size = (8, 16, 32), (4 if model_type == "global" else 1)
        epochs, batch_size = 1, 2
        train_batches, validation_batches, mode = 2, 1, "smoke_test"
    elif args.pilot:
        channels = tuple(int(value) for value in training["pilot_channels"])
        latent_size = int(training["pilot_latent_dim"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches = int(training["pilot_train_batches"])
        validation_batches = int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        channels = tuple(int(value) for value in config["model"]["channels"])
        latent_size = int(config["model"]["latent_dim"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "full"
    if args.run_epochs is not None:
        if args.run_epochs <= 0:
            raise ValueError("run-epochs must be positive")
        epochs = min(epochs, args.run_epochs)
    data_generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config["data"]["num_workers"]),
            generator=data_generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    upsampling_mode = str(
        args.decoder_upsampling_mode
        or config["model"].get("upsampling_mode", "nearest")
    )
    model = (
        SpatialConditionalVAE(
            channels=channels,
            latent_channels=latent_size,
            upsampling_mode=upsampling_mode,
        )
        if model_type == "spatial"
        else ConditionalVAE(
            channels=channels,
            latent_dim=latent_size,
            upsampling_mode=upsampling_mode,
        )
    ).to(device)
    initialize_from = config["model"].get("initialize_from")
    initialized_from: str | None = None
    if initialize_from:
        if model_type != "global":
            raise ValueError("spatial CVAE cannot silently accept a global checkpoint")
        initialization = torch.load(
            Path(initialize_from), map_location=device, weights_only=False
        )
        if tuple(initialization["channels_used"]) != channels or int(
            initialization["latent_dim"]
        ) != latent_size:
            raise ValueError("initialization checkpoint shape differs from configuration")
        model.load_state_dict(initialization["model_state"])
        initialized_from = str(initialize_from)
    support_path = config["model"].get("support_checkpoint")
    support_model = load_support_predictor(
        Path(support_path) if support_path else None, device
    )
    loss_config = dict(config["loss"])
    ridge_report_path = config["data"].get("ridge_band_report")
    ridge_report = None
    if ridge_report_path:
        ridge_report = json.loads(Path(ridge_report_path).read_text(encoding="utf-8"))
        if ridge_report["split"] != "train" or ridge_report.get("test_accessed") is not False:
            raise ValueError("ridge frequencies must be estimated on train with sealed test")
        loss_config["ridge_band_frequencies"] = tuple(
            float(value) for value in ridge_report["recommended_gabor_frequencies"]
        )
    objective = RegisteredApproximateCVAELoss(**loss_config).to(device)
    optimizer = None
    if evaluation_checkpoint is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "mode": mode,
        "test_loaded": requested_split == "test",
        "model_type": model_type,
        "latent_parameter": latent_size,
        "decoder_upsampling_mode": upsampling_mode,
        "initialization": initialized_from,
        "evaluation_checkpoint": (
            str(args.evaluate_checkpoint) if args.evaluate_checkpoint else None
        ),
        "support_checkpoint": str(support_path) if support_path else None,
        "support_target_semantics": VISIBLE_LATENT_SUPPORT if support_model else None,
        "registered_target": "registered_approximate (not pixel ground truth)",
        "confidence_calibration": confidence_report,
        "ridge_band_calibration": ridge_report,
        "loss": loss_config,
        "provenance": code_provenance(),
    }
    (args.output / "method-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    history: list[dict[str, object]] = []
    best_loss, best_epoch, stale = math.inf, -1, 0
    if evaluation_checkpoint is not None:
        model.load_state_dict(evaluation_checkpoint["model_state"])
        best_loss = float(evaluation_checkpoint.get("validation_loss", math.nan))
        best_epoch = int(evaluation_checkpoint.get("epoch", -1))
    started = time.perf_counter()
    for epoch in range(epochs):
        beta = float(training["beta_max"]) * min(
            1.0, (epoch + 1) / int(training["kl_warmup_epochs"])
        )
        train_metrics = run_epoch(
            model=model,
            support_model=support_model,
            loader=loaders["train"],
            objective=objective,
            device=device,
            beta=beta,
            optimizer=optimizer,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            max_batches=train_batches,
        )
        validation_metrics = run_epoch(
            model=model,
            support_model=support_model,
            loader=loaders["validation"],
            objective=objective,
            device=device,
            beta=beta,
            optimizer=None,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            max_batches=validation_batches,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if best_loss - validation_metrics["loss"] > float(
            training.get("minimum_improvement", 1e-5)
        ):
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch + 1, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "channels_used": list(channels),
                    "model_type": model_type,
                    "latent_dim": latent_size,
                    "epoch": best_epoch,
                    "validation_loss": best_loss,
                    "target_semantics": "registered_approximate",
                    "support_target_semantics": (
                        VISIBLE_LATENT_SUPPORT if support_model else None
                    ),
                },
                args.output / "best-checkpoint.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch + 1}/{epochs} beta={beta:.4f} "
            f"train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"kl={validation_metrics['kl']:.5f}",
            flush=True,
        )
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    checkpoint = evaluation_checkpoint
    if checkpoint is None:
        checkpoint = torch.load(
            args.output / "best-checkpoint.pt", map_location=device, weights_only=False
        )
    model.load_state_dict(checkpoint["model_state"])
    evaluation = config["evaluation"]
    selected_support_mode = str(args.support_mode or evaluation["support_mode"])
    selected_support_threshold = float(
        args.support_threshold
        if args.support_threshold is not None
        else evaluation["support_threshold"]
    )
    selected_feather_sigma = float(
        args.support_feather_sigma
        if args.support_feather_sigma is not None
        else evaluation.get("support_feather_sigma", 1.5)
    )
    k_values = tuple(
        int(value)
        for value in (
            evaluation["pilot_k_values"]
            if mode in {"smoke_test", "pilot"}
            else evaluation["k_values"]
        )
    )
    records = probabilistic_evaluation(
        model,
        support_model,
        loaders["validation"],
        device,
        k_values=k_values,
        nominal_coverages=tuple(float(value) for value in evaluation["nominal_coverages"]),
        sample_chunk_size=int(evaluation["sample_chunk_size"]),
        evaluation_seed=int(evaluation["seed"]),
        support_mode=selected_support_mode,
        support_threshold=selected_support_threshold,
        support_feather_sigma=selected_feather_sigma,
        max_batches=validation_batches,
    )
    with (args.output / "per-image-probabilistic-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    preview_k = min(max(k_values), int(evaluation.get("preview_samples", 10)))
    save_preview(
        model,
        support_model,
        next(iter(loaders["validation"])),
        device,
        args.output / "samples-preview.png",
        samples_k=preview_k,
        sample_chunk_size=int(evaluation["sample_chunk_size"]),
        evaluation_seed=int(evaluation["seed"]),
        support_mode=selected_support_mode,
        support_threshold=selected_support_threshold,
        support_feather_sigma=selected_feather_sigma,
        title=f"NIST302 {model_type} CVAE {mode}: K={preview_k}, epoch {best_epoch}",
    )
    example_observed = next(iter(loaders["validation"]))["observed"][:1].to(device)
    report = {
        "mode": mode,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_type": model_type,
        "latent_parameter": latent_size,
        "latent_scalar_count": model.latent_scalar_count(example_observed),
        "kl_reduction": "mean per latent scalar; kl_total_nats also reported",
        "k_values": list(k_values),
        "evaluation_seed": int(evaluation["seed"]),
        "support_composition": {
            "mode": selected_support_mode,
            "threshold": selected_support_threshold,
            "feather_sigma": selected_feather_sigma,
        },
        "history": history,
        "probabilistic_evaluation": summarize_records(records),
        "train_images": len(datasets["train"]),
        "validation_images": len(datasets["validation"]),
        "test_loaded": requested_split == "test",
        "seed_state": seed_state.as_dict(),
        "provenance": code_provenance(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["probabilistic_evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
