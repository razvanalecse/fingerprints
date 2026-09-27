#!/usr/bin/env python3
"""Fine-tune a gated model on confidence-weighted SD302 approximate registrations."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.data.geometry_conditioned import TrustWeightedDataset
from fingerprint_reconstruction.losses import RegisteredApproximateLoss
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import (
    VISIBLE_LATENT_SUPPORT,
    FingerprintSupportPredictor,
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
    parser.add_argument("--run-epochs", type=int)
    return parser.parse_args()


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: RegisteredApproximateLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    max_batches: int | None,
    support_model: torch.nn.Module | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        evaluation_roi = batch["evaluation_roi"].to(device)
        confidence = batch["geometric_confidence"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            raw = model(torch.cat((observed, mask), dim=1))
            prediction = mask * observed + (1.0 - mask) * raw
            support_probability = None
            if support_model is not None:
                with torch.no_grad():
                    support_probability = torch.maximum(
                        support_model(torch.cat((observed, mask), dim=1)), mask
                    )
            result = loss_function(
                prediction,
                target,
                mask,
                observed=observed,
                evaluation_roi=evaluation_roi,
                geometric_confidence=confidence,
                support_probability=support_probability,
            )
            if not torch.isfinite(result.total):
                raise FloatingPointError(f"non-finite loss at batch {batch_index}")
            if training:
                result.total.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite gradient in {name}")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite total gradient norm")
                optimizer.step()
                totals["gradient_norm"] += float(gradient_norm.detach()) * len(observed)
        batch_size = len(observed)
        examples += batch_size
        totals["loss"] += float(result.total.detach()) * batch_size
        for name, value in result.components.items():
            totals[name] += float(value) * batch_size
    if examples == 0:
        raise RuntimeError("epoch contained no examples")
    return {name: value / examples for name, value in totals.items()} | {
        "examples": float(examples)
    }


@torch.no_grad()
def save_preview(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    output: Path,
    support_model: torch.nn.Module | None,
) -> None:
    model.eval()
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    target = batch["target"][:1].to(device)
    roi = batch["evaluation_roi"][:1].to(device)
    confidence = batch["geometric_confidence"][:1].to(device)
    raw = model(torch.cat((observed, mask), dim=1))
    reconstruction = mask * observed + (1.0 - mask) * raw
    support_probability = (
        torch.maximum(support_model(torch.cat((observed, mask), dim=1)), mask)
        if support_model is not None
        else torch.ones_like(mask)
    )
    support_constrained = mask * observed + (1.0 - mask) * (
        support_probability * raw + (1.0 - support_probability)
    )
    display_observed = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    panels = (
        batch["latent_image"][0, 0].numpy(),
        display_observed[0, 0].cpu().numpy(),
        target[0, 0].cpu().numpy(),
        reconstruction[0, 0].cpu().numpy(),
        support_probability[0, 0].cpu().numpy(),
        support_constrained[0, 0].cpu().numpy(),
        confidence[0, 0].cpu().numpy(),
        (torch.abs(reconstruction - target) * roi)[0, 0].cpu().numpy(),
    )
    titles = (
        "Real latent",
        "Y (quality>=2)",
        "Registered exemplar X_pseudo",
        "Fine-tuned reconstruction",
        "P(visible support | Y,M)",
        "Soft support-constrained",
        "Geometric confidence",
        "|reconstruction-X_pseudo| in ROI",
    )
    figure, axes = plt.subplots(1, 8, figsize=(25, 3.4))
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(
        "SD302 registered_approximate — pseudo-target is structural, not pixel-exact ground truth",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    confidence_report = json.loads(
        Path(config["data"]["geometric_confidence_report"]).read_text(encoding="utf-8")
    )
    ridge_report = json.loads(
        Path(config["data"]["ridge_band_report"]).read_text(encoding="utf-8")
    )
    if confidence_report["source_split"] != "validation":
        raise ValueError("geometric confidence must be calibrated on validation")
    if ridge_report["split"] != "train" or ridge_report.get("test_accessed") is not False:
        raise ValueError("ridge band must be estimated on training without test access")
    confidence_weights = tuple(float(value) for value in confidence_report["ordered_weights"])
    ridge_frequencies = tuple(
        float(value) for value in ridge_report["recommended_gabor_frequencies"]
    )
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    shape = tuple(int(value) for value in config["data"]["image_size"])
    datasets = {
        split: Nist302RegisteredDataset(
            manifest_path=args.manifest,
            latent_root=args.latent_root,
            annotation_root=args.annotation_root,
            exemplar_roots=roots,
            split=split,
            output_shape=shape,
            geometric_confidence_weights=confidence_weights,
        )
        for split in ("train", "validation")
    }
    trust_cache = Path(config["data"]["pseudo_trust_cache"])
    datasets = {
        split: TrustWeightedDataset(dataset, trust_cache)
        for split, dataset in datasets.items()
    }
    training = config["training"]
    if args.smoke_test:
        epochs, batch_size, train_batches, validation_batches, mode = 1, 2, 2, 1, "smoke_test"
    elif args.pilot:
        epochs = int(training["pilot_epochs"])
        batch_size = int(training["batch_size"])
        train_batches = int(training["pilot_train_batches"])
        validation_batches = int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        epochs = int(training["epochs"])
        batch_size = int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "full"
    if args.run_epochs is not None:
        if args.run_epochs <= 0:
            raise ValueError("run-epochs must be positive")
        epochs = min(epochs, args.run_epochs)
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(config["data"]["num_workers"]),
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    initialization_path = Path(config["model"]["initialize_from"])
    initialization = torch.load(initialization_path, map_location=device, weights_only=False)
    channels = tuple(int(value) for value in initialization["channels_used"])
    if tuple(config["model"]["channels"]) != channels:
        raise ValueError("configured channels differ from initialization checkpoint")
    model = build_reconstruction_model(config["model"], channels=channels).to(device)
    model.load_state_dict(initialization["model_state"])
    support_model = None
    support_checkpoint_path = config["model"].get("support_checkpoint")
    if support_checkpoint_path:
        support_checkpoint = torch.load(
            Path(support_checkpoint_path), map_location=device, weights_only=False
        )
        if support_checkpoint.get("target_semantics") != VISIBLE_LATENT_SUPPORT:
            raise ValueError(
                "support checkpoint must predict official visible latent support"
            )
        support_model = FingerprintSupportPredictor(
            channels=tuple(support_checkpoint["channels_used"])
        ).to(device)
        support_model.load_state_dict(support_checkpoint["model_state"])
        support_model.eval()
        support_model.requires_grad_(False)
    loss_config = dict(config["loss"])
    loss_config["ridge_band_frequencies"] = ridge_frequencies
    loss_function = RegisteredApproximateLoss(**loss_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["lr_reduce_factor"]),
        patience=int(training["lr_reduce_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "mode": mode,
        "test_loaded": False,
        "initialization": str(initialization_path),
        "registered_target": "registered_approximate (not pixel ground truth)",
        "confidence_calibration": confidence_report,
        "ridge_band_calibration": ridge_report,
        "loss": loss_config,
        "support_checkpoint": str(support_checkpoint_path) if support_checkpoint_path else None,
        "support_target_semantics": (
            VISIBLE_LATENT_SUPPORT if support_model is not None else None
        ),
    }
    (args.output / "method-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    history = []
    best_loss = math.inf
    best_epoch = -1
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            max_batches=train_batches,
            support_model=support_model,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=loaders["validation"],
            loss_function=loss_function,
            device=device,
            optimizer=None,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            max_batches=validation_batches,
            support_model=support_model,
        )
        scheduler.step(validation_metrics["loss"])
        record = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "channels_used": list(channels),
                "epoch": epoch + 1,
                "validation_loss": validation_metrics["loss"],
                "validation_components": validation_metrics,
                "confidence_weights": list(confidence_weights),
                "ridge_band_frequencies": list(ridge_frequencies),
                "target_semantics": "registered_approximate",
            },
            args.output / f"checkpoint-epoch-{epoch + 1:03d}.pt",
        )
        improvement = best_loss - validation_metrics["loss"]
        if improvement > float(training["minimum_improvement"]):
            best_loss = validation_metrics["loss"]
            best_epoch = epoch + 1
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "channels_used": list(channels),
                    "epoch": best_epoch,
                    "validation_loss": best_loss,
                    "confidence_weights": list(confidence_weights),
                    "ridge_band_frequencies": list(ridge_frequencies),
                    "target_semantics": "registered_approximate",
                },
                args.output / "best-checkpoint.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"grad={train_metrics.get('gradient_norm', float('nan')):.4f}",
            flush=True,
        )
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    best = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    save_preview(
        model,
        next(iter(loaders["validation"])),
        device,
        args.output / "preview.png",
        support_model,
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    epoch_indices = [record["epoch"] for record in history]
    axis.plot(epoch_indices, [record["train"]["loss"] for record in history], marker="o", label="train")
    axis.plot(
        epoch_indices,
        [record["validation"]["loss"] for record in history],
        marker="o",
        label="validation",
    )
    axis.set(xlabel="Epoch", ylabel="Confidence-weighted composite loss", title="SD302 fine-tuning")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "training-curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    report = {
        "mode": mode,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "train_images": len(datasets["train"]),
        "validation_images": len(datasets["validation"]),
        "test_loaded": False,
        "seed_state": seed_state.as_dict(),
    }
    (args.output / "training-metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
