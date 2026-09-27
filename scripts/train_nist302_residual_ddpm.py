#!/usr/bin/env python3
"""Train a residual conditional DDPM on SD302: diffuses R = X_pseudo - X_coarse.

X_coarse is the frozen output of the best deterministic Gated fine-tune. The
denoiser is conditioned on [Y, M, visible-support probability, X_coarse] and
diffuses the residual in a [0,1]-remapped space so the existing scheduler /
DDIM / RePaint machinery in ConditionalDDPM is reused unchanged. This should
let the diffusion model spend its capacity on ridge texture instead of
re-deriving the smooth structure the deterministic model already recovers."""

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
from fingerprint_reconstruction.models.diffusion import (
    ConditionalDDPM,
    DDPMScheduler,
    DiffusionUNet,
)
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.support import apply_support_constraint
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_nist302_cvae import load_support_predictor, support_probability


def to_residual_space(image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    """Map an intensity image in [0,1] into a [0,1]-remapped residual around coarse."""

    return (0.5 + 0.5 * (image - coarse)).clamp(0.0, 1.0)


def from_residual_space(residual_image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    """Invert :func:`to_residual_space`."""

    return (coarse + 2.0 * (residual_image - 0.5)).clamp(0.0, 1.0)


@torch.no_grad()
def load_coarse_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


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
    return parser.parse_args()


def run_epoch(
    model: ConditionalDDPM,
    support_model: torch.nn.Module,
    coarse_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    observed_weight: float,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
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
        support = support_probability(support_model, observed, mask)
        with torch.no_grad():
            coarse = mask * observed + (1.0 - mask) * coarse_model(
                torch.cat((observed, mask), dim=1)
            )
        target_r = to_residual_space(target, coarse)
        observed_r = to_residual_space(observed, coarse)
        auxiliary = torch.cat((support, coarse), dim=1)
        # Supervise pseudo-target noise only where the registered exemplar is
        # evaluable, weighted by distance-to-correspondence confidence and the
        # leakage-safe visible-support predictor.
        missing_weight = roi * confidence * support
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, components = model.training_loss(
                target_r,
                observed_r,
                mask,
                observed_weight=observed_weight,
                missing_weight_map=missing_weight,
                auxiliary_condition=auxiliary,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite DDPM loss at batch {batch_index}")
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                )
                optimizer.step()
                totals["gradient_norm_before_clip"] += float(gradient_norm) * len(target)
        size = len(target)
        examples += size
        totals["loss"] += float(loss.detach()) * size
        totals["effective_missing_weight"] += float(missing_weight.mean()) * size
        for name, value in components.items():
            totals[name] += float(value) * size
    if not examples:
        raise RuntimeError("DDPM epoch contained no examples")
    return {name: value / examples for name, value in totals.items()} | {
        "examples": float(examples)
    }


@torch.no_grad()
def save_preview(
    model: ConditionalDDPM,
    support_model: torch.nn.Module,
    coarse_model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    output: Path,
    *,
    samples_k: int,
    ddim_steps: int,
    title: str,
    support_mode: str,
    support_threshold: float,
) -> None:
    model.eval()
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    latent = batch["latent_image"][:1].to(device)
    target = batch["target"][:1].to(device)
    support = support_probability(support_model, observed, mask)
    coarse = mask * observed + (1.0 - mask) * coarse_model(torch.cat((observed, mask), dim=1))
    observed_r = to_residual_space(observed, coarse)
    auxiliary = torch.cat((support, coarse), dim=1)
    samples_r = model.sample_ddim(
        observed_r,
        mask,
        inference_steps=ddim_steps,
        num_samples=samples_k,
        eta=0.0,
        enforce_data_consistency=True,
        auxiliary_condition=auxiliary,
    )[0]
    samples = from_residual_space(samples_r, coarse.expand_as(samples_r))
    samples = mask.expand_as(samples) * observed.expand_as(samples) + (
        1.0 - mask.expand_as(samples)
    ) * samples
    samples = apply_support_constraint(
        samples,
        observed.expand_as(samples),
        mask.expand_as(samples),
        support.expand_as(samples),
        mode=support_mode,
        threshold=support_threshold,
    )
    mean, std = samples.mean(0), samples.std(0)
    observed_display = torch.where(
        mask.bool(), observed, torch.full_like(observed, 0.72)
    )
    panels = [
        latent[0, 0],
        observed_display[0, 0],
        target[0, 0],
        coarse[0, 0],
        mean[0],
        samples[0, 0],
        std[0],
        support[0, 0],
    ]
    labels = [
        "Real latent",
        "Y,M",
        "X_pseudo",
        "X_coarse (frozen)",
        f"coarse+residual mean K={samples_k}",
        "Sample 1",
        "Predictive std",
        "P(visible support|Y,M)",
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel.cpu().numpy(), cmap="magma" if label == "Predictive std" else "gray")
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    figure.suptitle(title)
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
    roots = {
        "sd302a": args.sd302a_root,
        "sd302b": args.sd302b_root,
        "sd302d": args.sd302d_root,
    }
    datasets = {
        split: Nist302RegisteredDataset(
            manifest_path=args.manifest,
            latent_root=args.latent_root,
            annotation_root=args.annotation_root,
            exemplar_roots=roots,
            split=split,
            output_shape=tuple(config["data"]["image_size"]),
            geometric_confidence_weights=tuple(confidence_report["ordered_weights"]),
        )
        for split in ("train", "validation")
    }
    training, diffusion = config["training"], config["diffusion"]
    if args.smoke_test:
        channels, time_dim, timesteps = (8, 16, 32), 32, 20
        epochs, batch_size, train_batches, validation_batches = 1, 2, 2, 1
        mode, ddim_steps, preview_samples = "smoke_test", 10, 3
    elif args.pilot:
        channels = tuple(training["pilot_channels"])
        time_dim = int(training["pilot_time_dim"])
        timesteps = int(diffusion["pilot_timesteps"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches = int(training["pilot_train_batches"])
        validation_batches = int(training["pilot_validation_batches"])
        mode = "pilot"
        ddim_steps = int(config["evaluation"]["pilot_ddim_steps"])
        preview_samples = int(config["evaluation"]["preview_samples"])
    else:
        channels = tuple(config["model"]["channels"])
        time_dim = int(config["model"]["time_dim"])
        timesteps = int(diffusion["timesteps"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "full"
        ddim_steps = int(config["evaluation"]["ddim_steps"])
        preview_samples = int(config["evaluation"]["preview_samples"])
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            generator=generator, num_workers=int(config["data"]["num_workers"]),
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=0
        ),
    }
    scheduler = DDPMScheduler(
        timesteps=timesteps,
        schedule=str(diffusion["schedule"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=channels,
        time_dim=time_dim,
        multiscale_conditioning=bool(config["model"]["multiscale_conditioning"]),
        auxiliary_condition_channels=2,
        middle_attention=bool(config["model"]["middle_attention"]),
        attention_heads=int(config["model"]["attention_heads"]),
        upsampling_mode=str(config["model"]["upsampling_mode"]),
    )
    model = ConditionalDDPM(denoiser, scheduler).to(device)
    support_model = load_support_predictor(
        Path(config["model"]["support_checkpoint"]), device
    )
    coarse_model = load_coarse_model(Path(config["model"]["coarse_checkpoint"]), device)
    for parameter in coarse_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "mode": mode,
        "splits_loaded": ["train", "validation"],
        "test_loaded": False,
        "target": "registered_approximate (not pixel ground truth)",
        "missing_noise_weight": "evaluation_roi * geometric_confidence * visible_support_probability",
        "support_checkpoint": config["model"]["support_checkpoint"],
        "coarse_checkpoint": config["model"]["coarse_checkpoint"],
        "residual_target": "R = X_pseudo - X_coarse, remapped to [0,1] for the shared scheduler",
        "data_consistency": True,
        "decoder_upsampling": config["model"]["upsampling_mode"],
    }
    (args.output / "method-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model,
            support_model,
            coarse_model,
            loaders["train"],
            device,
            observed_weight=float(diffusion["observed_noise_loss_weight"]),
            optimizer=optimizer,
            max_batches=train_batches,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation_metrics = run_epoch(
            model,
            support_model,
            coarse_model,
            loaders["validation"],
            device,
            observed_weight=float(diffusion["observed_noise_loss_weight"]),
            optimizer=None,
            max_batches=validation_batches,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch + 1, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "channels_used": list(channels),
                    "time_dim": time_dim,
                    "timesteps": timesteps,
                    "epoch": best_epoch,
                    "validation_loss": best_loss,
                    "target_semantics": "registered_approximate",
                    "residual_target": True,
                    "coarse_checkpoint": config["model"]["coarse_checkpoint"],
                    "auxiliary_condition_channels": 2,
                    "test_loaded": False,
                },
                args.output / "best-checkpoint.pt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f}", flush=True
        )
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(
        args.output / "best-checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    save_preview(
        model,
        support_model,
        coarse_model,
        next(iter(loaders["validation"])),
        device,
        args.output / "samples-preview.png",
        samples_k=preview_samples,
        ddim_steps=ddim_steps,
        title=f"NIST302 conditional DDPM {mode}: T={timesteps}, DDIM={ddim_steps}",
        support_mode=str(config["evaluation"]["support_mode"]),
        support_threshold=float(config["evaluation"]["support_threshold"]),
    )
    report = {
        "mode": mode,
        "device": str(device),
        "timesteps": timesteps,
        "ddim_preview_steps": ddim_steps,
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "train_images": len(datasets["train"]),
        "validation_images": len(datasets["validation"]),
        "test_loaded": False,
        "seed_state": seed_state.as_dict(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("mode", "timesteps", "best_validation_loss", "parameters", "test_loaded")}, indent=2))


if __name__ == "__main__":
    main()
