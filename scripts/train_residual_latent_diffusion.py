#!/usr/bin/env python3
"""Train latent diffusion on target-minus-coarse fingerprint residuals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.diffusion import DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.models.latent_diffusion import (
    ResidualConditionalLatentDDPM,
    estimate_latent_scale,
    estimate_residual_scale,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from train_latent_diffusion import load_autoencoder, load_structure_predictor, run_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-epochs", type=int)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_coarse_predictor(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_reconstruction_model(
        checkpoint["config"]["model"], channels=tuple(checkpoint["channels_used"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval().requires_grad_(False)


@torch.no_grad()
def save_preview(model, batch, device, output, *, steps=25, samples_k=4):
    model.eval()
    target = batch["target"][:1].to(device)
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    coarse = model.predict_coarse(observed, mask)
    samples = model.sample_ddim(
        observed, mask, inference_steps=steps, num_samples=samples_k,
        latent_data_consistency=True, pixel_projection_interval=5,
    )[0, :, 0]
    mean, std = samples.mean(0), samples.std(0, unbiased=True)
    partial = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    panels = (target[0, 0], partial[0, 0], coarse[0, 0], mean, samples[0], std)
    labels = ("Target", "Partial", "Coarse predictor", "Residual mean", "Residual sample", "Predictive std")
    figure, axes = plt.subplots(1, 6, figsize=(18, 3))
    for index, (axis, panel, label) in enumerate(zip(axes, panels, labels)):
        axis.imshow(panel.cpu(), cmap="magma" if index == 5 else "gray", vmin=0, vmax=None if index == 5 else 1)
        axis.set_title(label); axis.axis("off")
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_state, device = seed_everything(seed), select_device(args.device)
    common = dict(
        manifest_path=args.manifest, image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"], base_seed=seed,
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    training, diffusion = config["training"], config["diffusion"]
    if args.smoke_test:
        channels, time_dim, timesteps = (8, 16, 32), 32, 8
        epochs, batch_size, train_batches, validation_batches = 1, 2, 2, 1
    else:
        channels = tuple(config["model"]["channels"])
        time_dim, timesteps = int(config["model"]["time_dim"]), int(diffusion["timesteps"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None
    if args.batch_size is not None:
        batch_size = args.batch_size
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    autoencoder = load_autoencoder(Path(config["autoencoder"]["checkpoint"]), device)
    coarse = load_coarse_predictor(Path(config["coarse_predictor"]["checkpoint"]), device)
    scale_batches = 2 if args.smoke_test else int(config["autoencoder"]["scale_batches"])
    latent_scale = estimate_latent_scale(autoencoder, train_loader, device, max_batches=scale_batches)
    residual_scale = estimate_residual_scale(
        autoencoder, coarse, train_loader, device,
        latent_scale=latent_scale, max_batches=scale_batches,
    )
    structure = load_structure_predictor(
        Path(config["structure_predictor"]["checkpoint"]), device
    )
    scheduler = DDPMScheduler(
        timesteps=timesteps, schedule=str(diffusion["schedule"]),
        beta_start=float(diffusion["beta_start"]), beta_end=float(diffusion["beta_end"]),
    )
    denoiser = DiffusionUNet(
        channels=channels, time_dim=time_dim,
        data_channels=autoencoder.latent_channels,
        condition_channels=2 * autoencoder.latent_channels,
        normalized_condition=True,
        multiscale_conditioning=True,
        auxiliary_condition_channels=4,
        middle_attention=bool(config["model"].get("middle_attention", True)),
        attention_heads=int(config["model"].get("attention_heads", 4)),
    )
    model = ResidualConditionalLatentDDPM(
        autoencoder, coarse, denoiser, scheduler,
        latent_scale=latent_scale, residual_scale=residual_scale,
        structure_predictor=structure,
    ).to(device)
    initialization = None
    if args.initialize_from is not None:
        initialization_checkpoint = torch.load(
            args.initialize_from, map_location=device, weights_only=False
        )
        model.denoiser.load_state_dict(initialization_checkpoint["denoiser_state"])
        initialization = str(args.initialize_from)
        print(f"initialized from {args.initialize_from}", flush=True)
    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(training["learning_rate"])
    )
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(), lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale, history = math.inf, -1, 0, []
    stop_epoch = min(epochs, args.run_epochs) if args.run_epochs else epochs
    started = time.perf_counter()
    for epoch in range(stop_epoch):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer, train_batches, float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, device, None, validation_batches)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({
                "denoiser_state": model.denoiser.state_dict(), "config": config,
                "autoencoder_checkpoint": config["autoencoder"]["checkpoint"],
                "coarse_checkpoint": config["coarse_predictor"]["checkpoint"],
                "channels_used": list(channels), "time_dim": time_dim,
                "timesteps": timesteps, "latent_scale": latent_scale,
                "residual_scale": residual_scale, "epoch": epoch,
                "validation_loss": best_loss,
            }, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f}", flush=True)
        if not args.smoke_test and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True); break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.denoiser.load_state_dict(checkpoint["denoiser_state"])
    preview_steps = min(int(config["evaluation"]["preview_ddim_steps"]), timesteps)
    save_preview(model, next(iter(validation_loader)), device, args.output / "samples-preview.png", steps=preview_steps)
    report = {
        "device": str(device), "elapsed_seconds": elapsed,
        "latent_scale": latent_scale, "residual_scale": residual_scale,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "denoiser_parameters": sum(p.numel() for p in model.denoiser.parameters()),
        "history": history, "seed_state": seed_state.as_dict(),
        "target": "scaled latent residual E(X)-E(X_coarse)",
        "initialization": initialization, "learning_rate": learning_rate,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("elapsed_seconds", "latent_scale", "residual_scale", "best_epoch", "best_validation_loss")}, indent=2))


if __name__ == "__main__":
    main()
