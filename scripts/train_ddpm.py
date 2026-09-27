#!/usr/bin/env python3
"""Train the explicit conditional pixel-space DDPM."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.diffusion import ConditionalDDPM, DDPMScheduler, DiffusionUNet
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, device, observed_weight, optimizer=None, max_batches=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, components = model.training_loss(
                target, observed, mask, observed_weight=observed_weight
            )
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite DDPM loss at batch {batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        totals["loss"] += float(loss.detach()) * size
        for key, value in components.items():
            totals[key] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty DDPM epoch")
    return {**{key: value / examples for key, value in totals.items()}, "examples": examples}


@torch.no_grad()
def save_preview(model, batch, device, output, *, samples_k, title, data_consistency):
    model.eval()
    target = batch["target"][:1].to(device)
    observed = batch["observed"][:1].to(device)
    mask = batch["mask"][:1].to(device)
    samples = model.sample(
        observed, mask, num_samples=samples_k, enforce_data_consistency=data_consistency
    )[0, :, 0]
    mean, std = samples.mean(0), samples.std(0, unbiased=True)
    observed_display = torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))
    panels = [target[0, 0], observed_display[0, 0], mean, std]
    labels = ["Target X", "Observed support", "Predictive mean", "Predictive std"]
    panels.extend(samples[index] for index in range(min(4, samples_k)))
    labels.extend(f"DDPM sample {index + 1}" for index in range(min(4, samples_k)))
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, panel, label in zip(axes.ravel(), panels, labels):
        axis.imshow(panel.cpu().numpy(), cmap="magma" if label == "Predictive std" else "gray", vmin=0)
        axis.set_title(label); axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(); figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    common = dict(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=int(config["experiment"]["seed"]),
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    training, diffusion = config["training"], config["diffusion"]
    if args.smoke_test:
        channels, time_dim, timesteps = (8, 16, 32), 32, 8
        epochs, batch_size, train_batches, validation_batches, mode = 1, 2, 2, 1, "smoke_test"
    elif args.pilot:
        channels = tuple(training["pilot_channels"]); time_dim = int(training["pilot_time_dim"])
        timesteps = int(diffusion["pilot_timesteps"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches, validation_batches = int(training["pilot_train_batches"]), int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        channels = tuple(config["model"]["channels"]); time_dim = int(config["model"]["time_dim"])
        timesteps = int(diffusion["timesteps"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None; mode = "full"
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    scheduler = DDPMScheduler(
        timesteps=timesteps,
        schedule=str(diffusion["schedule"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    )
    model = ConditionalDDPM(
        DiffusionUNet(channels=channels, time_dim=time_dim), scheduler
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, float(diffusion["observed_noise_loss_weight"]), optimizer, train_batches, float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, device, float(diffusion["observed_noise_loss_weight"]), None, validation_batches)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config, "channels_used": list(channels), "time_dim": time_dim, "timesteps": timesteps, "epoch": epoch, "validation_loss": best_loss}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f}", flush=True)
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True); break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_preview(
        model, next(iter(validation_loader)), device, args.output / "samples-preview.png",
        samples_k=int(config["evaluation"]["preview_samples"]),
        title=f"Conditional DDPM {mode}: T={timesteps}, best epoch {best_epoch + 1}",
        data_consistency=bool(diffusion["enforce_data_consistency"]),
    )
    report = {"mode": mode, "device": str(device), "timesteps": timesteps, "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "parameters": sum(parameter.numel() for parameter in model.parameters()), "history": history, "seed_state": seed_state.as_dict()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "timesteps", "elapsed_seconds", "best_epoch", "best_validation_loss", "parameters")}, indent=2))


if __name__ == "__main__":
    main()
