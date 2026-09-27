#!/usr/bin/env python3
"""Train and validate the spatial KL autoencoder used by latent diffusion."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.evaluation.evaluator import summarize_records
from fingerprint_reconstruction.losses.autoencoder import AutoencoderKLLoss
from fingerprint_reconstruction.metrics import region_image_metrics
from fingerprint_reconstruction.models.latent_diffusion import FingerprintAutoencoderKL
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask,
    estimate_orientation_field,
    orientation_error,
)
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


def run_epoch(model, loader, objective, device, optimizer=None, max_batches=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, distribution = model(target, sample_posterior=training)
            result = objective(reconstruction, target, distribution)
            if training:
                if not torch.isfinite(result.total):
                    raise FloatingPointError(f"non-finite autoencoder loss at batch {batch_index}")
                result.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        totals["loss"] += float(result.total.detach()) * size
        for name, value in result.components.items():
            totals[name] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty autoencoder epoch")
    return {**{name: value / examples for name, value in totals.items()}, "examples": examples}


@torch.no_grad()
def evaluate_codec(model, loader, device, max_batches=None):
    model.eval()
    records = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = batch["target"].to(device)
        reconstruction, _ = model(target, sample_posterior=False)
        target_np = target[:, 0].cpu().numpy()
        reconstruction_np = reconstruction[:, 0].cpu().numpy()
        for index in range(target.shape[0]):
            roi = estimate_foreground_mask(target_np[index])
            metrics = region_image_metrics(target_np[index], reconstruction_np[index], roi)
            try:
                orientation = orientation_error(
                    estimate_orientation_field(target_np[index]),
                    estimate_orientation_field(reconstruction_np[index], use_foreground_mask=False),
                    region_mask=roi,
                )
            except ValueError:
                orientation = float("nan")
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "mask_family": "full_image_codec",
                    "observed_fraction": 1.0,
                    **{f"roi_{name}": value for name, value in metrics.items()},
                    "roi_orientation_error": orientation,
                }
            )
    if not records:
        raise ValueError("codec evaluation produced no records")
    return records


@torch.no_grad()
def save_preview(model, batch, device, output, title):
    model.eval()
    target = batch["target"][:4].to(device)
    reconstruction, distribution = model(target, sample_posterior=False)
    error = torch.abs(reconstruction - target)
    figure, axes = plt.subplots(3, target.shape[0], figsize=(3 * target.shape[0], 8))
    for column in range(target.shape[0]):
        axes[0, column].imshow(target[column, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(reconstruction[column, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[2, column].imshow(error[column, 0].cpu(), cmap="magma", vmin=0, vmax=0.5)
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Target"); axes[1, 0].set_ylabel("Decoded"); axes[2, 0].set_ylabel("|Error|")
    figure.suptitle(f"{title}; latent {tuple(distribution.mean.shape[1:])}")
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
    training = config["training"]
    if args.smoke_test:
        channels, latent_channels = (8, 16), 2
        epochs, batch_size, train_batches, validation_batches, mode = 1, 4, 2, 1, "smoke_test"
    elif args.pilot:
        channels = tuple(config["model"]["channels"])
        latent_channels = int(config["model"]["latent_channels"])
        epochs, batch_size = int(training["pilot_epochs"]), int(training["batch_size"])
        train_batches = int(training["pilot_train_batches"])
        validation_batches = int(training["pilot_validation_batches"])
        mode = "pilot"
    else:
        channels = tuple(config["model"]["channels"])
        latent_channels = int(config["model"]["latent_channels"])
        epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
        train_batches = validation_batches = None
        mode = "full"
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model = FingerprintAutoencoderKL(channels=channels, latent_channels=latent_channels).to(device)
    objective = AutoencoderKLLoss(**config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, objective, device, optimizer, train_batches, float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, objective, device, None, validation_batches)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config, "channels_used": list(channels), "latent_channels": latent_channels, "epoch": epoch, "validation_loss": best_loss}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f} l1={validation_metrics['missing_l1']:.6f} kl={validation_metrics['kl']:.6f}", flush=True)
        if mode == "full" and stale >= int(training["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch + 1}", flush=True); break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    records = evaluate_codec(model, validation_loader, device, validation_batches)
    with (args.output / "per-image-codec-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = summarize_records(records)
    metrics = summary["metrics"]
    acceptance_config = config["evaluation"]["acceptance"]
    acceptance = {
        "roi_psnr": metrics["roi_psnr"]["mean"] >= float(acceptance_config["validation_roi_psnr_min"]),
        "roi_ssim": metrics["roi_ssim_map_mean"]["mean"] >= float(acceptance_config["validation_roi_ssim_min"]),
        "roi_orientation": metrics["roi_orientation_error"]["mean"] <= float(acceptance_config["validation_orientation_error_max"]),
    }
    acceptance["passed"] = all(acceptance.values())
    save_preview(model, next(iter(validation_loader)), device, args.output / "codec-preview.png", f"Fingerprint Autoencoder-KL {mode}, f={model.compression_factor}")
    report = {"mode": mode, "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters()), "compression_factor": model.compression_factor, "latent_channels": latent_channels, "history": history, "codec_evaluation": summary, "acceptance": acceptance, "seed_state": seed_state.as_dict()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"codec_metrics": {key: metrics[key]["mean"] for key in ("roi_mae", "roi_psnr", "roi_ssim_map_mean", "roi_orientation_error")}, "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
