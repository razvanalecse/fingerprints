#!/usr/bin/env python3
"""Train a leakage-safe coarse support/orientation/coherence predictor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.models.structure import FingerprintStructurePredictor, structure_loss
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
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, device, *, optimizer=None, max_batches=None, loss_options=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "support_bce", "support_dice", "orientation", "coherence", "support_iou", "circular_error")}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        target = batch["structure"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(observed, mask)
            loss, terms = structure_loss(prediction, target, **(loss_options or {}))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
        support = target[:, :1]
        predicted_support = prediction[:, :1] >= 0.5
        target_support = support >= 0.5
        intersection = (predicted_support & target_support).flatten(1).sum(1).float()
        union = (predicted_support | target_support).flatten(1).sum(1).float().clamp_min(1)
        dot = prediction[:, 1:2] * target[:, 1:2] + prediction[:, 2:3] * target[:, 2:3]
        weights = target[:, 4:5] * target[:, 3:4]
        circular_error = ((1.0 - dot.clamp(-1, 1)) * weights).sum() / weights.sum().clamp_min(1e-6)
        values = {"loss": loss.detach(), **{k: v.detach() for k, v in terms.items()}, "support_iou": (intersection / union).mean().detach(), "circular_error": circular_error.detach()}
        size = target.shape[0]
        for key, value in values.items():
            totals[key] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty structure-prediction epoch")
    return {key: value / examples for key, value in totals.items()}


def _draw_orientation(axis, support, x_vector, y_vector, coherence, title):
    axis.imshow(support, cmap="gray", vmin=0, vmax=1)
    height, width = support.shape
    for row in range(1, height, 2):
        for column in range(1, width, 2):
            if support[row, column] < 0.50 or coherence[row, column] < 0.20:
                continue
            theta = 0.5 * np.arctan2(y_vector[row, column], x_vector[row, column])
            length = 0.75
            dx, dy = length * np.cos(theta), length * np.sin(theta)
            axis.plot((column - dx, column + dx), (row - dy, row + dy), color="#00d4ff", linewidth=0.75)
    axis.set_title(title)
    axis.axis("off")


@torch.no_grad()
def save_preview(model, batch, device, output, title):
    model.eval()
    observed, mask = batch["observed"][:4].to(device), batch["mask"][:4].to(device)
    target = batch["structure"][:4].cpu().numpy()
    prediction = model(observed, mask).cpu().numpy()
    figure, axes = plt.subplots(4, 5, figsize=(13, 10))
    for row in range(observed.shape[0]):
        display = torch.where(mask[row].bool(), observed[row], torch.full_like(observed[row], 0.72))[0].cpu()
        axes[row, 0].imshow(display, cmap="gray", vmin=0, vmax=1); axes[row, 0].set_title("Partial input"); axes[row, 0].axis("off")
        _draw_orientation(axes[row, 1], target[row, 0], target[row, 1], target[row, 2], target[row, 3] * target[row, 4], "Target structure")
        _draw_orientation(axes[row, 2], prediction[row, 0], prediction[row, 1], prediction[row, 2], prediction[row, 3], "Predicted structure")
        axes[row, 3].imshow(np.abs(prediction[row, 0] - target[row, 0]), cmap="magma", vmin=0, vmax=1); axes[row, 3].set_title("Support error"); axes[row, 3].axis("off")
        dot = np.clip(prediction[row, 1] * target[row, 1] + prediction[row, 2] * target[row, 2], -1, 1)
        orientation_error = (1.0 - dot) * target[row, 4]
        axes[row, 4].imshow(orientation_error, cmap="magma", vmin=0, vmax=2); axes[row, 4].set_title("Axial orientation error"); axes[row, 4].axis("off")
    figure.suptitle(title); figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


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
        include_structure=True, structure_shape=tuple(config["data"]["structure_size"]),
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    training = config["training"]
    channels = (8, 16, 24, 32) if args.smoke_test else tuple(config["model"]["channels"])
    epochs, batch_size = (1, 2) if args.smoke_test else (int(training["epochs"]), int(training["batch_size"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model = FingerprintStructurePredictor(channels=channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    options = {"orientation_weight": float(training["orientation_weight"]), "coherence_weight": float(training["coherence_weight"])}
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, max_batches=2 if args.smoke_test else None, loss_options=options, gradient_clip=float(training["gradient_clip_norm"]))
        validation_metrics = run_epoch(model, validation_loader, device, max_batches=1 if args.smoke_test else None, loss_options=options)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "channels_used": list(channels), "config": config, "epoch": epoch, "validation_loss": best_loss}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f} IoU={validation_metrics['support_iou']:.4f} orient={validation_metrics['circular_error']:.4f}", flush=True)
        if not args.smoke_test and stale >= int(training["early_stopping_patience"]):
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_preview(model, next(iter(validation_loader)), device, args.output / "structure-preview.png", f"Leakage-safe structure predictor; best epoch {best_epoch + 1}")
    report = {"mode": "smoke_test" if args.smoke_test else "full", "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "model_parameters": sum(p.numel() for p in model.parameters()), "history": history, "seed_state": seed_state.as_dict(), "target_definition": "support/orientation/coherence derived from complete image only as a supervised target; inference receives only Y and M"}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss}, indent=2))


if __name__ == "__main__":
    main()
