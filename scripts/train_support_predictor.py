#!/usr/bin/env python3
"""Train a leakage-safe fingerprint-support predictor from partial inputs."""

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
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor, support_loss
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


def run_epoch(model, loader, device, *, optimizer=None, max_batches=None, dice_weight=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "iou": 0.0}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        observation_mask = batch["mask"].to(device)
        target = batch["foreground"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            probability = model(torch.cat((observed, observation_mask), dim=1))
            loss, terms = support_loss(probability, target, dice_weight=dice_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        prediction = probability >= 0.5
        target_binary = target.bool()
        intersection = (prediction & target_binary).flatten(1).sum(1).float()
        union = (prediction | target_binary).flatten(1).sum(1).float().clamp_min(1)
        size = target.shape[0]
        values = {
            "loss": loss.detach(),
            "bce": terms["bce"].detach(),
            "dice_loss": terms["dice_loss"].detach(),
            "iou": (intersection / union).mean(),
        }
        for key, value in values.items():
            totals[key] += float(value) * size
        examples += size
    if not examples:
        raise ValueError("empty support-prediction epoch")
    return {key: value / examples for key, value in totals.items()}


@torch.no_grad()
def save_preview(model, batch, device, output, title):
    model.eval()
    observed = batch["observed"][:4].to(device)
    mask = batch["mask"][:4].to(device)
    target = batch["foreground"][:4].to(device)
    probability = model(torch.cat((observed, mask), dim=1))
    prediction = probability >= 0.5
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(observed.shape[0]):
        display = torch.where(mask[row].bool(), observed[row], torch.full_like(observed[row], 0.72))
        panels = (display[0], target[row, 0], probability[row, 0], prediction[row, 0])
        labels = ("Partial input", "Training target S", "P(S=1|Y,M)", "Predicted support")
        for axis, panel, label in zip(axes[row], panels, labels):
            axis.imshow(panel.cpu(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(label); axis.axis("off")
    figure.suptitle(title); figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight"); plt.close(figure)


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    seed_state = seed_everything(seed)
    device = select_device(args.device)
    common = dict(
        manifest_path=args.manifest,
        image_root=args.image_root,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=seed,
        include_foreground=True,
    )
    train_dataset = SocofingPartialDataset(split="train", **common)
    validation_dataset = SocofingPartialDataset(split="validation", **common)
    training = config["training"]
    channels = (8, 16) if args.smoke_test else tuple(config["model"]["channels"])
    epochs = 1 if args.smoke_test else int(training["epochs"])
    batch_size = 2 if args.smoke_test else int(training["batch_size"])
    max_train = 2 if args.smoke_test else None
    max_validation = 1 if args.smoke_test else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    model = FingerprintSupportPredictor(channels=channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, max_batches=max_train, dice_weight=float(training["dice_weight"]))
        validation_metrics = run_epoch(model, validation_loader, device, max_batches=max_validation, dice_weight=float(training["dice_weight"]))
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "channels_used": list(channels), "config": config, "epoch": epoch, "validation_loss": best_loss}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f} val_iou={validation_metrics['iou']:.4f}", flush=True)
        if not args.smoke_test and stale >= int(training["early_stopping_patience"]):
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_preview(model, next(iter(validation_loader)), device, args.output / "support-preview.png", f"Leakage-safe support predictor; best epoch {best_epoch + 1}")
    report = {"mode": "smoke_test" if args.smoke_test else "full", "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "model_parameters": sum(p.numel() for p in model.parameters()), "history": history, "seed_state": seed_state.as_dict(), "target_definition": "foreground extracted from complete training/validation images; never supplied at inference"}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss}, indent=2))


if __name__ == "__main__":
    main()
