#!/usr/bin/env python3
"""Train leakage-safe anatomical support prediction for registered SD302 pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data import Nist302RegisteredDataset
from fingerprint_reconstruction.models.support import FingerprintSupportPredictor, support_loss
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def arguments() -> argparse.Namespace:
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
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def epoch(model, loader, device, *, optimizer=None, max_batches=None, dice_weight=1.0, clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "bce", "dice_loss", "iou", "observed_recall")}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        # Official EFS quality>=1 defines the annotated latent-print support.
        # It is distinct from the larger, ambiguous full-print extrapolation
        # support obtained by warping an exemplar.
        target = (batch["quality"].to(device) >= 1).to(observed.dtype)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            probability = torch.maximum(model(torch.cat((observed, mask), dim=1)), mask)
            loss, terms = support_loss(probability, target, dice_weight=dice_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
                optimizer.step()
        predicted = probability >= 0.5
        truth = target >= 0.5
        intersection = (predicted & truth).flatten(1).sum(1).float()
        union = (predicted | truth).flatten(1).sum(1).float().clamp_min(1)
        observed_count = mask.flatten(1).sum(1).clamp_min(1)
        observed_recall = ((predicted.float() * mask).flatten(1).sum(1) / observed_count).mean()
        size = len(observed)
        values = {
            "loss": loss.detach(),
            "bce": terms["bce"].detach(),
            "dice_loss": terms["dice_loss"].detach(),
            "iou": (intersection / union).mean(),
            "observed_recall": observed_recall,
        }
        for key, value in values.items():
            totals[key] += float(value) * size
        count += size
    return {key: value / count for key, value in totals.items()} | {"examples": float(count)}


@torch.no_grad()
def preview(model, batch, device, output):
    model.eval()
    observed = batch["observed"][:4].to(device)
    mask = batch["mask"][:4].to(device)
    target = (batch["quality"][:4].to(device) >= 1).to(observed.dtype)
    probability = torch.maximum(model(torch.cat((observed, mask), dim=1)), mask)
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(len(observed)):
        shown = torch.where(mask[row].bool(), observed[row], torch.full_like(observed[row], 0.72))
        panels = (shown[0], target[row, 0], probability[row, 0], probability[row, 0] >= 0.5)
        titles = ("Partial latent", "Official quality>=1 support", "P(support|Y,M)", "Threshold 0.5")
        for axis, panel, title in zip(axes[row], panels, titles):
            axis.imshow(panel.cpu(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
    figure.suptitle("Support is predicted only from partial latent Y and M")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_state = seed_everything(int(config["experiment"]["seed"]))
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    common = dict(
        manifest_path=args.manifest,
        latent_root=args.latent_root,
        annotation_root=args.annotation_root,
        exemplar_roots=roots,
        output_shape=tuple(config["data"]["image_size"]),
    )
    datasets = {split: Nist302RegisteredDataset(split=split, **common) for split in ("train", "validation")}
    training = config["training"]
    epochs = int(training["pilot_epochs"] if args.pilot else training["epochs"])
    max_train = int(training["pilot_train_batches"]) if args.pilot else None
    max_validation = int(training["pilot_validation_batches"]) if args.pilot else None
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=int(training["batch_size"]), shuffle=True, generator=generator, num_workers=0),
        "validation": DataLoader(datasets["validation"], batch_size=int(training["batch_size"]), shuffle=False, num_workers=0),
    }
    model = FingerprintSupportPredictor(channels=tuple(config["model"]["channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for index in range(epochs):
        train = epoch(model, loaders["train"], device, optimizer=optimizer, max_batches=max_train, dice_weight=float(training["dice_weight"]), clip=float(training["gradient_clip_norm"]))
        validation = epoch(model, loaders["validation"], device, max_batches=max_validation, dice_weight=float(training["dice_weight"]), clip=float(training["gradient_clip_norm"]))
        history.append({"epoch": index + 1, "train": train, "validation": validation})
        if validation["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation["loss"], index + 1, 0
            torch.save({"model_state": model.state_dict(), "channels_used": config["model"]["channels"], "config": config, "epoch": best_epoch, "validation_loss": best_loss, "target_semantics": "registered_approximate support"}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={index + 1}/{epochs} train={train['loss']:.5f} val={validation['loss']:.5f} IoU={validation['iou']:.4f}", flush=True)
        if not args.pilot and stale >= int(training["early_stopping_patience"]):
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    preview(model, next(iter(loaders["validation"])), device, args.output / "preview.png")
    report = {"mode": "pilot" if args.pilot else "full", "device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history, "train_images": len(datasets["train"]), "validation_images": len(datasets["validation"]), "test_loaded": False, "seed_state": seed_state.as_dict(), "target_definition": "official EFS quality>=1 latent-print support", "scope_warning": "This predicts the visible/annotated latent support, not the unknown full-print extent."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
