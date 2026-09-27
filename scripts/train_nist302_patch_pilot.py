#!/usr/bin/env python3
"""Pilot: does training on native-resolution patches reduce the ridge-topology artifact?

Small, same-day pilot for the "native-resolution patches" item scoped (not
built end-to-end) in docs/nist302_native_resolution_feasibility.md. Trains a
single deterministic gated-conv model directly on
Sd302SyntheticPatchDataset (exact ground truth, no whole-image resize), then
the companion evaluate_nist302_patch_pilot.py script compares its
ridge-topology metrics (section 15's diagnostic) against the same
whole-image-trained checkpoint used throughout this project. Not a
replacement for the full multi-day patch pipeline: no PPI canonicalization,
no confidence-weighted registration loss (Track E has exact ground truth
instead), no overlap-add stitching.
"""

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
from torch.utils.data import DataLoader

from fingerprint_reconstruction.data.nist302_patch_dataset import Sd302SyntheticPatchDataset
from fingerprint_reconstruction.losses.reconstruction import MaskedReconstructionLoss
from fingerprint_reconstruction.models.factory import build_reconstruction_model
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-manifest", type=Path, required=True)
    parser.add_argument("--registered-manifest", type=Path, required=True)
    parser.add_argument("--sd302a-root", type=Path, required=True)
    parser.add_argument("--sd302b-root", type=Path, required=True)
    parser.add_argument("--sd302d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-train-images", type=int, default=1200)
    parser.add_argument("--max-validation-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_epoch(model, objective, loader, device, *, optimizer, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total, examples = 0.0, 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model.reconstruct(observed, mask)
            result = objective(prediction, target, mask)
            if training:
                if not torch.isfinite(result.total):
                    raise FloatingPointError(f"non-finite patch-pilot loss at batch {batch_index}")
                result.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        size = target.shape[0]
        total += float(result.total.detach()) * size
        examples += size
    if not examples:
        raise ValueError("empty patch-pilot epoch")
    return {"loss": total / examples, "examples": float(examples)}


@torch.no_grad()
def save_preview(model, batch, device, output, title):
    model.eval()
    observed = batch["observed"][:4].to(device)
    mask = batch["mask"][:4].to(device)
    target = batch["target"][:4].to(device)
    prediction = model.reconstruct(observed, mask)
    figure, axes = plt.subplots(3, observed.shape[0], figsize=(3 * observed.shape[0], 8))
    for column in range(observed.shape[0]):
        display = torch.where(mask[column].bool(), observed[column], torch.full_like(observed[column], 0.72))
        axes[0, column].imshow(display[0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(prediction[column, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[2, column].imshow(target[column, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Partial")
    axes[1, 0].set_ylabel("Reconstruction")
    axes[2, 0].set_ylabel("Target (exact)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_state = seed_everything(args.seed)
    device = select_device(args.device)
    roots = {"sd302a": args.sd302a_root, "sd302b": args.sd302b_root, "sd302d": args.sd302d_root}
    train_dataset = Sd302SyntheticPatchDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="train", patch_size=args.patch_size, base_seed=args.seed,
        max_images=(6 if args.smoke_test else args.max_train_images),
    )
    validation_dataset = Sd302SyntheticPatchDataset(
        exemplar_manifest_path=args.exemplar_manifest, registered_manifest_path=args.registered_manifest,
        exemplar_roots=roots, split="validation", patch_size=args.patch_size, base_seed=args.seed,
        max_images=(4 if args.smoke_test else args.max_validation_images),
    )
    batch_size = 2 if args.smoke_test else args.batch_size
    epochs = 1 if args.smoke_test else args.epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    channels = (16, 32, 64, 128) if args.smoke_test else (32, 64, 128, 256)
    model = build_reconstruction_model({"architecture": "gated_conv", "channels": channels}).to(device)
    objective = MaskedReconstructionLoss(
        missing_l1_weight=0.25, missing_mse_weight=0.05, orientation_weight=0.35, orientation_window=9,
        gradient_weight=0.10, ridge_energy_weight=0.10, ridge_band_weight=0.25, ridge_spectrum_weight=0.50,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    args.output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        train_metrics = run_epoch(model, objective, train_loader, device, optimizer=optimizer, max_batches=(2 if args.smoke_test else None))
        validation_metrics = run_epoch(model, objective, validation_loader, device, optimizer=None, max_batches=(1 if args.smoke_test else None))
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_metrics["loss"], epoch + 1, 0
            torch.save({"model_state": model.state_dict(), "config": {"model": {"architecture": "gated_conv", "channels": list(channels)}}, "channels_used": list(channels), "epoch": best_epoch, "validation_loss": best_loss, "test_loaded": False, "training_target": "Track E patches, native resolution, exact ground truth"}, args.output / "best-checkpoint.pt")
        else:
            stale += 1
        print(f"epoch={epoch + 1}/{epochs} train={train_metrics['loss']:.5f} val={validation_metrics['loss']:.5f}", flush=True)
        if not args.smoke_test and stale >= 4:
            print(f"early stopping at epoch {epoch + 1}", flush=True)
            break
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best-checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    save_preview(model, next(iter(validation_loader)), device, args.output / "preview.png", f"NIST302 patch pilot, patch_size={args.patch_size}")
    report = {"device": str(device), "elapsed_seconds": elapsed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "parameters": sum(p.numel() for p in model.parameters()), "patch_size": args.patch_size, "train_images": len(train_dataset), "validation_images": len(validation_dataset), "history": history, "test_loaded": False, "seed_state": seed_state.as_dict()}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_loss": best_loss}, indent=2))


if __name__ == "__main__":
    main()
