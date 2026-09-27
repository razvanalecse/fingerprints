#!/usr/bin/env python3
"""Go/no-go evaluation of one latent model across observed fractions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fingerprint_reconstruction.analysis.socofing_eda import read_manifest
from fingerprint_reconstruction.data.partial_pairs import build_partial_pair
from fingerprint_reconstruction.metrics import region_image_metrics
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale
from fingerprint_reconstruction.preprocessing.orientation import estimate_foreground_mask
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from evaluate_ddim import safe_orientation_error
from preview_shape_constrained_latent import load_latent_model, load_support_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples-k", type=int, default=4)
    parser.add_argument("--latent-data-consistency", action="store_true")
    parser.add_argument("--mask-family", choices=[item.value for item in MaskFamily], default="random_rectangles")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = select_device(args.device)
    latent, config = load_latent_model(args.latent_checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    seed_everything(int(config["experiment"]["seed"]))
    validation_rows = [row for row in read_manifest(args.manifest) if row["split"] == "validation"]
    if not validation_rows:
        raise ValueError("no validation image")
    row = validation_rows[0]
    target_np = load_grayscale(
        args.image_root / row["relative_path"], output_shape=tuple(config["data"]["image_size"])
    )
    true_support = estimate_foreground_mask(target_np)
    family = MaskFamily(args.mask_family)
    fractions = (0.10, 0.30, 0.50, 0.80)
    results, visual_rows = [], []
    for fraction in fractions:
        pair = build_partial_pair(
            target_np,
            sample_id=row["sample_id"],
            family=family,
            observed_fraction=fraction,
            replicate=0,
            base_seed=int(config["experiment"]["seed"]),
        )
        observed = torch.from_numpy(pair.observed)[None, None].to(device)
        mask = torch.from_numpy(pair.mask.astype("float32"))[None, None].to(device)
        probability = support_model(torch.cat((observed, mask), dim=1))
        predicted_support = probability >= 0.5
        raw = latent.sample_ddim(
            observed, mask, inference_steps=args.steps, num_samples=args.samples_k,
            enforce_data_consistency=True,
            latent_data_consistency=args.latent_data_consistency,
        )[0, :, 0]
        shaped = torch.where(predicted_support[0, 0], raw, torch.ones_like(raw))
        shaped = torch.where(mask[0, 0].bool(), observed[0, 0], shaped)
        raw_np = raw.cpu().numpy()
        shaped_np = shaped.cpu().numpy()
        raw_mean = raw_np.mean(0)
        shaped_mean = shaped_np.mean(0)
        std = shaped_np.std(0, ddof=1)
        missing_roi = true_support & ~pair.mask.astype(bool)
        support_np = predicted_support[0, 0].cpu().numpy()
        intersection = np.logical_and(true_support, support_np).sum()
        union = np.logical_or(true_support, support_np).sum()
        raw_metrics = region_image_metrics(target_np, raw_mean, missing_roi)
        shaped_metrics = region_image_metrics(target_np, shaped_mean, missing_roi)
        result = {
            "observed_fraction": fraction,
            "actual_observed_fraction": float(pair.mask.mean()),
            "missing_roi_pixels": int(missing_roi.sum()),
            "support_iou": float(intersection / max(union, 1)),
            "raw_mean": raw_metrics,
            "shape_constrained_mean": shaped_metrics,
            "raw_orientation_error": safe_orientation_error(target_np, raw_mean, missing_roi),
            "shape_constrained_orientation_error": safe_orientation_error(target_np, shaped_mean, missing_roi),
            "mean_predictive_std_missing_roi": float(std[missing_roi].mean()),
            "max_observed_consistency_error": float(np.max(np.abs(shaped_np[:, pair.mask.astype(bool)] - pair.observed[pair.mask.astype(bool)]))),
        }
        results.append(result)
        partial_display = np.where(pair.mask.astype(bool), pair.observed, 0.72)
        visual_rows.append((target_np, partial_display, probability[0, 0].cpu().numpy(), shaped_mean, shaped_np[0], std))
        print(json.dumps(result, sort_keys=True), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_id": row["sample_id"],
        "mask_family": family.value,
        "inference_steps": args.steps,
        "samples_k": args.samples_k,
        "latent_data_consistency": args.latent_data_consistency,
        "no_go_rule": "redesign conditioning if fidelity does not materially improve from r=0.10 to r=0.80",
        "results": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    labels = ("Target", "Partial input", "Predicted P(S=1)", "Constrained mean", "Constrained sample", "Predictive std")
    figure, axes = plt.subplots(len(fractions), len(labels), figsize=(16, 11))
    for row_index, (fraction, panels) in enumerate(zip(fractions, visual_rows)):
        for column, (axis, panel, label) in enumerate(zip(axes[row_index], panels, labels)):
            axis.imshow(panel, cmap="magma" if column == 5 else "gray", vmin=0, vmax=None if column == 5 else 1)
            axis.set_title(f"r={fraction:.1f} | {label}")
            axis.axis("off")
    figure.suptitle("Same fingerprint, same model, increasing observed information")
    figure.tight_layout(); figure.savefig(args.output / "observed-fraction-grid.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = [item["observed_fraction"] for item in results]
    axes[0].plot(x, [item["shape_constrained_mean"]["mae"] for item in results], marker="o")
    axes[0].set(xlabel="Observed fraction r", ylabel="MAE (missing ROI)", title="Ground-truth fidelity")
    axes[1].plot(x, [item["shape_constrained_orientation_error"] for item in results], marker="o")
    axes[1].set(xlabel="Observed fraction r", ylabel="Circular orientation error", title="Structural fidelity")
    axes[2].plot(x, [item["mean_predictive_std_missing_roi"] for item in results], marker="o")
    axes[2].set(xlabel="Observed fraction r", ylabel="Mean predictive std", title="Predictive uncertainty")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(args.output / "observed-fraction-curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
