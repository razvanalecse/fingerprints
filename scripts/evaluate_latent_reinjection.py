#!/usr/bin/env python3
"""Paired latent-DDIM ablation with identical cases and initial noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.data.partial_pairs import observed_fraction_report
from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite
from fingerprint_reconstruction.metrics import (
    empirical_interval_coverage,
    pairwise_diversity_mae,
    paired_ridge_frequency_error,
    region_image_metrics,
    uncertainty_error_spearman,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
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
    parser.add_argument("--samples-k", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=90)
    parser.add_argument("--pixel-projection-interval", type=int, default=0)
    parser.add_argument("--pixel-projection-strength", type=float, default=1.0)
    return parser.parse_args()


def evaluate_samples(samples, target, mask, foreground):
    samples_np = samples.cpu().numpy()
    target_np = target.cpu().numpy()
    mask_np = mask.cpu().numpy().astype(bool)
    roi = foreground.cpu().numpy().astype(bool) & ~mask_np
    mean = samples_np.mean(0)
    std = samples_np.std(0, ddof=1)
    sample_mae = np.mean(np.abs(samples_np[:, roi] - target_np[roi]), axis=1)
    best_index = int(np.argmin(sample_mae))
    outputs = {"single": samples_np[0], "mean": mean, "best_of_k": samples_np[best_index]}
    record = {
        "pairwise_diversity_mae": pairwise_diversity_mae(samples_np, roi),
        "mean_predictive_std": float(std[roi].mean()),
        "uncertainty_error_spearman": uncertainty_error_spearman(samples_np, target_np, roi),
        "best_sample_index": float(best_index),
        "missing_roi_pixels": float(roi.sum()),
    }
    for nominal in (0.5, 0.8, 0.9):
        coverage, width = empirical_interval_coverage(samples_np, target_np, roi, nominal_coverage=nominal)
        label = int(100 * nominal)
        record[f"coverage_{label}"] = coverage
        record[f"interval_width_{label}"] = width
    for label, output in outputs.items():
        metrics = region_image_metrics(target_np, output, roi)
        record.update({f"{label}_{name}": value for name, value in metrics.items()})
        record[f"{label}_orientation_error"] = safe_orientation_error(target_np, output, roi)
        frequency_metrics = paired_ridge_frequency_error(
            target_np, output, roi, minimum_region_fraction=0.45
        )
        record.update(
            {f"{label}_{name}": value for name, value in frequency_metrics.items()}
        )
    return record, mean, std


@torch.no_grad()
def main():
    args = parse_args()
    device = select_device(args.device)
    latent, config = load_latent_model(args.latent_checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    base_seed = int(config["experiment"]["seed"])
    seed_everything(base_seed)
    dataset = SocofingPartialDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        split="validation",
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"],
        base_seed=base_seed,
        include_foreground=True,
    )
    if not 2 <= args.num_images <= len(dataset):
        raise ValueError("num-images must be between 2 and validation-set size")
    indices = np.linspace(0, len(dataset) - 1, args.num_images).round().astype(int)
    if np.unique(indices).size != args.num_images:
        raise RuntimeError("selection contains duplicate cases")
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)
    records, previews = [], []
    total_seconds = {"plain": 0.0, "reinjected": 0.0}
    started_all = time.perf_counter()
    for case_index, batch in enumerate(loader):
        target = batch["target"][0, 0].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        foreground = batch["foreground"][0, 0].to(device).bool()
        probability = support_model(torch.cat((observed, mask), dim=1))
        support = probability >= 0.5
        row = {
            "sample_id": batch["sample_id"][0],
            "mask_family": batch["mask_family"][0],
            "observed_fraction": float(batch["observed_fraction"][0]),
            "selected_dataset_index": int(indices[case_index]),
            **observed_fraction_report(
                mask[0, 0].cpu().numpy(), foreground.cpu().numpy()
            ),
        }
        preview = None
        for mode, reinject in (("plain", False), ("reinjected", True)):
            case_seed = base_seed + 1_000_003 * case_index
            seed_everything(case_seed)
            started = time.perf_counter()
            samples = latent.sample_ddim(
                observed,
                mask,
                inference_steps=args.steps,
                num_samples=args.samples_k,
                enforce_data_consistency=True,
                latent_data_consistency=reinject,
                pixel_projection_interval=(
                    args.pixel_projection_interval if reinject else 0
                ),
                pixel_projection_strength=args.pixel_projection_strength,
            )[0, :, 0]
            total_seconds[mode] += time.perf_counter() - started
            samples = torch.where(support[0, 0], samples, torch.ones_like(samples))
            samples = torch.where(mask[0, 0].bool(), observed[0, 0], samples)
            metrics, mean, std = evaluate_samples(samples, target, mask[0, 0], foreground)
            row.update({f"{mode}_{name}": value for name, value in metrics.items()})
            if mode == "reinjected" and case_index == 0:
                preview = (target.cpu(), torch.where(mask.bool(), observed, torch.full_like(observed, 0.72))[0, 0].cpu(), probability[0, 0].cpu(), mean, std, samples[0].cpu())
        records.append(row)
        if preview is not None:
            previews.append(preview)
        if (case_index + 1) % 10 == 0 or case_index == 0:
            print(f"case={case_index + 1}/{args.num_images} sample={row['sample_id']} plain_mae={row['plain_mean_mae']:.4f} reinjected_mae={row['reinjected_mean_mae']:.4f}", flush=True)

    comparison_fields = {
        "mean_mae": False,
        "mean_psnr": True,
        "mean_ssim_map_mean": True,
        "mean_orientation_error": False,
        "single_mae": False,
        "best_of_k_mae": False,
        "uncertainty_error_spearman": True,
    }
    baseline = {name: [float(row[f"plain_{name}"]) for row in records] for name in comparison_fields}
    candidate = {name: [float(row[f"reinjected_{name}"]) for row in records] for name in comparison_fields}
    tests = paired_metric_suite(baseline, candidate, comparison_fields)
    means = {
        mode: {
            name: float(np.nanmean([float(row[f"{mode}_{name}"]) for row in records]))
            for name in comparison_fields
        }
        for mode in ("plain", "reinjected")
    }
    report = {
        "num_pairs": len(records),
        "case_selection": "evenly spaced validation cases; one per selected row",
        "steps": args.steps,
        "samples_k": args.samples_k,
        "identical_initial_noise_within_pair": True,
        "support_constraint": "predicted from Y,M; target foreground used only to define evaluation ROI",
        "reinjected_pixel_projection_interval": args.pixel_projection_interval,
        "pixel_projection_strength": args.pixel_projection_strength,
        "means": means,
        "paired_tests": tests,
        "sampling_seconds": total_seconds,
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired-per-image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = (("mean_mae", "Mean MAE", False), ("mean_ssim_map_mean", "Mean SSIM", True), ("mean_orientation_error", "Orientation error", False))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (field, title, _) in zip(axes, metrics):
        x = [float(row[f"plain_{field}"]) for row in records]
        y = [float(row[f"reinjected_{field}"]) for row in records]
        axis.scatter(x, y, s=18, alpha=0.55)
        bounds = (min(x + y), max(x + y)); axis.plot(bounds, bounds, "k--", linewidth=1)
        axis.set(xlabel="Plain latent DDIM", ylabel="Reinjected latent DDIM", title=title)
        axis.grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(args.output / "paired-scatter.png", dpi=180, bbox_inches="tight"); plt.close(figure)

    if previews:
        target, partial, probability, mean, std, sample = previews[0]
        panels = (target, partial, probability, mean, sample, std)
        labels = ("Target", "Partial", "Predicted support", "Reinjected mean", "Reinjected sample", "Predictive std")
        figure, axes = plt.subplots(1, 6, figsize=(18, 3))
        for index, (axis, panel, label) in enumerate(zip(axes, panels, labels)):
            axis.imshow(panel, cmap="magma" if index == 5 else "gray", vmin=0, vmax=None if index == 5 else 1)
            axis.set_title(label); axis.axis("off")
        figure.tight_layout(); figure.savefig(args.output / "preview.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
