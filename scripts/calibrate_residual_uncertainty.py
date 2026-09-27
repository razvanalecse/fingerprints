#!/usr/bin/env python3
"""Calibrate predictive intervals on validation and evaluate on unseen test subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, Subset

from fingerprint_reconstruction.data.torch_dataset import SocofingPartialDataset
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.reproducibility import seed_everything
from fingerprint_reconstruction.training.trainer import select_device
from evaluate_residual_latent import load_model
from preview_shape_constrained_latent import load_support_model


NOMINAL_LEVELS = (0.50, 0.80, 0.90)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples-k", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--calibration-images", type=int, default=30)
    parser.add_argument("--test-images", type=int, default=30)
    return parser.parse_args()


def make_dataset(split, config, args, seed):
    return SocofingPartialDataset(
        manifest_path=args.manifest, image_root=args.image_root, split=split,
        output_shape=tuple(config["data"]["image_size"]),
        families=[MaskFamily(value) for value in config["data"]["mask_families"]],
        observed_fractions=config["data"]["observed_fractions"], base_seed=seed,
        include_foreground=True,
    )


@torch.no_grad()
def collect(model, support_model, dataset, count, device, *, seed, args):
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, shuffle=False)
    cases = []
    for case_index, batch in enumerate(loader):
        target = batch["target"].to(device)
        observed, mask = batch["observed"].to(device), batch["mask"].to(device)
        foreground = batch["foreground"].to(device).bool()
        support = support_model(torch.cat((observed, mask), dim=1)) >= 0.5
        seed_everything(seed + 1_000_003 * case_index)
        samples = model.sample_ddim(
            observed, mask, inference_steps=args.steps, num_samples=args.samples_k,
            latent_data_consistency=True, pixel_projection_interval=5,
        )[0, :, 0]
        samples = torch.where(support[0, 0], samples, torch.ones_like(samples))
        samples = torch.where(mask[0, 0].bool(), observed[0, 0], samples)
        roi = (foreground[0, 0] & ~mask[0, 0].bool()).cpu().numpy()
        samples_np = samples.cpu().numpy()
        target_np = target[0, 0].cpu().numpy()
        mean = samples_np.mean(0)
        std = samples_np.std(0, ddof=1)
        cases.append({
            "sample_id": batch["sample_id"][0], "ratio": np.abs(mean[roi] - target_np[roi]) / np.maximum(std[roi], 1e-6),
            "std": std[roi], "target": target_np[roi], "mean": mean[roi],
            "samples": samples_np[:, roi],
        })
        if (case_index + 1) % 10 == 0 or case_index == 0:
            print(f"split={dataset.rows[0]['split']} case={case_index + 1}/{count}", flush=True)
    return cases


def evaluate_intervals(cases, multipliers):
    output = {}
    for nominal in NOMINAL_LEVELS:
        q = float(multipliers[str(nominal)])
        per_case_coverage, per_case_width = [], []
        empirical_coverage, empirical_width = [], []
        for case in cases:
            error = np.abs(case["mean"] - case["target"])
            half_width = q * case["std"]
            calibrated_lower = np.clip(case["mean"] - half_width, 0.0, 1.0)
            calibrated_upper = np.clip(case["mean"] + half_width, 0.0, 1.0)
            per_case_coverage.append(float(np.mean(
                (case["target"] >= calibrated_lower)
                & (case["target"] <= calibrated_upper)
            )))
            per_case_width.append(float(np.mean(calibrated_upper - calibrated_lower)))
            lower_q, upper_q = (1.0 - nominal) / 2.0, 1.0 - (1.0 - nominal) / 2.0
            lower = np.quantile(case["samples"], lower_q, axis=0)
            upper = np.quantile(case["samples"], upper_q, axis=0)
            empirical_coverage.append(float(np.mean((case["target"] >= lower) & (case["target"] <= upper))))
            empirical_width.append(float(np.mean(upper - lower)))
        output[str(nominal)] = {
            "multiplier": q,
            "calibrated_coverage_mean_across_images": float(np.mean(per_case_coverage)),
            "calibrated_coverage_sd_across_images": float(np.std(per_case_coverage, ddof=1)),
            "calibrated_interval_width": float(np.mean(per_case_width)),
            "empirical_quantile_coverage": float(np.mean(empirical_coverage)),
            "empirical_quantile_width": float(np.mean(empirical_width)),
        }
    return output


def main():
    args = parse_args()
    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device)
    support_model = load_support_model(args.support_checkpoint, device)
    seed = int(config["experiment"]["seed"])
    calibration_dataset = make_dataset("validation", config, args, seed)
    test_dataset = make_dataset("test", config, args, seed)
    started = time.perf_counter()
    calibration = collect(model, support_model, calibration_dataset, args.calibration_images, device, seed=seed, args=args)
    pooled_ratios = np.concatenate([case["ratio"] for case in calibration])
    learned = {str(level): float(np.quantile(pooled_ratios, level)) for level in NOMINAL_LEVELS}
    gaussian = {str(level): float(norm.ppf((1.0 + level) / 2.0)) for level in NOMINAL_LEVELS}
    test = collect(model, support_model, test_dataset, args.test_images, device, seed=seed + 97_003, args=args)
    report = {
        "calibration_split": "validation subjects", "evaluation_split": "test subjects",
        "calibration_images": len(calibration), "test_images": len(test),
        "samples_k": args.samples_k, "steps": args.steps,
        "learned_multipliers": learned, "gaussian_reference_multipliers": gaussian,
        "validation_calibrated": evaluate_intervals(calibration, learned),
        "test_calibrated": evaluate_intervals(test, learned),
        "test_gaussian_reference": evaluate_intervals(test, gaussian),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "pointwise pixel intervals; calibration targets are never supplied to the reconstruction model",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    nominal = np.asarray(NOMINAL_LEVELS)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(nominal, nominal, "k--", label="Ideal")
    for section, label, marker in (
        ("test_gaussian_reference", "Gaussian reference", "o"),
        ("test_calibrated", "Validation-calibrated", "s"),
    ):
        coverage = [report[section][str(level)]["calibrated_coverage_mean_across_images"] for level in NOMINAL_LEVELS]
        axis.plot(nominal, coverage, marker=marker, label=label)
    empirical = [report["test_calibrated"][str(level)]["empirical_quantile_coverage"] for level in NOMINAL_LEVELS]
    axis.plot(nominal, empirical, marker="^", label=f"Raw empirical quantiles (K={args.samples_k})")
    axis.set(xlabel="Nominal coverage", ylabel="Observed test coverage", xlim=(0.45, 0.95), ylim=(0.2, 1.0), title="Predictive interval calibration on unseen test subjects")
    axis.grid(alpha=0.2); axis.legend(); figure.tight_layout()
    figure.savefig(args.output / "test-calibration.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
