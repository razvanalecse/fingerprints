#!/usr/bin/env python3
"""Plot stratified U-Net reconstruction metrics from a training report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Reconstruction model")
    return parser.parse_args()


def metric_series(groups: dict, metric: str):
    labels = list(groups)
    values = np.asarray([groups[label]["metrics"][metric]["mean"] for label in labels])
    lower = np.asarray([groups[label]["metrics"][metric]["ci95_low"] for label in labels])
    upper = np.asarray([groups[label]["metrics"][metric]["ci95_high"] for label in labels])
    return labels, values, np.vstack((values - lower, upper - values))


def main() -> None:
    args = parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    args.output.mkdir(parents=True, exist_ok=True)

    if report.get("history"):
        history = report["history"]
        epochs = [int(item["epoch"]) + 1 for item in history]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(epochs, [item["train"]["loss"] for item in history], label="train")
        axis.plot(
            epochs,
            [item["validation"]["loss"] for item in history],
            label="validation",
        )
        axis.set(
            xlabel="Epoch",
            ylabel="Composite loss",
            title=f"{args.label}: learning curve",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / "training-curve.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    by_fraction = evaluation["by_observed_fraction"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for axis, metric, ylabel in (
        (axes[0], "missing_roi_mae", "MAE on missing fingerprint ROI ↓"),
        (axes[1], "missing_roi_psnr", "PSNR on missing fingerprint ROI [dB] ↑"),
        (axes[2], "missing_roi_ssim_map_mean", "Local SSIM-map mean in missing ROI ↑"),
        (axes[3], "missing_roi_orientation_error", "Axial orientation error in missing ROI ↓"),
    ):
        labels, values, errors = metric_series(by_fraction, metric)
        ratios = np.asarray([float(label) for label in labels])
        axis.errorbar(ratios, values, yerr=errors, marker="o", capsize=3)
        axis.set(xlabel="Observed fraction r", ylabel=ylabel)
        axis.grid(alpha=0.25)
    figure.suptitle(f"{args.label}: performance versus observed information (mean ± 95% CI)")
    figure.tight_layout()
    figure.savefig(args.output / "performance-vs-observed-fraction.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    by_mask = evaluation["by_mask_family"]
    labels, mae, mae_error = metric_series(by_mask, "missing_roi_mae")
    _, ssim, ssim_error = metric_series(by_mask, "missing_roi_ssim_map_mean")
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(positions, mae, yerr=mae_error, capsize=3, color="#4477aa")
    axes[0].set(ylabel="Missing fingerprint-ROI MAE ↓")
    axes[1].bar(positions, ssim, yerr=ssim_error, capsize=3, color="#44aa77")
    axes[1].set(ylabel="Missing fingerprint-ROI SSIM-map mean ↑")
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"{args.label}: performance by mask geometry (mean ± 95% CI)")
    figure.tight_layout()
    figure.savefig(args.output / "performance-by-mask-family.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
