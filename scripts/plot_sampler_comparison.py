#!/usr/bin/env python3
"""Create a compact quality, uncertainty, and cost comparison for two samplers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean_ci(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    mean = values.mean()
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(mean), float(half)


def grouped(axis, datasets, fields, labels, title, *, lower_better=None, log=False):
    colors = ("#2b6cb0", "#c53030")
    width = 0.34
    x = np.arange(len(fields))
    for model_index, (name, rows) in enumerate(datasets):
        estimates = [mean_ci(rows, field) for field in fields]
        offset = (model_index - 0.5) * width
        axis.bar(
            x + offset,
            [value[0] for value in estimates],
            width,
            yerr=[value[1] for value in estimates],
            capsize=3,
            label=name,
            color=colors[model_index],
            alpha=0.9,
        )
    axis.set_xticks(x, labels)
    axis.set_title(title, loc="left", fontweight="bold", y=1.075)
    if lower_better is not None:
        axis.text(0.0, 1.01, lower_better, transform=axis.transAxes, color="#555555", fontsize=9)
    if log:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddim", type=Path, required=True)
    parser.add_argument("--repaint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = [("DDIM-50", load(args.ddim)), ("RePaint U=2", load(args.repaint))]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    grouped(axes[0, 0], datasets, ["single_mae", "mean_mae", "best_of_k_mae"], ["single", "mean", "best-of-5"], "Missing-ROI MAE", lower_better="lower is better")
    grouped(axes[0, 1], datasets, ["single_ssim_map_mean", "mean_ssim_map_mean"], ["single", "mean"], "Missing-ROI SSIM", lower_better="higher is better")
    grouped(axes[0, 2], datasets, ["single_orientation_error", "mean_orientation_error"], ["single", "mean"], "Orientation error", lower_better="lower is better")
    grouped(axes[1, 0], datasets, ["pairwise_diversity_mae"], ["pairwise L1"], "Sample diversity", lower_better="spread, not accuracy")
    grouped(axes[1, 1], datasets, ["uncertainty_error_spearman", "coverage_80"], ["Spearman ρ", "80% coverage"], "Uncertainty diagnostics", lower_better="higher is better")
    grouped(axes[1, 2], datasets, ["seconds_per_reconstruction"], ["seconds/sample"], "Inference cost", lower_better="lower is better (log scale)", log=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "Sampler comparison on 90 subject-distributed validation cases (K=5)",
        y=0.995,
        fontweight="bold",
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88), h_pad=3.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
