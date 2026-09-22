#!/usr/bin/env python3
"""Plot legible DDIM reconstruction and uncertainty trends over observed fraction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def mean_ci(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    mean = float(array.mean())
    half = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    return mean, half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fractions = sorted({round(float(row["observed_fraction"]), 2) for row in rows})
    groups = {
        fraction: [row for row in rows if round(float(row["observed_fraction"]), 2) == fraction]
        for fraction in fractions
    }
    specifications = [
        ("mean_mae", "Predictive-mean MAE", "lower is better", "#2b6cb0"),
        ("mean_orientation_error", "Orientation error", "lower is better", "#c53030"),
        ("pairwise_diversity_mae", "Pairwise sample diversity", "distribution spread", "#6b46c1"),
        ("uncertainty_error_spearman", "Uncertainty–error Spearman ρ", "higher association is better", "#2f855a"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (metric, title, subtitle, color) in zip(axes.ravel(), specifications):
        estimates = [mean_ci([float(row[metric]) for row in groups[f]]) for f in fractions]
        means = [item[0] for item in estimates]
        errors = [item[1] for item in estimates]
        axis.errorbar(
            fractions, means, yerr=errors, color=color, marker="o", linewidth=2,
            capsize=4, markersize=6,
        )
        axis.set_title(title, loc="left", fontweight="bold", y=1.075)
        axis.text(0.0, 1.012, subtitle, transform=axis.transAxes, fontsize=9, color="#555555")
        axis.grid(alpha=0.22)
        axis.set_xticks(fractions)
    for axis in axes[-1]:
        axis.set_xlabel("Observed fraction r")
    figure.suptitle("Conditional DDIM validation trends (50 steps, K=10)", fontweight="bold")
    figure.tight_layout(h_pad=2.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
