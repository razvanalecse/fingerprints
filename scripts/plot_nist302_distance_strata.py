#!/usr/bin/env python3
"""Plot subject-level validation metrics by verified-anchor distance."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    strata = ("nearest_0_2mm", "nearest_2_5mm", "nearest_gt_5mm")
    labels = ("0–2 mm", "2–5 mm", ">5 mm")
    specifications = (
        ("orientation_error", "Orientation error", "lower is better"),
        ("mae", "Pixel MAE", "alignment-sensitive"),
        ("ssim_map_mean", "SSIM-map mean", "alignment/content-sensitive"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    generator = np.random.default_rng(302)
    for axis, (metric, title, subtitle) in zip(axes, specifications):
        subject_values = defaultdict(lambda: defaultdict(list))
        for row in rows:
            for stratum in strata:
                value = float(row[f"{stratum}_{metric}"])
                if np.isfinite(value):
                    subject_values[row["subject_id"]][stratum].append(value)
        means, lower, upper = [], [], []
        for stratum in strata:
            values = np.asarray(
                [
                    np.mean(groups[stratum])
                    for groups in subject_values.values()
                    if groups[stratum]
                ],
                dtype=np.float64,
            )
            boot = values[
                generator.integers(0, len(values), size=(10000, len(values)))
            ].mean(axis=1)
            mean = float(values.mean())
            means.append(mean)
            lower.append(mean - float(np.quantile(boot, 0.025)))
            upper.append(float(np.quantile(boot, 0.975)) - mean)
        positions = np.arange(3)
        axis.errorbar(
            positions,
            means,
            yerr=np.asarray([lower, upper]),
            marker="o",
            linewidth=2,
            capsize=4,
            color="#1f5a94",
        )
        axis.set_xticks(positions, labels)
        axis.set_title(f"{title}\n{subtitle}")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Nearest-observed baseline on SD302 validation — distance to verified correspondence",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.02,
        "Points are subject means; bars are subject-bootstrap 95% CIs. "
        "Targets are registered_approximate.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
