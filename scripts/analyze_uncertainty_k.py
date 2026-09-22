#!/usr/bin/env python3
"""Summarize fidelity, coverage, and compute scaling across Monte-Carlo K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("K", "DIRECTORY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for k_text, directory_text in args.run:
        k, directory = int(k_text), Path(directory_text)
        metadata = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        with (directory / "paired-per-image.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        mean = lambda field: float(np.nanmean([float(row[field]) for row in rows]))
        records.append({
            "K": k, "num_cases": len(rows),
            "mean_mae": mean("reinjected_mean_mae"),
            "mean_ssim": mean("reinjected_mean_ssim_map_mean"),
            "best_of_k_mae": mean("reinjected_best_of_k_mae"),
            "uncertainty_error_spearman": mean("reinjected_uncertainty_error_spearman"),
            "coverage_50": mean("reinjected_coverage_50"),
            "coverage_80": mean("reinjected_coverage_80"),
            "coverage_90": mean("reinjected_coverage_90"),
            "width_50": mean("reinjected_interval_width_50"),
            "width_80": mean("reinjected_interval_width_80"),
            "width_90": mean("reinjected_interval_width_90"),
            "diversity_mae": mean("reinjected_pairwise_diversity_mae"),
            "sampling_seconds": float(metadata["sampling_seconds"]),
            "seconds_per_case": float(metadata["sampling_seconds"]) / len(rows),
        })
    records.sort(key=lambda item: item["K"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "k-scaling.json").write_text(
        json.dumps({"runs": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "k-scaling.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    k = [item["K"] for item in records]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(k, [item["mean_mae"] for item in records], "o-", label="Predictive mean MAE")
    axes[0, 0].plot(k, [item["best_of_k_mae"] for item in records], "s--", label="Best-of-K MAE")
    axes[0, 0].set(ylabel="MAE", title="Fidelity versus K"); axes[0, 0].legend()
    for nominal, color in ((50, "#4778b3"), (80, "#e68633"), (90, "#3b9b5f")):
        axes[0, 1].plot(k, [item[f"coverage_{nominal}"] for item in records], "o-", color=color, label=f"Empirical {nominal}%")
        axes[0, 1].axhline(nominal / 100, color=color, linestyle=":", alpha=0.5)
    axes[0, 1].set(ylabel="Pixel coverage", title="Central predictive-interval coverage", ylim=(0, 1)); axes[0, 1].legend()
    axes[1, 0].plot(k, [item["uncertainty_error_spearman"] for item in records], "o-", color="#7a4da0")
    axes[1, 0].set(ylabel="Mean Spearman correlation", title="Uncertainty–error association")
    axes[1, 1].plot(k, [item["seconds_per_case"] for item in records], "o-", color="#a54343")
    axes[1, 1].set(ylabel="Seconds per case", title="Sampling cost")
    for axis in axes.ravel():
        axis.set_xlabel("Monte-Carlo samples K"); axis.set_xticks(k); axis.grid(alpha=0.2)
    figure.suptitle("Monte-Carlo sample count: fidelity, calibration, and cost")
    figure.tight_layout(); figure.savefig(args.output / "k-scaling.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"runs": records}, indent=2))


if __name__ == "__main__":
    main()
