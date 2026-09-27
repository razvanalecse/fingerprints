#!/usr/bin/env python3
"""Stratified reconstruction analysis by effective ROI fraction and mask geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METRICS = {
    "reinjected_mean_mae": ("Mean MAE", False),
    "reinjected_mean_psnr": ("PSNR (dB)", True),
    "reinjected_mean_ssim_map_mean": ("SSIM", True),
    "reinjected_mean_orientation_error": ("Orientation error", False),
    "reinjected_mean_predictive_std": ("Predictive std", None),
    "reinjected_uncertainty_error_spearman": ("Uncertainty–error Spearman", True),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = len(array)
    mean = float(array.mean()) if count else float("nan")
    standard_deviation = float(array.std(ddof=1)) if count > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, count - 1) * standard_deviation / np.sqrt(count))
        if count > 1 else 0.0
    )
    return {
        "n": count, "mean": mean, "standard_deviation": standard_deviation,
        "ci95_low": mean - half_width, "ci95_high": mean + half_width,
    }


def grouped(rows, key):
    result = {}
    for value in sorted({row[key] for row in rows}):
        subset = [row for row in rows if row[key] == value]
        result[str(value)] = {
            "n": len(subset),
            "mean_r_roi": float(np.mean([row["r_roi"] for row in subset])),
            "metrics": {
                metric: summarize([row[metric] for row in subset]) for metric in METRICS
            },
        }
    return result


def main():
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        converted = {
            "sample_id": row["sample_id"], "mask_family": row["mask_family"],
            "nominal_r": float(row["observed_fraction"]),
            "r_roi": float(row["observed_fraction_fingerprint_roi"]),
        }
        converted.update({metric: float(row[metric]) for metric in METRICS})
        rows.append(converted)
    for row in rows:
        row["nominal_label"] = f"{row['nominal_r']:.2f}"
        bin_index = min(9, int(np.floor(row["r_roi"] * 10)))
        row["roi_bin"] = f"{bin_index / 10:.1f}–{(bin_index + 1) / 10:.1f}"

    r_values = np.asarray([row["r_roi"] for row in rows])
    correlations = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows])
        correlation, pvalue = stats.spearmanr(r_values, values, nan_policy="omit")
        correlations[metric] = {"spearman": float(correlation), "pvalue": float(pvalue)}

    report = {
        "input": str(args.input), "n": len(rows),
        "fraction_definition": "r_roi=sum(M*S)/sum(S), with target support used only for evaluation stratification",
        "correlation_with_r_roi": correlations,
        "by_nominal_fraction": grouped(rows, "nominal_label"),
        "by_effective_roi_bin": grouped(rows, "roi_bin"),
        "by_mask_family": grouped(rows, "mask_family"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stratified-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    selected = [
        "reinjected_mean_mae", "reinjected_mean_ssim_map_mean",
        "reinjected_mean_orientation_error", "reinjected_mean_predictive_std",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = np.asarray([row["nominal_r"] for row in rows])
    for axis, metric in zip(axes.ravel(), selected):
        values = np.asarray([row[metric] for row in rows])
        scatter = axis.scatter(r_values, values, c=colors, cmap="viridis", s=24, alpha=0.55)
        nominal_groups = report["by_nominal_fraction"]
        points = sorted(
            (entry["mean_r_roi"], entry["metrics"][metric])
            for entry in nominal_groups.values()
        )
        x = [point[0] for point in points]
        y = [point[1]["mean"] for point in points]
        yerr = [[point[1]["mean"] - point[1]["ci95_low"] for point in points],
                [point[1]["ci95_high"] - point[1]["mean"] for point in points]]
        axis.errorbar(x, y, yerr=yerr, color="black", marker="o", linewidth=1.5, capsize=3, label="nominal-r mean ±95% CI")
        rho = correlations[metric]["spearman"]
        axis.set(xlabel="Effective observed fraction in fingerprint ROI", ylabel=METRICS[metric][0], title=f"Spearman ρ={rho:.3f}")
        axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label="Nominal canvas fraction", shrink=0.8)
    figure.suptitle("Residual diffusion performance versus effective fingerprint information")
    figure.savefig(args.output / "metrics-vs-effective-roi-fraction.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    families = sorted(report["by_mask_family"])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, metric in zip(axes, selected[:3]):
        means = [report["by_mask_family"][family]["metrics"][metric]["mean"] for family in families]
        low = [means[i] - report["by_mask_family"][family]["metrics"][metric]["ci95_low"] for i, family in enumerate(families)]
        high = [report["by_mask_family"][family]["metrics"][metric]["ci95_high"] - means[i] for i, family in enumerate(families)]
        axis.bar(np.arange(len(families)), means, yerr=[low, high], capsize=3, color="#4778b3")
        axis.set_xticks(np.arange(len(families)), families, rotation=50, ha="right")
        axis.set(ylabel=METRICS[metric][0], title=METRICS[metric][0]); axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Performance by mask geometry (mean ±95% CI)")
    figure.tight_layout(); figure.savefig(args.output / "metrics-by-mask-family.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for family in families:
        entry = report["by_mask_family"][family]
        table_rows.append({
            "mask_family": family, "n": entry["n"], "mean_r_roi": entry["mean_r_roi"],
            **{metric: entry["metrics"][metric]["mean"] for metric in METRICS},
        })
    with (args.output / "mask-family-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0])); writer.writeheader(); writer.writerows(table_rows)
    print(json.dumps({"n": len(rows), "correlation_with_r_roi": correlations}, indent=2))


if __name__ == "__main__":
    main()
