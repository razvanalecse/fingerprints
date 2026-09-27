#!/usr/bin/env python3
"""Paired comparison of DDIM outputs with a deterministic reconstruction model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import holm_adjust, paired_comparison


def keyed(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["sample_id"], row["mask_family"], row["observed_fraction"]): row
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deterministic, diffusion = keyed(args.deterministic), keyed(args.diffusion)
    if deterministic.keys() != diffusion.keys():
        raise ValueError("tables do not contain identical paired cases")
    keys = sorted(deterministic)
    comparisons = {
        "single_mae": ("missing_roi_mae", "single_mae", False),
        "predictive_mean_mae": ("missing_roi_mae", "mean_mae", False),
        "best_of_k_mae": ("missing_roi_mae", "best_of_k_mae", False),
        "single_ssim": ("missing_roi_ssim_map_mean", "single_ssim_map_mean", True),
        "predictive_mean_ssim": ("missing_roi_ssim_map_mean", "mean_ssim_map_mean", True),
        "single_orientation": ("missing_roi_orientation_error", "single_orientation_error", False),
        "predictive_mean_orientation": ("missing_roi_orientation_error", "mean_orientation_error", False),
    }
    results = {}
    for label, (baseline_field, diffusion_field, higher_is_better) in comparisons.items():
        results[label] = paired_comparison(
            [float(deterministic[key][baseline_field]) for key in keys],
            [float(diffusion[key][diffusion_field]) for key in keys],
            higher_is_better=higher_is_better,
        )
    adjusted = holm_adjust(
        {label: result.wilcoxon_pvalue for label, result in results.items()}
    )
    report = {
        "reference": "gated_convolution",
        "candidate": "conditional_ddpm_sampled_with_ddim50",
        "num_pairs": len(keys),
        "positive_improvement_favours_diffusion": True,
        "warning": "best_of_k is selected by missing-ROI MAE and is oracle-assisted",
        "comparisons": {
            label: asdict(replace(result, wilcoxon_pvalue_holm=adjusted[label]))
            for label, result in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
