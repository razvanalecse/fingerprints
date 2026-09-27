#!/usr/bin/env python3
"""Paired statistical comparison of two DDIM step-count evaluations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import holm_adjust, paired_comparison


METRICS = {
    "single_mae": False,
    "mean_mae": False,
    "best_of_k_mae": False,
    "single_ssim_map_mean": True,
    "mean_ssim_map_mean": True,
    "single_orientation_error": False,
    "mean_orientation_error": False,
    "pairwise_diversity_mae": True,
    "uncertainty_error_spearman": True,
}


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {(r["sample_id"], r["mask_family"], r["observed_fraction"]): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = load(args.reference), load(args.candidate)
    keys = sorted(reference.keys() & candidate.keys())
    if not keys:
        raise ValueError("evaluations contain no paired cases")
    results = {}
    for metric, higher_is_better in METRICS.items():
        results[metric] = paired_comparison(
            [float(reference[key][metric]) for key in keys],
            [float(candidate[key][metric]) for key in keys],
            higher_is_better=higher_is_better,
        )
    adjusted = holm_adjust(
        {metric: result.wilcoxon_pvalue for metric, result in results.items()}
    )
    report = {
        "num_pairs": len(keys),
        "positive_improvement_favours_candidate": True,
        "comparisons": {
            metric: asdict(replace(result, wilcoxon_pvalue_holm=adjusted[metric]))
            for metric, result in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
