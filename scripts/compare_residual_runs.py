#!/usr/bin/env python3
"""Compare two residual-diffusion evaluations on exactly paired cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


METRICS = {
    "mean_mae": False,
    "mean_psnr": True,
    "mean_ssim_map_mean": True,
    "mean_orientation_error": False,
    "mean_ridge_frequency_mae_cpx": False,
    "mean_ridge_period_mae_pixels": False,
    "single_mae": False,
    "best_of_k_mae": False,
    "uncertainty_error_spearman": True,
}


def load_rows(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    key_fields = ("sample_id", "mask_family", "observed_fraction", "selected_dataset_index")
    missing = [name for name in key_fields if not rows or name not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks pairing fields: {missing}")
    keyed = {tuple(row[name] for name in key_fields): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"{path} contains duplicate paired-case keys")
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference, candidate = load_rows(args.reference), load_rows(args.candidate)
    if reference.keys() != candidate.keys():
        only_reference = len(reference.keys() - candidate.keys())
        only_candidate = len(candidate.keys() - reference.keys())
        raise ValueError(
            "evaluations are not exactly paired: "
            f"reference-only={only_reference}, candidate-only={only_candidate}"
        )
    keys = sorted(reference)
    baseline_values, candidate_values, skipped_metrics = {}, {}, []
    for metric in METRICS:
        column = f"reinjected_{metric}"
        if column not in reference[keys[0]] or column not in candidate[keys[0]]:
            skipped_metrics.append(metric)
            continue
        baseline_values[metric] = [float(reference[key][column]) for key in keys]
        candidate_values[metric] = [float(candidate[key][column]) for key in keys]

    compared_directions = {
        metric: METRICS[metric] for metric in baseline_values
    }

    report = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "num_exactly_paired_cases": len(keys),
        "skipped_metrics_missing_from_at_least_one_run": skipped_metrics,
        "positive_improvement_favours_candidate": True,
        "multiple_comparison_correction": "Holm correction across listed metrics",
        "comparisons": paired_metric_suite(
            baseline_values, candidate_values, compared_directions
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
