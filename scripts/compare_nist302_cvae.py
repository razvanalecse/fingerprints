#!/usr/bin/env python3
"""Paired subject-level comparison of two probabilistic SD302 evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["sample_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("duplicate sample IDs")
    return keyed


def aggregate(
    rows: dict[str, dict[str, str]], metrics: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows.values():
        for metric in metrics:
            value = float(row[metric])
            if np.isfinite(value):
                grouped[row["subject_id"]][metric].append(value)
    return {
        subject: {name: float(np.mean(values)) for name, values in data.items()}
        for subject, data in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k < 2:
        raise ValueError("k must be at least two")
    directions = {
        "single_heldout_q1_mae": False,
        f"mean_k{args.k}_heldout_q1_mae": False,
        f"mean_k{args.k}_heldout_q1_ssim_map_mean": True,
        f"best_k{args.k}_heldout_q1_mae": False,
        "mean_heldout_q1_orientation_error": False,
        "mean_heldout_q1_ridge_frequency_relative_mae": False,
        f"coverage_error_k{args.k}_50": False,
        f"coverage_error_k{args.k}_80": False,
        f"coverage_error_k{args.k}_90": False,
        f"coverage_error_k{args.k}_95": False,
        f"mean_k{args.k}_background_darkness": False,
    }
    metrics = tuple(directions)
    reference_rows, candidate_rows = read_rows(args.reference), read_rows(args.candidate)
    if set(reference_rows) != set(candidate_rows):
        raise ValueError("evaluation files do not contain identical samples")
    reference = aggregate(reference_rows, metrics)
    candidate = aggregate(candidate_rows, metrics)
    common = sorted(set(reference) & set(candidate))
    metric_subjects = {
        metric: [
            subject
            for subject in common
            if metric in reference[subject] and metric in candidate[subject]
        ]
        for metric in metrics
    }
    report = {
        "split": "validation",
        "test_loaded": False,
        "unit_of_analysis": "subject mean",
        "num_images": len(reference_rows),
        "num_subjects": len(common),
        "subjects_per_metric": {
            name: len(subjects) for name, subjects in metric_subjects.items()
        },
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "positive_mean_improvement_means_candidate_is_better": True,
        "tests": paired_metric_suite(
            {
                metric: [reference[s][metric] for s in metric_subjects[metric]]
                for metric in metrics
            },
            {
                metric: [candidate[s][metric] for s in metric_subjects[metric]]
                for metric in metrics
            },
            directions,
        ),
        "descriptive_only_not_tested": [
            f"diversity_k{args.k}_heldout_q1_mae",
            f"uncertainty_error_spearman_k{args.k}",
            f"interval_width_k{args.k}_50",
            f"interval_width_k{args.k}_80",
            f"interval_width_k{args.k}_90",
            f"interval_width_k{args.k}_95",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
