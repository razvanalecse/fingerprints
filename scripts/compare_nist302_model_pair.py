#!/usr/bin/env python3
"""Paired subject-level comparison of two SD302 registered-model evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


METRICS = {
    "evaluation_mae": False,
    "evaluation_ssim_map_mean": True,
    "evaluation_orientation_error": False,
    "evaluation_ridge_frequency_relative_mae": False,
}


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["sample_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate sample IDs in {path}")
    return keyed


def subject_means(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows.values():
        for metric in METRICS:
            value = float(row[metric])
            if np.isfinite(value):
                grouped[row["subject_id"]][metric].append(value)
    return {
        subject: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for subject, metrics in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_rows = read_rows(args.reference)
    candidate_rows = read_rows(args.candidate)
    if set(reference_rows) != set(candidate_rows):
        raise ValueError("evaluation tables do not contain identical samples")

    reference_subjects = subject_means(reference_rows)
    candidate_subjects = subject_means(candidate_rows)
    subjects = sorted(set(reference_subjects) & set(candidate_subjects))
    reference = {
        metric: [reference_subjects[subject][metric] for subject in subjects]
        for metric in METRICS
    }
    candidate = {
        metric: [candidate_subjects[subject][metric] for subject in subjects]
        for metric in METRICS
    }
    report = {
        "split": "validation",
        "unit_of_analysis": "subject mean",
        "num_subjects": len(subjects),
        "num_images": len(reference_rows),
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "positive_mean_improvement_means_candidate_is_better": True,
        "tests": paired_metric_suite(reference, candidate, METRICS),
        "warning": (
            "Targets are registered_approximate; pixel metrics include residual "
            "registration and cross-impression appearance error."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
