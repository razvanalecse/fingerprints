#!/usr/bin/env python3
"""Paired subject-level inference for latent-support enhancement models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


METRICS = {
    "heldout_q1_mae": False,
    "heldout_q1_ssim_map_mean": True,
    "heldout_q1_orientation_error": False,
    "heldout_q1_ridge_frequency_relative_mae": False,
    "background_darkness": False,
}


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample identifiers")
    return result


def aggregate(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows.values():
        for metric in METRICS:
            value = float(row[metric])
            if np.isfinite(value):
                grouped[row["subject_id"]][metric].append(value)
    return {
        subject: {metric: float(np.mean(values)) for metric, values in data.items()}
        for subject, data in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_rows, candidate_rows = read(args.reference), read(args.candidate)
    if set(reference_rows) != set(candidate_rows):
        raise ValueError("evaluation tables do not contain identical samples")
    reference, candidate = aggregate(reference_rows), aggregate(candidate_rows)
    subjects = sorted(set(reference) & set(candidate))
    metric_subjects = {
        metric: [
            subject
            for subject in subjects
            if metric in reference[subject] and metric in candidate[subject]
        ]
        for metric in METRICS
    }
    report = {
        "split": "validation",
        "unit_of_analysis": "subject mean",
        "num_subjects": len(subjects),
        "num_images": len(reference_rows),
        "subjects_per_metric": {
            metric: len(values) for metric, values in metric_subjects.items()
        },
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "tests": paired_metric_suite(
            {
                metric: [reference[s][metric] for s in metric_subjects[metric]]
                for metric in METRICS
            },
            {
                metric: [candidate[s][metric] for s in metric_subjects[metric]]
                for metric in METRICS
            },
            METRICS,
        ),
        "positive_mean_improvement_means_candidate_is_better": True,
        "test_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
