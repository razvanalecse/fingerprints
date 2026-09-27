#!/usr/bin/env python3
"""Subject-level paired comparisons for registered-approximate SD302 models."""

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


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["sample_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate sample IDs in {path}")
    return keyed


def _subject_means(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(lambda: defaultdict(list))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {
        "classical": _read(args.baseline),
        "unet": _read(args.unet),
        "gated": _read(args.gated),
    }
    sample_sets = [set(rows) for rows in models.values()]
    if not all(samples == sample_sets[0] for samples in sample_sets[1:]):
        raise ValueError("model tables do not contain identical samples")
    subject_models = {name: _subject_means(rows) for name, rows in models.items()}
    subjects = sorted(set.intersection(*(set(rows) for rows in subject_models.values())))
    comparisons = {}
    for baseline_name, candidate_name in (
        ("classical", "unet"),
        ("classical", "gated"),
        ("unet", "gated"),
    ):
        baseline = {
            metric: [subject_models[baseline_name][subject][metric] for subject in subjects]
            for metric in METRICS
        }
        candidate = {
            metric: [subject_models[candidate_name][subject][metric] for subject in subjects]
            for metric in METRICS
        }
        comparisons[f"{candidate_name}_vs_{baseline_name}"] = paired_metric_suite(
            baseline, candidate, METRICS
        )
    report = {
        "split": "validation",
        "unit_of_analysis": "subject mean",
        "num_subjects": len(subjects),
        "num_images": len(sample_sets[0]),
        "positive_mean_improvement_means_candidate_is_better": True,
        "comparisons": comparisons,
        "warning": "Targets are registered_approximate and pixel metrics are alignment-sensitive.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
