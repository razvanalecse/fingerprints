#!/usr/bin/env python3
"""Run paired statistical comparisons from per-image metric tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


METRICS = {
    "missing_roi_mae": False,
    "missing_roi_psnr": True,
    "missing_roi_ssim_map_mean": True,
    "missing_roi_orientation_error": False,
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {
        (row["sample_id"], row["mask_family"], row["observed_fraction"]): row
        for row in rows
    }
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate evaluation keys in {path}")
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_rows = read_rows(args.baseline)
    candidate_rows = read_rows(args.candidate)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("model tables do not contain identical paired cases")
    keys = sorted(baseline_rows)
    baseline = {
        metric: [float(baseline_rows[key][metric]) for key in keys] for metric in METRICS
    }
    candidate = {
        metric: [float(candidate_rows[key][metric]) for key in keys] for metric in METRICS
    }
    results = paired_metric_suite(baseline, candidate, METRICS)
    report = {
        "pairing_key": ["sample_id", "mask_family", "observed_fraction"],
        "num_cases": len(keys),
        "positive_mean_improvement_means_candidate_is_better": True,
        "metrics": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
