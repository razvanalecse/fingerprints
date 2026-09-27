#!/usr/bin/env python3
"""Paired comparison of reinjected outputs from two latent model versions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["sample_id"], row["mask_family"], row["observed_fraction"]): row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("model tables do not contain identical paired cases")
    keys = sorted(baseline)
    directions = {
        "reinjected_mean_mae": False,
        "reinjected_mean_psnr": True,
        "reinjected_mean_ssim_map_mean": True,
        "reinjected_mean_orientation_error": False,
        "reinjected_single_mae": False,
        "reinjected_best_of_k_mae": False,
        "reinjected_uncertainty_error_spearman": True,
    }
    reference = {name: [float(baseline[key][name]) for key in keys] for name in directions}
    model = {name: [float(candidate[key][name]) for key in keys] for name in directions}
    report = {
        "num_pairs": len(keys),
        "positive_improvement_favours_candidate": True,
        "comparisons": paired_metric_suite(reference, model, directions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
