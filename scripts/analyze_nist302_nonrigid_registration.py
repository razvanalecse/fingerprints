#!/usr/bin/env python3
"""Compare thin-plate-spline and affine registration on SD302.

Motivated by a methodological audit's suggestion that SD302's affine registration
may not capture genuine local (elastic) skin deformation between the latent
and exemplar impressions. Fair test: **leave-one-correspondence-out** RMSE
for both affine and TPS on every manifest row with enough points -- fitting
on N-1 correspondences and predicting the held-out one, which (unlike the
manifest's own in-sample `affine_rmse_mm`) cannot be gamed by a more
flexible model that simply interpolates its own training points exactly.

Uses only the manifest's stored correspondence points (already
examiner-verified); no new registration data collection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.evaluation.nonrigid_registration import (
    affine_point_mapper,
    leave_one_out_residuals_mm,
    thin_plate_spline_mapper,
)
from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite

MINIMUM_CORRESPONDENCES = 4  # need >=3 remaining after holding one out for affine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("test split must remain sealed; use train or validation")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"no manifest rows for split={args.split}")

    per_row: list[dict[str, object]] = []
    skipped_too_few_points = 0
    for row in rows:
        latent_points = np.asarray(ast.literal_eval(row["latent_correspondences_0_01mm"]), dtype=np.float64)
        exemplar_points = np.asarray(ast.literal_eval(row["exemplar_correspondences_0_01mm"]), dtype=np.float64)
        if latent_points.shape[0] < MINIMUM_CORRESPONDENCES:
            skipped_too_few_points += 1
            continue
        affine_residuals = leave_one_out_residuals_mm(latent_points, exemplar_points, affine_point_mapper)
        tps_residuals = leave_one_out_residuals_mm(latent_points, exemplar_points, thin_plate_spline_mapper)
        per_row.append(
            {
                "relative_path": row["relative_path"],
                "subject_id": row["subject_id"],
                "num_correspondences": int(latent_points.shape[0]),
                "manifest_affine_rmse_mm_insample": float(row["affine_rmse_mm"]),
                "affine_loo_rmse_mm": float(np.sqrt(np.mean(np.square(affine_residuals)))),
                "tps_loo_rmse_mm": float(np.sqrt(np.mean(np.square(tps_residuals)))),
            }
        )

    subject_affine: dict[str, list[float]] = defaultdict(list)
    subject_tps: dict[str, list[float]] = defaultdict(list)
    for entry in per_row:
        subject_affine[entry["subject_id"]].append(entry["affine_loo_rmse_mm"])
        subject_tps[entry["subject_id"]].append(entry["tps_loo_rmse_mm"])
    subjects = sorted(subject_affine)
    affine_means = [float(np.mean(subject_affine[s])) for s in subjects]
    tps_means = [float(np.mean(subject_tps[s])) for s in subjects]

    tests = paired_metric_suite(
        {"loo_rmse_mm": affine_means},
        {"loo_rmse_mm": tps_means},
        {"loo_rmse_mm": False},  # lower is better
    )

    report = {
        "split": args.split,
        "test_loaded": False,
        "rows_used": len(per_row),
        "rows_skipped_too_few_points": skipped_too_few_points,
        "minimum_correspondences_required": MINIMUM_CORRESPONDENCES,
        "num_subjects": len(subjects),
        "affine_loo_rmse_mm_mean": float(np.mean(affine_means)),
        "tps_loo_rmse_mm_mean": float(np.mean(tps_means)),
        "manifest_insample_affine_rmse_mm_mean": float(np.mean([e["manifest_affine_rmse_mm_insample"] for e in per_row])),
        "paired_subject_test": tests,
        "interpretation": (
            "positive_mean_improvement_means_tps_is_better (lower leave-one-out "
            "RMSE); the manifest's own in-sample affine RMSE is reported "
            "separately purely for context and is NOT a fair comparison target "
            "(it is not cross-validated)."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {k: report[k] for k in (
            "rows_used", "affine_loo_rmse_mm_mean", "tps_loo_rmse_mm_mean",
            "manifest_insample_affine_rmse_mm_mean",
        )},
        indent=2,
    ))


if __name__ == "__main__":
    main()
