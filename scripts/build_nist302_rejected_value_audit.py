#!/usr/bin/env python3
"""Build a validation-only qualitative audit set rejected by registration filters."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def rejection_reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if row["determination"] != "INDIV":
        reasons.append("determination_not_INDIV")
    if int(row["num_correspondences"]) < 6:
        reasons.append("fewer_than_6_correspondences")
    residual = float(row["affine_rmse_mm"]) if row["affine_rmse_mm"] else np.inf
    if residual > 0.30:
        reasons.append("affine_rmse_above_0.30mm")
    if float(row["minimum_hull_coverage"]) < 0.01:
        reasons.append("hull_coverage_below_0.01")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-reason", type=int, default=4)
    args = parser.parse_args()
    with args.diagnostics.open(newline="", encoding="utf-8") as stream:
        diagnostics = list(csv.DictReader(stream))
    with args.latent_manifest.open(newline="", encoding="utf-8") as stream:
        manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}

    candidates = []
    for row in diagnostics:
        if row["split"] != "validation" or row["assessment"] != "VALUE":
            continue
        reasons = rejection_reasons(row)
        if not reasons:
            continue
        latent = manifest[row["latent_sample_id"]]
        candidates.append(
            {
                "latent_sample_id": row["latent_sample_id"],
                "lffs_filename": row["lffs_filename"],
                "subject_id": row["subject_id"],
                "split": "validation",
                "assessment": "VALUE",
                "determination": row["determination"],
                "num_correspondences": row["num_correspondences"],
                "affine_rmse_mm": row["affine_rmse_mm"],
                "minimum_hull_coverage": row["minimum_hull_coverage"],
                "rejection_reasons": ";".join(reasons),
                "latent_relative_path": latent["original_masked_path"],
                "qualitative_only": "True",
                "loss_allowed": "False",
                "metric_reporting_allowed": "False",
            }
        )

    selected, used_samples, used_subjects = [], set(), set()
    reason_order = (
        "determination_not_INDIV",
        "fewer_than_6_correspondences",
        "affine_rmse_above_0.30mm",
        "hull_coverage_below_0.01",
    )
    for reason in reason_order:
        eligible = sorted(
            (row for row in candidates if reason in row["rejection_reasons"]),
            key=lambda row: (row["subject_id"], row["latent_sample_id"]),
        )
        picked = 0
        for row in eligible:
            if row["latent_sample_id"] in used_samples or row["subject_id"] in used_subjects:
                continue
            selected.append({**row, "sampling_stratum": reason})
            used_samples.add(row["latent_sample_id"])
            used_subjects.add(row["subject_id"])
            picked += 1
            if picked >= args.per_reason:
                break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected:
        with (args.output_dir / "rejected_value_qualitative.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    report = {
        "split": "validation",
        "test_accessed": False,
        "assessment": "VALUE",
        "purpose": "qualitative generalization stress audit only",
        "prohibited_uses": ["training loss", "model selection metric", "claim of population performance"],
        "frozen_acceptance_filter": {
            "determination": "INDIV",
            "minimum_correspondences": 6,
            "maximum_affine_rmse_mm": 0.30,
            "minimum_hull_coverage": 0.01,
        },
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "selected_subjects": len({row["subject_id"] for row in selected}),
        "candidate_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in candidates
                    for reason in row["rejection_reasons"].split(";")
                ).items()
            )
        ),
        "sampling": "deterministic, at most one latent per subject, stratified by rejection reason",
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
