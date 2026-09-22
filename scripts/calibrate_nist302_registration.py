#!/usr/bin/env python3
"""Summarize candidate SD302 registration filters using validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.diagnostics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validation = [row for row in rows if row["split"] == "validation"]
    confirmed = [
        row
        for row in validation
        if row["determination"] == "INDIV" and row["affine_rmse_mm"] != ""
    ]
    candidates = []
    for minimum_points in (4, 6, 8, 10):
        for maximum_rmse_mm in (0.20, 0.30, 0.40, 0.50):
            for minimum_coverage in (0.01, 0.025, 0.05, 0.10):
                accepted = [
                    row
                    for row in confirmed
                    if int(row["num_correspondences"]) >= minimum_points
                    and float(row["affine_rmse_mm"]) <= maximum_rmse_mm
                    and float(row["minimum_hull_coverage"]) >= minimum_coverage
                ]
                candidates.append(
                    {
                        "determination": "INDIV",
                        "minimum_correspondences": minimum_points,
                        "maximum_affine_rmse_mm": maximum_rmse_mm,
                        "minimum_hull_coverage": minimum_coverage,
                        "accepted_validation": len(accepted),
                        "retention_among_indiv_affine": (
                            len(accepted) / len(confirmed) if confirmed else 0.0
                        ),
                    }
                )
    report = {
        "selection_split": "validation",
        "num_validation_comp": len(validation),
        "num_validation_indiv_with_affine": len(confirmed),
        "candidate_filters": candidates,
        "warning": (
            "No test rows are used here. A final filter must be frozen before "
            "reporting any test-set reconstruction result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
