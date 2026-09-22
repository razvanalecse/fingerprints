#!/usr/bin/env python3
"""Derive preregistered distance weights from validation-only structural error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STRATA = ("nearest_0_2mm", "nearest_2_5mm", "nearest_gt_5mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.distance_tests.read_text(encoding="utf-8"))
    if source.get("split") != "validation":
        raise ValueError("confidence calibration is allowed only on validation")

    orientation = {
        test["first_stratum"]: float(test["first_mean"])
        for test in source["tests"]
        if test["metric"] == "orientation_error"
    }
    orientation.update(
        {
            test["second_stratum"]: float(test["second_mean"])
            for test in source["tests"]
            if test["metric"] == "orientation_error"
        }
    )
    if set(orientation) != set(STRATA) or any(orientation[key] <= 0 for key in STRATA):
        raise ValueError("all three positive orientation-error stratum means are required")
    reference = orientation[STRATA[0]]
    weights = {key: min(1.0, reference / orientation[key]) for key in STRATA}
    report = {
        "status": "frozen_from_validation",
        "source": str(args.distance_tests),
        "source_split": "validation",
        "source_model": source.get("model"),
        "source_metric": "subject-level mean orientation_error",
        "stratum_orientation_error": orientation,
        "formula": "w_s = min(1, E_0-2mm / E_s)",
        "weights": weights,
        "ordered_weights": [weights[key] for key in STRATA],
        "allowed_use": "training and validation registered-approximate loss",
        "test_policy": "test remains sealed; weights must not be revised from test outcomes",
        "interpretation_warning": (
            "The curve combines residual registration uncertainty, reconstruction "
            "difficulty, and cross-impression variation; weights are empirical reliability "
            "weights, not calibrated probabilities."
        ),
        "required_ablation": "compare calibrated weights with uniform [1,1,1] on validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
