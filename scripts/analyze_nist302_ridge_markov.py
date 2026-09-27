#!/usr/bin/env python3
"""Subject-level paired test for the ridge-Markov predictability diagnostic.

Reads the per-image rows written by `diagnose_nist302_ridge_markov.py` and
asks, with this project's standard protocol (aggregate per subject, paired
Wilcoxon, Holm correction, Cohen dz), whether the reconstructed orientation
field is measurably more predictable than the real ridges occupying the same
held-out region.

Aggregating per subject first matters here for the same reason it does
everywhere else in this project: several images can come from one finger, and
counting them as independent evidence would inflate significance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fingerprint_reconstruction.evaluation.statistical_tests import paired_metric_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def subject_means(rows, column):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(column, "")
        if value == "":
            continue
        number = float(value)
        if np.isfinite(number):
            grouped[row["subject_id"]].append(number)
    return {subject: float(np.mean(values)) for subject, values in grouped.items()}


def main() -> None:
    args = parse_args()
    with args.per_image.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("no per-image rows found")

    steps = sorted({
        int(name.split("step")[1])
        for name in rows[0]
        if name.startswith("entropy_diff_step")
    })

    baseline: dict[str, list[float]] = {}
    candidate: dict[str, list[float]] = {}
    directions: dict[str, bool] = {}

    for step in steps:
        for quantity, higher_is_better in (
            # "improvement" is coded so that a positive value means the
            # reconstruction looks MORE like real ridges. For entropy, real
            # ridges are the less predictable arm, so a reconstruction that
            # matches them has HIGHER entropy.
            ("entropy", True),
            # The self-transition rate uses the opposite direction:
            # real ridges change orientation more often, so a lower rate is the
            # realistic one.
            ("self_rate", False),
        ):
            real_col = f"real_{quantity}_step{step}"
            recon_col = f"recon_{quantity}_step{step}"
            if real_col not in rows[0]:
                continue
            real = subject_means(rows, real_col)
            recon = subject_means(rows, recon_col)
            shared = sorted(set(real) & set(recon))
            name = f"{quantity}_step{step}"
            baseline[name] = [real[s] for s in shared]
            candidate[name] = [recon[s] for s in shared]
            directions[name] = higher_is_better

    results = paired_metric_suite(baseline, candidate, directions)
    subjects = len(next(iter(baseline.values()))) if baseline else 0
    report = {
        "per_image_rows": len(rows),
        "subjects": subjects,
        "coding": (
            "mean_improvement > 0 means the reconstruction resembles real ridges; "
            "negative means the reconstruction is more predictable (entropy) or "
            "flatter (self-transition rate) than the real ridges it replaces."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            name: {
                "mean_improvement": round(value["mean_improvement"], 5),
                "cohen_dz": round(value["cohen_dz"], 3),
                "holm_p": value["wilcoxon_pvalue_holm"],
            }
            for name, value in results.items()
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
