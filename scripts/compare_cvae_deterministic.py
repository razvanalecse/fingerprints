#!/usr/bin/env python3
"""Paired MAE comparisons between CVAE summaries and a deterministic model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

from fingerprint_reconstruction.evaluation.statistical_tests import holm_adjust, paired_comparison


def keyed(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["sample_id"], row["mask_family"], row["observed_fraction"]): row
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--cvae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deterministic, cvae = keyed(args.deterministic), keyed(args.cvae)
    if deterministic.keys() != cvae.keys():
        raise ValueError("tables do not contain identical paired cases")
    keys = sorted(deterministic)
    baseline = [float(deterministic[key]["missing_roi_mae"]) for key in keys]
    fields = ["single_sample_mae", "mean_reconstruction_mae", "best_of_k_mae"]
    results = {
        field: paired_comparison(
            baseline, [float(cvae[key][field]) for key in keys], higher_is_better=False
        )
        for field in fields
    }
    adjusted = holm_adjust({name: result.wilcoxon_pvalue for name, result in results.items()})
    report = {
        "reference": "gated_convolution_missing_roi_mae",
        "positive_improvement_favours_cvae": True,
        "warning": "best_of_k is oracle-assisted and is not single-sample performance",
        "num_pairs": len(keys),
        "comparisons": {
            name: asdict(replace(result, wilcoxon_pvalue_holm=adjusted[name]))
            for name, result in results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
