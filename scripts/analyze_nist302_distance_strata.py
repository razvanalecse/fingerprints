#!/usr/bin/env python3
"""Paired subject-level tests across nearest-correspondence distance strata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 302, draws: int = 10000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    strata = ("nearest_0_2mm", "nearest_2_5mm", "nearest_gt_5mm")
    metrics = ("orientation_error", "mae", "ssim_map_mean")
    comparisons = ((0, 1), (1, 2), (0, 2))
    results = []
    for metric in metrics:
        subject_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            for stratum in strata:
                value = float(row[f"{stratum}_{metric}"])
                if np.isfinite(value):
                    subject_values[row["subject_id"]][stratum].append(value)
        subject_means = {
            subject: {key: float(np.mean(values)) for key, values in groups.items()}
            for subject, groups in subject_values.items()
        }
        metric_results = []
        for first_index, second_index in comparisons:
            first, second = strata[first_index], strata[second_index]
            paired = [
                (values[first], values[second])
                for values in subject_means.values()
                if first in values and second in values
            ]
            first_values = np.asarray([pair[0] for pair in paired])
            second_values = np.asarray([pair[1] for pair in paired])
            differences = second_values - first_values
            statistic, p_value = wilcoxon(
                differences, alternative="two-sided", zero_method="wilcox"
            )
            ci_low, ci_high = _bootstrap_mean_ci(differences)
            metric_results.append(
                {
                    "metric": metric,
                    "first_stratum": first,
                    "second_stratum": second,
                    "num_subjects": len(paired),
                    "first_mean": float(first_values.mean()),
                    "second_mean": float(second_values.mean()),
                    "paired_mean_difference_second_minus_first": float(differences.mean()),
                    "paired_median_difference_second_minus_first": float(
                        np.median(differences)
                    ),
                    "mean_difference_ci95_low": ci_low,
                    "mean_difference_ci95_high": ci_high,
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                    "rank_biserial_second_minus_first": _rank_biserial(differences),
                }
            )
        adjusted = _holm([result["p_value"] for result in metric_results])
        for result, adjusted_value in zip(metric_results, adjusted):
            result["holm_adjusted_p"] = adjusted_value
        results.extend(metric_results)

    report = {
        "unit_of_analysis": "subject mean",
        "split": "validation",
        "model": "nearest_observed_interpolation",
        "num_images": len(rows),
        "tests": results,
        "interpretation_warning": (
            "Distance-stratum differences combine reconstruction difficulty, residual "
            "registration uncertainty, and cross-impression appearance variation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
