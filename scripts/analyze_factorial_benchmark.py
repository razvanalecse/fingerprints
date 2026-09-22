#!/usr/bin/env python3
"""Analyze the nested repeated-measures mask-by-r reconstruction benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from fingerprint_reconstruction.evaluation.statistical_tests import holm_adjust


PRIMARY_METRICS = {
    "reinjected_mean_mae": ("Predictive-mean MAE", False),
    "reinjected_mean_ssim_map_mean": ("SSIM", True),
    "reinjected_mean_orientation_error": ("Axial orientation error", False),
    "reinjected_pairwise_diversity_mae": ("Pairwise sample diversity", None),
    "reinjected_mean_predictive_std": ("Predictive standard deviation", None),
    "reinjected_uncertainty_error_spearman": ("Uncertainty–error Spearman", True),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    sem = float(stats.sem(values)) if len(values) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def friedman_result(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] == 2:
        differences = matrix[:, 1] - matrix[:, 0]
        nonzero = differences != 0
        ranks = stats.rankdata(np.abs(differences[nonzero]))
        denominator = float(ranks.sum())
        rank_biserial = (
            float((ranks[differences[nonzero] > 0].sum() - ranks[differences[nonzero] < 0].sum()) / denominator)
            if denominator > 0 else 0.0
        )
        result = stats.wilcoxon(matrix[:, 0], matrix[:, 1])
        return {
            "test": "paired Wilcoxon (two repeated conditions)",
            "n_blocks": int(matrix.shape[0]),
            "n_conditions": 2,
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "matched_rank_biserial_second_minus_first": rank_biserial,
        }
    result = stats.friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    return {
        "test": "Friedman",
        "n_blocks": int(matrix.shape[0]),
        "n_conditions": int(matrix.shape[1]),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "kendall_w": float(result.statistic / (matrix.shape[0] * (matrix.shape[1] - 1))),
    }


def paired_wilcoxon(a, b, *, higher_is_better):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    improvement = b - a if higher_is_better else a - b
    result = stats.wilcoxon(improvement)
    sd = improvement.std(ddof=1)
    half = stats.t.ppf(0.975, len(improvement) - 1) * stats.sem(improvement)
    return {
        "n": len(improvement),
        "mean_improvement_favouring_second": float(improvement.mean()),
        "ci95_low": float(improvement.mean() - half),
        "ci95_high": float(improvement.mean() + half),
        "cohen_dz": float(improvement.mean() / sd) if sd > 0 else float("inf"),
        "wilcoxon_statistic": float(result.statistic),
        "wilcoxon_pvalue": float(result.pvalue),
    }


def within_subject_trend(matrix, fractions):
    """Test subject-level monotonic associations without pooling pixels/cases."""

    correlations = np.asarray([
        stats.spearmanr(fractions, row).statistic for row in np.asarray(matrix)
    ])
    result = stats.wilcoxon(correlations, alternative="two-sided")
    return {
        "n_subjects": len(correlations),
        "mean_subject_spearman": float(correlations.mean()),
        "median_subject_spearman": float(np.median(correlations)),
        "positive_subject_fraction": float(np.mean(correlations > 0)),
        "wilcoxon_pvalue_vs_zero": float(result.pvalue),
    }


def subject_fraction_matrix(rows, metric, subjects, families, fractions):
    lookup = {(row["subject_id"], row["mask_family"], row["r"]): row[metric] for row in rows}
    return np.asarray([
        [np.mean([lookup[(subject, family, fraction)] for family in families]) for fraction in fractions]
        for subject in subjects
    ])


def subject_family_matrix(rows, metric, subjects, families, fractions):
    lookup = {(row["subject_id"], row["mask_family"], row["r"]): row[metric] for row in rows}
    return np.asarray([
        [np.mean([lookup[(subject, family, fraction)] for fraction in fractions]) for family in families]
        for subject in subjects
    ])


def fixed_subject_design(subject_indices, r_values, hinge=None):
    count = len(subject_indices)
    n_subjects = int(np.max(subject_indices)) + 1
    columns = [np.ones(count)]
    columns.extend((subject_indices == value).astype(float) for value in range(1, n_subjects))
    columns.append(r_values)
    if hinge is not None:
        columns.append(np.maximum(0.0, hinge - r_values))
    return np.column_stack(columns)


def bic(y, design, extra_parameters=0):
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    rss = max(float(np.sum(residual**2)), np.finfo(float).tiny)
    n, parameters = len(y), design.shape[1] + extra_parameters
    return float(n * np.log(rss / n) + parameters * np.log(n))


def segmented_threshold(matrix, fractions, *, bootstrap, rng):
    matrix = np.asarray(matrix, dtype=float)
    subject_indices = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    r_values = np.tile(np.asarray(fractions, dtype=float), matrix.shape[0])
    y = matrix.ravel()
    linear_bic = bic(y, fixed_subject_design(subject_indices, r_values))
    knots = np.asarray(fractions[1:-1], dtype=float)

    def best_for(data):
        subject_index = np.repeat(np.arange(data.shape[0]), data.shape[1])
        repeated_r = np.tile(np.asarray(fractions, dtype=float), data.shape[0])
        values = data.ravel()
        scores = [
            bic(values, fixed_subject_design(subject_index, repeated_r, knot), extra_parameters=1)
            for knot in knots
        ]
        best = int(np.argmin(scores))
        return float(knots[best]), float(scores[best])

    best_knot, segmented_bic = best_for(matrix)
    boot_knots = []
    for _ in range(bootstrap):
        sampled = rng.integers(0, matrix.shape[0], matrix.shape[0])
        boot_knots.append(best_for(matrix[sampled])[0])
    low, high = np.quantile(boot_knots, (0.025, 0.975))
    return {
        "linear_bic": linear_bic,
        "best_segmented_bic": segmented_bic,
        "delta_bic_segmented_minus_linear": segmented_bic - linear_bic,
        "best_knot": best_knot,
        "bootstrap_knot_ci95": [float(low), float(high)],
        "bootstrap_knot_probabilities": {
            f"{knot:.2f}": float(np.mean(np.asarray(boot_knots) == knot)) for knot in knots
        },
        "interpretation_rule": "Only delta BIC <= -10 with concentrated bootstrap support is strong exploratory evidence for a transition; this is not a confirmatory critical threshold test.",
    }


def main():
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    fraction_domains = {row.get("fraction_domain", "canvas") for row in raw}
    if len(fraction_domains) != 1:
        raise ValueError(f"mixed fraction domains are not supported: {fraction_domains}")
    fraction_domain = next(iter(fraction_domains))
    for row in raw:
        converted = {
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "mask_family": row["mask_family"],
            "r": float(row["observed_fraction"]),
            "r_roi": float(row["observed_fraction_fingerprint_roi"]),
        }
        converted.update({metric: float(row[metric]) for metric in PRIMARY_METRICS})
        rows.append(converted)

    # Severe-partial is defined only for r<=0.3 and is therefore reported
    # descriptively, but excluded from the complete repeated-measures matrix.
    complete = [row for row in rows if row["mask_family"] != "severe_partial"]
    subjects = sorted({row["subject_id"] for row in complete})
    families = sorted({row["mask_family"] for row in complete})
    fractions = sorted({row["r"] for row in complete})
    expected = len(subjects) * len(families) * len(fractions)
    if len(complete) != expected:
        raise ValueError(f"incomplete factorial matrix: found {len(complete)}, expected {expected}")
    if len({(row["subject_id"], row["mask_family"], row["r"]) for row in complete}) != expected:
        raise ValueError("factorial matrix contains duplicate cells")

    rng = np.random.default_rng(args.seed)
    inferential = {}
    cell_summary = {}
    for metric, (_, direction) in PRIMARY_METRICS.items():
        by_r = subject_fraction_matrix(complete, metric, subjects, families, fractions)
        by_family = subject_family_matrix(complete, metric, subjects, families, fractions)
        inferential[metric] = {
            "r_main_effect_friedman": friedman_result(by_r),
            "within_subject_monotonic_trend": within_subject_trend(by_r, fractions),
            "mask_main_effect_friedman": friedman_result(by_family),
            "segmented_regression_exploratory": segmented_threshold(
                by_r, fractions, bootstrap=args.bootstrap, rng=rng
            ),
        }
        cell_summary[metric] = {
            family: {
                f"{fraction:.2f}": mean_ci([
                    row[metric] for row in complete
                    if row["mask_family"] == family and row["r"] == fraction
                ])
                for fraction in fractions
            }
            for family in families
        }
        if direction is not None:
            comparisons = {}
            for fraction in fractions:
                central = [
                    next(row[metric] for row in complete if row["subject_id"] == subject and row["mask_family"] == "central_only" and row["r"] == fraction)
                    for subject in subjects
                ]
                peripheral = [
                    next(row[metric] for row in complete if row["subject_id"] == subject and row["mask_family"] == "peripheral_only" and row["r"] == fraction)
                    for subject in subjects
                ]
                comparisons[f"{fraction:.2f}"] = paired_wilcoxon(
                    peripheral, central, higher_is_better=direction
                )
            adjusted = holm_adjust({key: value["wilcoxon_pvalue"] for key, value in comparisons.items()})
            for key in comparisons:
                comparisons[key]["wilcoxon_pvalue_holm"] = adjusted[key]
            inferential[metric]["central_vs_peripheral_by_r"] = comparisons

    severe = {
        metric: {
            f"{fraction:.2f}": mean_ci([
                row[metric] for row in rows
                if row["mask_family"] == "severe_partial" and row["r"] == fraction
            ])
            for fraction in sorted({row["r"] for row in rows if row["mask_family"] == "severe_partial"})
        }
        for metric in PRIMARY_METRICS
    }
    report = {
        "input": str(args.input),
        "design": "complete repeated measures over subject x mask family x r; nested masks within subject/family",
        "fraction_domain": fraction_domain,
        "num_subjects": len(subjects),
        "complete_matrix_conditions": len(complete),
        "families_complete_matrix": families,
        "fractions": fractions,
        "severe_partial_excluded_from_factorial_tests_reason": "defined only for r<=0.30",
        "cell_summary": cell_summary,
        "severe_partial_descriptive": severe,
        "inferential": inferential,
        "effective_roi_fraction_by_family_and_nominal_r": {
            family: {
                f"{fraction:.2f}": mean_ci([
                    row["r_roi"] for row in complete
                    if row["mask_family"] == family and row["r"] == fraction
                ])
                for fraction in fractions
            }
            for family in families
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "factorial-analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plot_metrics = [
        "reinjected_mean_mae", "reinjected_mean_ssim_map_mean",
        "reinjected_mean_orientation_error", "reinjected_pairwise_diversity_mae",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fraction_label = (
        "Observed fingerprint-ROI fraction r_ROI"
        if fraction_domain == "fingerprint_roi"
        else "Observed canvas fraction r"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for axis, metric in zip(axes.ravel(), plot_metrics):
        for color, family in zip(colors, families):
            cells = [cell_summary[metric][family][f"{fraction:.2f}"] for fraction in fractions]
            means = np.asarray([cell["mean"] for cell in cells])
            lower = means - np.asarray([cell["ci95_low"] for cell in cells])
            upper = np.asarray([cell["ci95_high"] for cell in cells]) - means
            axis.errorbar(fractions, means, yerr=[lower, upper], marker="o", capsize=2, linewidth=1.4, color=color, label=family)
        axis.set(title=PRIMARY_METRICS[metric][0], xlabel=fraction_label, ylabel=PRIMARY_METRICS[metric][0])
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("Nested repeated-measures benchmark: geometry × observed fraction")
    figure.tight_layout()
    figure.savefig(args.output / "geometry-by-r.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, metric in zip(axes.ravel(), plot_metrics):
        matrix = subject_fraction_matrix(complete, metric, subjects, families, fractions)
        means = matrix.mean(axis=0)
        half = stats.t.ppf(0.975, len(subjects) - 1) * stats.sem(matrix, axis=0)
        axis.plot(fractions, means, "o-", color="#193b66", linewidth=2)
        axis.fill_between(fractions, means - half, means + half, color="#4f81bd", alpha=0.25)
        axis.set(title=PRIMARY_METRICS[metric][0], xlabel=fraction_label, ylabel=PRIMARY_METRICS[metric][0])
        axis.grid(alpha=0.2)
    figure.suptitle("Marginal effect of observed information (mean across geometries ±95% CI)")
    figure.tight_layout()
    figure.savefig(args.output / "marginal-performance-vs-r.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    roi_summary = report["effective_roi_fraction_by_family_and_nominal_r"]
    for color, family in zip(colors, families):
        means = [roi_summary[family][f"{fraction:.2f}"]["mean"] for fraction in fractions]
        axis.plot(fractions, means, "o-", color=color, label=family)
    axis.plot((0, 1), (0, 1), "k--", linewidth=1, label="canvas r = ROI r")
    axis.set(
        xlabel="Nominal observed canvas fraction r",
        ylabel="Effective observed fraction inside fingerprint ROI",
        title="Nominal area is not identical to observed fingerprint information",
        xlim=(0.05, 0.85), ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2); axis.legend(fontsize=8, ncol=2); figure.tight_layout()
    figure.savefig(args.output / "nominal-vs-effective-roi-fraction.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "num_subjects": len(subjects),
        "complete_matrix_conditions": len(complete),
        "r_effects": {metric: result["r_main_effect_friedman"] for metric, result in inferential.items()},
        "mask_effects": {metric: result["mask_main_effect_friedman"] for metric, result in inferential.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
