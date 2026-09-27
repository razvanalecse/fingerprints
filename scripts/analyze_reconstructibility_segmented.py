#!/usr/bin/env python3
"""Reconstructibility vs observed fraction r: smooth vs segmented (broken-stick) fit.

For each outcome (SSIM, orientation error, predictive-uncertainty magnitude,
sample diversity) this fits (a) a single linear model in r and (b) a
continuous piecewise-linear model with one breakpoint, grid-searched over the
midpoints between the discrete observed_fraction levels actually used. Models
are compared with a nested F-test. The breakpoint location is reported as a
number, not asserted to be a scientifically meaningful "critical threshold" --
that interpretive claim is deliberately left to the write-up, not automated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def design_smooth(r: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(r), r])


def design_segmented(r: np.ndarray, breakpoint: float) -> np.ndarray:
    return np.column_stack([np.ones_like(r), r, np.clip(r - breakpoint, 0.0, None)])


def fit_ols(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    return coefficients, float(np.sum(residuals**2))


def fit_best_segmented(r: np.ndarray, y: np.ndarray, candidate_breakpoints: np.ndarray):
    best = None
    for breakpoint in candidate_breakpoints:
        design = design_segmented(r, breakpoint)
        coefficients, rss = fit_ols(design, y)
        if best is None or rss < best[1]:
            best = (breakpoint, rss, coefficients)
    return best


def analyze_outcome(r: np.ndarray, y: np.ndarray, candidate_breakpoints: np.ndarray) -> dict:
    n = len(y)
    smooth_design = design_smooth(r)
    smooth_coef, smooth_rss = fit_ols(smooth_design, y)
    breakpoint, seg_rss, seg_coef = fit_best_segmented(r, y, candidate_breakpoints)

    df_smooth, df_segmented = smooth_design.shape[1], 3
    df_num = df_segmented - df_smooth
    df_den = n - df_segmented
    f_statistic = ((smooth_rss - seg_rss) / df_num) / (seg_rss / df_den) if seg_rss > 0 else float("inf")
    p_value = float(1.0 - stats.f.cdf(f_statistic, df_num, df_den))

    smooth_r2 = 1.0 - smooth_rss / np.sum((y - y.mean()) ** 2)
    segmented_r2 = 1.0 - seg_rss / np.sum((y - y.mean()) ** 2)

    return {
        "n": n,
        "smooth_model": {"intercept": float(smooth_coef[0]), "slope": float(smooth_coef[1]), "r_squared": float(smooth_r2)},
        "segmented_model": {
            "breakpoint_r": float(breakpoint),
            "intercept": float(seg_coef[0]),
            "slope_before": float(seg_coef[1]),
            "slope_after": float(seg_coef[1] + seg_coef[2]),
            "slope_change": float(seg_coef[2]),
            "r_squared": float(segmented_r2),
        },
        "nested_f_test": {
            "f_statistic": float(f_statistic),
            "df_numerator": df_num,
            "df_denominator": df_den,
            "p_value": p_value,
            "segmented_preferred_at_0.05": bool(p_value < 0.05),
        },
        "interpretation_note": (
            "breakpoint_r is a curve-fit location, not an asserted mechanistic "
            "threshold; report it descriptively."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.per_image_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    r = np.array([float(row["observed_fraction"]) for row in rows])
    levels = np.array(sorted(set(np.round(r, 2))))
    candidate_breakpoints = (levels[:-1] + levels[1:]) / 2.0

    outcomes = {
        "ssim": "mean_ssim_map_mean",
        "orientation_error": "mean_orientation_error",
        "uncertainty_magnitude": "mean_predictive_std",
        "diversity": "pairwise_diversity_mae",
    }
    report = {
        "source": str(args.per_image_csv),
        "num_images": len(rows),
        "observed_fraction_levels": levels.tolist(),
        "candidate_breakpoints": candidate_breakpoints.tolist(),
        "outcomes": {},
    }
    for name, column in outcomes.items():
        y = np.array([float(row[column]) for row in rows])
        report["outcomes"][name] = analyze_outcome(r, y, candidate_breakpoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
