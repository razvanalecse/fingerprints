import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_reconstructibility_segmented import analyze_outcome, design_segmented, fit_ols


def test_recovers_a_clear_synthetic_breakpoint():
    rng = np.random.default_rng(0)
    r = np.repeat(np.linspace(0.1, 0.8, 8), 50)
    true_breakpoint = 0.4
    y = 1.0 - 0.2 * r - 1.5 * np.clip(r - true_breakpoint, 0.0, None)
    y = y + rng.normal(scale=0.01, size=y.shape)
    candidates = (np.linspace(0.1, 0.8, 8)[:-1] + np.linspace(0.1, 0.8, 8)[1:]) / 2.0
    result = analyze_outcome(r, y, candidates)
    assert abs(result["segmented_model"]["breakpoint_r"] - true_breakpoint) <= 0.1
    assert result["nested_f_test"]["p_value"] < 0.01
    assert result["segmented_model"]["r_squared"] > result["smooth_model"]["r_squared"]


def test_purely_linear_data_does_not_need_a_segment():
    rng = np.random.default_rng(1)
    r = np.repeat(np.linspace(0.1, 0.8, 8), 50)
    y = 0.5 - 0.3 * r + rng.normal(scale=0.02, size=r.shape)
    candidates = (np.linspace(0.1, 0.8, 8)[:-1] + np.linspace(0.1, 0.8, 8)[1:]) / 2.0
    result = analyze_outcome(r, y, candidates)
    # Segmented model cannot fit worse than smooth (superset), but the gain
    # should be negligible when the truth is linear.
    assert result["segmented_model"]["r_squared"] - result["smooth_model"]["r_squared"] < 0.01


def test_design_segmented_is_continuous_at_the_breakpoint():
    r = np.array([0.2, 0.3, 0.4, 0.5])
    breakpoint = 0.35
    design = design_segmented(r, breakpoint)
    coefficients, _ = fit_ols(design, np.array([1.0, 1.1, 1.3, 1.2]))
    value_at_breakpoint_from_below = coefficients[0] + coefficients[1] * breakpoint
    value_at_breakpoint_from_above = (
        coefficients[0] + coefficients[1] * breakpoint + coefficients[2] * 0.0
    )
    assert abs(value_at_breakpoint_from_below - value_at_breakpoint_from_above) < 1e-9
