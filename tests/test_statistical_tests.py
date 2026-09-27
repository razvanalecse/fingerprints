import numpy as np

from fingerprint_reconstruction.evaluation.statistical_tests import (
    holm_adjust,
    paired_comparison,
)


def test_paired_comparison_codes_improvement_positive():
    baseline = np.array([3.0, 4.0, 5.0, 6.0])
    candidate = baseline - np.array([0.5, 1.0, 1.5, 2.0])
    result = paired_comparison(baseline, candidate, higher_is_better=False)
    assert result.mean_improvement > 0
    assert result.ci95_low > 0


def test_holm_adjustment_is_monotonic_and_bounded():
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
    assert 0 <= adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1
