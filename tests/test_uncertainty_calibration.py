import numpy as np
import pytest

from fingerprint_reconstruction.metrics import (
    bounded_scaled_interval_coverage,
    scaled_std_interval_coverage,
    standardized_residual_scale,
)


def test_standardized_scale_recovers_requested_empirical_coverage():
    error = np.arange(1.0, 101.0)
    std = np.ones_like(error)
    scale = standardized_residual_scale(error, std, nominal_coverage=0.9, epsilon=1e-12)
    coverage, width = scaled_std_interval_coverage(error, std, scale=scale)
    assert coverage >= 0.9
    assert coverage <= 0.91
    assert width == pytest.approx(2.0 * scale)


def test_uncertainty_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        standardized_residual_scale([], [], nominal_coverage=0.9)
    with pytest.raises(ValueError):
        scaled_std_interval_coverage([1], [1], scale=-1)


def test_bounded_interval_width_never_exceeds_image_range():
    mean = np.asarray([0.1, 0.5, 0.9])
    std = np.ones(3)
    truth = np.asarray([0.0, 0.5, 1.0])
    coverage, width = bounded_scaled_interval_coverage(
        mean, std, truth, scale=100.0
    )
    assert coverage == 1.0
    assert width == 1.0
