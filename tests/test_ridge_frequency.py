import numpy as np
import pytest

from fingerprint_reconstruction.metrics.ridge_frequency import (
    estimate_local_ridge_frequency,
    paired_ridge_frequency_error,
)


def _grating(period: float, theta: float = 0.0, size: int = 128) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    coordinate = x * np.cos(theta) + y * np.sin(theta)
    return (0.5 + 0.45 * np.cos(2.0 * np.pi * coordinate / period)).astype(np.float32)


@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 3, np.pi / 2])
def test_frequency_estimator_is_rotation_invariant(theta):
    image = _grating(period=8.0, theta=theta)
    result = estimate_local_ridge_frequency(image, np.ones_like(image, dtype=bool))
    assert result.frequencies.size > 20
    assert float(np.median(result.frequencies)) == pytest.approx(1.0 / 8.0, abs=0.02)


def test_paired_frequency_error_detects_wrong_spacing():
    region = np.ones((128, 128), dtype=bool)
    matching = paired_ridge_frequency_error(_grating(8), _grating(8), region)
    mismatching = paired_ridge_frequency_error(_grating(8), _grating(12), region)
    assert matching["ridge_frequency_mae_cpx"] == pytest.approx(0.0)
    assert mismatching["ridge_frequency_mae_cpx"] > 0.02
    assert mismatching["ridge_period_mae_pixels"] > 2.0


def test_frequency_estimator_rejects_uniform_background():
    image = np.ones((64, 64), dtype=np.float32)
    result = estimate_local_ridge_frequency(image, np.ones_like(image, dtype=bool))
    assert result.frequencies.size == 0
