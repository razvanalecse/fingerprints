import numpy as np
import pytest

from fingerprint_reconstruction.preprocessing.orientation import (
    OrientationField,
    circular_orientation_distance,
    estimate_foreground_mask,
    estimate_orientation_field,
    orientation_error,
    smooth_axial_orientation,
    suppress_background,
)


def _ridge_grating(shape, theta, period=10.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    normal_coordinate = -np.sin(theta) * xx + np.cos(theta) * yy
    return (0.5 + 0.45 * np.cos(2.0 * np.pi * normal_coordinate / period)).astype(np.float32)


@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3])
def test_orientation_estimator_recovers_synthetic_ridge_direction(theta):
    image = _ridge_grating((128, 128), theta)
    field = estimate_orientation_field(image, coherence_threshold=0.1)
    interior = np.zeros(image.shape, dtype=bool)
    interior[16:-16, 16:-16] = True
    reference = OrientationField(
        theta=np.full(image.shape, theta, dtype=np.float32),
        coherence=np.ones(image.shape, dtype=np.float32),
        valid=interior,
    )
    error = orientation_error(reference, field, region_mask=interior)
    assert error < 0.02
    assert float(field.coherence[interior].mean()) > 0.8


def test_orientation_distance_has_pi_periodicity():
    theta = np.linspace(0, np.pi, 20, endpoint=False)
    assert np.allclose(circular_orientation_distance(theta, theta + np.pi), 0.0)
    assert np.allclose(circular_orientation_distance(theta, theta + np.pi / 2), 2.0)


def test_constant_image_has_no_valid_orientation():
    field = estimate_orientation_field(np.ones((32, 32), dtype=np.float32))
    assert not field.valid.any()
    assert np.all(field.coherence == 0)


def test_orientation_error_can_be_restricted_to_region():
    shape = (16, 16)
    reference = OrientationField(
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=bool),
    )
    estimate_theta = np.zeros(shape, dtype=np.float32)
    estimate_theta[:, 8:] = np.pi / 2
    estimate = OrientationField(
        estimate_theta,
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=bool),
    )
    left = np.zeros(shape, dtype=bool)
    left[:, :8] = True
    right = ~left
    assert orientation_error(reference, estimate, region_mask=left) == pytest.approx(0.0)
    assert orientation_error(reference, estimate, region_mask=right) == pytest.approx(2.0)


def test_foreground_mask_rejects_uniform_background():
    image = np.ones((128, 128), dtype=np.float32)
    image[32:96, 28:100] = _ridge_grating((64, 72), np.pi / 5, period=8.0)
    foreground = estimate_foreground_mask(image)
    assert float(foreground[40:88, 36:92].mean()) > 0.95
    background = foreground.copy()
    background[24:104, 20:108] = False
    assert float(background.mean()) < 0.01


def test_foreground_cleanup_does_not_fill_scanner_frame():
    image = np.ones((128, 128), dtype=np.float32)
    image[:5, :] = 0.0
    image[-5:, :] = 0.0
    image[:, :5] = 0.0
    image[:, -5:] = 0.0
    image[34:98, 30:102] = _ridge_grating((64, 72), np.pi / 7, period=9.0)
    foreground = estimate_foreground_mask(image)
    assert float(foreground[42:90, 38:94].mean()) > 0.95
    assert not foreground[:7, :].any()
    assert not foreground[-7:, :].any()
    assert not foreground[:, :7].any()
    assert not foreground[:, -7:].any()
    assert float(foreground.mean()) < 0.45


def test_axial_smoothing_respects_zero_pi_wraparound():
    theta = np.zeros((33, 33), dtype=np.float32)
    theta[:, :16] = 0.02
    theta[:, 16:] = np.pi - 0.02
    smoothed = smooth_axial_orientation(theta, sigma=4.0)
    distance_to_horizontal = np.minimum(smoothed, np.pi - smoothed)
    assert float(distance_to_horizontal[:, 14:19].max()) < 0.03


def test_background_suppression_preserves_only_foreground_pixels():
    image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[2:6, 2:6] = True
    cleaned = suppress_background(image, foreground)
    assert np.array_equal(cleaned[foreground], image[foreground])
    assert np.all(cleaned[~foreground] == 1.0)
