import numpy as np
import pytest

from fingerprint_reconstruction.evaluation.registration import (
    RegistrationError,
    fit_affine_transform,
    fit_similarity_transform,
    rescale_output_to_input_transform,
    transform_points,
    warp_image_output_to_input,
)


def test_similarity_transform_recovers_exact_mapping() -> None:
    source = np.array([[0, 0], [10, 0], [0, 20], [7, 9]], dtype=float)
    expected = np.array([[1.5, -0.5, 30], [0.5, 1.5, -12]], dtype=float)
    target = transform_points(source, expected)
    result = fit_similarity_transform(source, target)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-10)
    assert result.rmse_mm == pytest.approx(0.0, abs=1e-12)


def test_affine_transform_recovers_shear() -> None:
    source = np.array([[0, 0], [10, 0], [0, 20], [7, 9]], dtype=float)
    expected = np.array([[1.2, 0.3, 5], [-0.2, 0.9, 8]], dtype=float)
    target = transform_points(source, expected)
    result = fit_affine_transform(source, target)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-10)


def test_degenerate_affine_points_are_rejected() -> None:
    points = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    with pytest.raises(RegistrationError, match="rank deficient"):
        fit_affine_transform(points, points)


def test_warp_uses_output_to_input_xy_convention() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    image[2, 3] = 1.0
    # Output x=1,y=1 samples input x=3,y=2.
    output_to_input = np.array([[1, 0, 2], [0, 1, 1]], dtype=float)
    warped = warp_image_output_to_input(
        image, output_to_input, output_shape=(5, 6), order=0, cval=0
    )
    assert warped[1, 1] == 1.0
    assert int(np.count_nonzero(warped)) == 1


def test_rescaled_identity_accounts_for_different_resize_factors() -> None:
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    result = rescale_output_to_input_transform(
        identity,
        native_output_shape=(100, 200),
        native_input_shape=(100, 200),
        resized_output_shape=(50, 50),
        resized_input_shape=(25, 25),
    )
    np.testing.assert_allclose(result[:, :2], [[0.5, 0], [0, 0.5]])
