import numpy as np
import pytest

from fingerprint_reconstruction.evaluation.nonrigid_registration import (
    affine_point_mapper,
    fit_thin_plate_spline_transform,
    leave_one_out_residuals_mm,
    summarize_residuals_mm,
    thin_plate_spline_mapper,
    warp_image_thin_plate_spline,
)
from fingerprint_reconstruction.evaluation.registration import (
    transform_points,
    warp_image_output_to_input,
)


def _affine_points(n=8, seed=0):
    rng = np.random.default_rng(seed)
    source = rng.uniform(0, 1000, size=(n, 2))
    matrix = np.array([[1.1, 0.2, 30.0], [-0.15, 0.95, -12.0]])
    target = transform_points(source, matrix)
    return source, target, matrix


def test_tps_exactly_interpolates_its_own_training_points():
    source, target, _ = _affine_points()
    mapper = fit_thin_plate_spline_transform(source, target)
    predicted = mapper(source)
    np.testing.assert_allclose(predicted, target, atol=1e-6)


def test_tps_recovers_a_purely_affine_relationship_well():
    # TPS's basis includes the affine/polynomial part, so noiseless affine
    # data should be recovered almost exactly even off the training points.
    source, target, matrix = _affine_points()
    mapper = fit_thin_plate_spline_transform(source, target)
    probe = np.array([[500.0, 500.0], [12.0, 970.0]])
    expected = transform_points(probe, matrix)
    np.testing.assert_allclose(mapper(probe), expected, atol=1e-4)


def test_leave_one_out_residuals_are_near_zero_for_noiseless_affine_data_both_methods():
    source, target, _ = _affine_points(n=8)
    affine_residuals = leave_one_out_residuals_mm(source, target, affine_point_mapper)
    tps_residuals = leave_one_out_residuals_mm(source, target, thin_plate_spline_mapper)
    assert summarize_residuals_mm(affine_residuals)["rmse_mm"] < 1e-6
    assert summarize_residuals_mm(tps_residuals)["rmse_mm"] < 1e-3


def test_leave_one_out_detects_nonlinear_deformation_affine_cannot_fit():
    # A genuine local (quadratic) warp: affine has no way to represent this,
    # so its leave-one-out error should be much larger than TPS's, which can
    # bend to follow the curvature.
    rng = np.random.default_rng(1)
    source = rng.uniform(0, 1000, size=(10, 2))
    target = source.copy()
    target[:, 0] += 0.00015 * (source[:, 1] - 500.0) ** 2  # nonlinear x-warp
    affine_residuals = leave_one_out_residuals_mm(source, target, affine_point_mapper)
    tps_residuals = leave_one_out_residuals_mm(source, target, thin_plate_spline_mapper)
    affine_rmse = summarize_residuals_mm(affine_residuals)["rmse_mm"]
    tps_rmse = summarize_residuals_mm(tps_residuals)["rmse_mm"]
    assert affine_rmse > tps_rmse


def test_leave_one_out_requires_at_least_four_points():
    source = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    with pytest.raises(Exception):
        leave_one_out_residuals_mm(source, source, affine_point_mapper)


def test_summarize_residuals_reports_expected_keys():
    summary = summarize_residuals_mm(np.array([1.0, 2.0, 3.0, 4.0]))
    assert set(summary) == {"rmse_mm", "median_mm", "p95_mm", "count"}
    assert summary["count"] == 4


def test_tps_warp_matches_affine_warp_for_purely_affine_correspondences():
    source_points = np.array([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0], [90.0, 90.0], [50.0, 50.0]])
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])  # a simple shift
    target_points = transform_points(source_points, matrix)
    image = np.zeros((100, 100), dtype=np.float64)
    image[40:60, 40:60] = 1.0
    from fingerprint_reconstruction.evaluation.registration import fit_affine_transform

    diagnostics = fit_affine_transform(source_points, target_points)
    affine_warped = warp_image_output_to_input(
        image, diagnostics.matrix, output_shape=(100, 100), cval=0.0
    )
    tps_warped = warp_image_thin_plate_spline(
        image, source_points, target_points, output_shape=(100, 100), cval=0.0
    )
    # Both should place the bright square at roughly the same shifted location.
    assert np.argmax(affine_warped) != 0 or affine_warped.max() > 0
    correlation = np.corrcoef(affine_warped.ravel(), tps_warped.ravel())[0, 1]
    assert correlation > 0.9
