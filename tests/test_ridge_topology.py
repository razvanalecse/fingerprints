import numpy as np

from fingerprint_reconstruction.metrics.ridge_topology import (
    binarize_ridges,
    count_minutiae,
    orientation_curvature,
    skeleton_from_ridges,
    summarize_ridge_topology,
)
from fingerprint_reconstruction.preprocessing.orientation import estimate_orientation_field


def _ridge_grating(shape, theta, period=10.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    normal_coordinate = -np.sin(theta) * xx + np.cos(theta) * yy
    return (0.5 + 0.45 * np.cos(2.0 * np.pi * normal_coordinate / period)).astype(np.float32)


def test_binarize_ridges_marks_a_thin_dark_line_not_the_light_background():
    # A blob as large as the adaptive window has no local contrast to detect
    # against (the window mostly sees the blob itself) -- a thin ridge-like
    # line, much narrower than block_size, is the realistic case.
    image = np.full((40, 40), 0.9, dtype=np.float64)
    image[19:21, 5:35] = 0.1  # a thin horizontal ridge line
    roi = np.ones_like(image, dtype=bool)
    ridge = binarize_ridges(image, roi, block_size=15, offset=0.1)
    assert ridge[20, 20]
    assert not ridge[2, 2]


def test_count_minutiae_detects_a_ridge_ending_on_an_open_line():
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[10, 5:15] = True  # a straight open segment: two endpoints
    roi = np.ones_like(skeleton, dtype=bool)
    result = count_minutiae(skeleton, roi, erosion_px=0)
    assert result["endings"] >= 2
    assert result["bifurcations"] == 0


def test_count_minutiae_detects_a_bifurcation_on_a_y_shape():
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[10, 2:10] = True  # stem into the junction at (10, 9)
    for step in range(6):
        skeleton[10 - step, 9 + step] = True  # upper branch
        skeleton[10 + step, 9 + step] = True  # lower branch
    roi = np.ones_like(skeleton, dtype=bool)
    result = count_minutiae(skeleton, roi, erosion_px=0)
    assert result["bifurcations"] >= 1


def test_count_minutiae_erosion_excludes_boundary_endpoints():
    skeleton = np.zeros((30, 30), dtype=bool)
    skeleton[15, 0:30] = True  # spans edge-to-edge: endpoints sit on the ROI border
    roi = np.ones_like(skeleton, dtype=bool)
    eroded = count_minutiae(skeleton, roi, erosion_px=5)
    unmasked = count_minutiae(skeleton, roi, erosion_px=0)
    assert eroded["endings"] <= unmasked["endings"]


def test_orientation_curvature_is_near_zero_for_a_uniform_field():
    theta = np.full((30, 30), 0.7, dtype=np.float64)
    valid = np.ones_like(theta, dtype=bool)
    roi = valid
    curvature = orientation_curvature(theta, valid, roi)
    assert curvature < 1e-6


def test_orientation_curvature_is_higher_for_a_spatially_varying_field():
    yy, xx = np.mgrid[:30, :30]
    varying_theta = (xx / 30.0) * np.pi  # sweeps across the full orientation range
    uniform_theta = np.full((30, 30), 0.7, dtype=np.float64)
    valid = np.ones((30, 30), dtype=bool)
    roi = valid
    assert orientation_curvature(varying_theta, valid, roi) > orientation_curvature(uniform_theta, valid, roi)


def test_summarize_ridge_topology_runs_end_to_end_on_a_synthetic_grating():
    image = _ridge_grating((96, 96), theta=0.3, period=9.0)
    roi = np.zeros(image.shape, dtype=bool)
    roi[16:-16, 16:-16] = True
    field = estimate_orientation_field(image, coherence_threshold=0.1)
    summary = summarize_ridge_topology(image, roi, field.theta, field.coherence, field.valid)
    assert summary.ridge_pixel_count > 0
    assert np.isfinite(summary.orientation_coherence_mean)
    assert np.isfinite(summary.orientation_curvature_mean)
    # A clean synthetic grating should be highly coherent (close to parallel lines).
    assert summary.orientation_coherence_mean > 0.5


def test_skeleton_from_ridges_is_thin():
    image = np.full((40, 40), 0.9, dtype=np.float64)
    image[18:22, 5:35] = 0.1  # a thick horizontal bar
    roi = np.ones_like(image, dtype=bool)
    skeleton = skeleton_from_ridges(image, roi, block_size=15, offset=0.1)
    # A thinned bar should have far fewer "on" pixels than the 4-pixel-thick original.
    assert skeleton.sum() < (image[18:22, 5:35] < 0.5).sum()
