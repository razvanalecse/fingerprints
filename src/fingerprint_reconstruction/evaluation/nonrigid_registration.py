"""Thin-plate-spline (TPS) registration, compared to affine via leave-one-out.

Motivated by an external review's concern that SD302's affine registration
may leave residual geometric error that grows away from the correspondence
points -- exactly the pattern the confidence-weighted losses in this project
already try to compensate for (`docs/nist302_ablation_master_table.md`
section 4). A non-rigid (thin-plate-spline) warp has more degrees of freedom
and could in principle fit the true, locally-varying skin deformation better.

The manifest's own `affine_rmse_mm` is an **in-sample** residual (the same
points used to fit the transform), which is not a fair way to ask "does more
flexibility help" -- a TPS with N free control points fits its own N training
points exactly (residual identically zero) regardless of whether it captures
anything real. The only fair comparison is **leave-one-correspondence-out**:
fit on N-1 points, predict the held-out point, measure the residual, for
both methods identically. This module provides exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator

from fingerprint_reconstruction.evaluation.registration import (
    RegistrationError,
    TransformDiagnostics,
    _validate_points,
    fit_affine_transform,
    transform_points,
)


@dataclass(frozen=True)
class ThinPlateSplineTransform:
    """A fitted TPS point-mapper; callable like the affine matrix transform."""

    _interpolator: RBFInterpolator

    def __call__(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise RegistrationError("points must have shape Nx2")
        return self._interpolator(array)


def fit_thin_plate_spline_transform(
    source: np.ndarray, target: np.ndarray, *, smoothing: float = 0.0
) -> ThinPlateSplineTransform:
    """Fit a 2D thin-plate-spline mapping source points onto target points."""

    source_array, target_array = _validate_points(source, target, minimum=3)
    interpolator = RBFInterpolator(
        source_array, target_array, kernel="thin_plate_spline", smoothing=smoothing
    )
    return ThinPlateSplineTransform(interpolator)


def leave_one_out_residuals_mm(
    source: np.ndarray,
    target: np.ndarray,
    fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """Fit on all-but-one correspondence, predict the held-out point, in mm.

    ``fit`` must accept (source_subset, target_subset) and return a callable
    mapping Nx2 source points to Nx2 predicted target points -- both
    :func:`fit_thin_plate_spline_transform` and a thin wrapper around
    :func:`fingerprint_reconstruction.evaluation.registration.fit_affine_transform`
    satisfy this interface (see ``affine_point_mapper`` below).
    """

    source_array, target_array = _validate_points(source, target, minimum=4)
    count = source_array.shape[0]
    residuals = np.empty(count, dtype=np.float64)
    keep = np.ones(count, dtype=bool)
    for index in range(count):
        keep[:] = True
        keep[index] = False
        mapper = fit(source_array[keep], target_array[keep])
        predicted = mapper(source_array[index : index + 1])[0]
        residuals[index] = np.linalg.norm(predicted - target_array[index])
    return residuals / 100.0  # stored units are 0.01mm


def affine_point_mapper(source: np.ndarray, target: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt :func:`fit_affine_transform` to the ``fit`` interface above."""

    diagnostics = fit_affine_transform(source, target)
    return lambda points: transform_points(points, diagnostics.matrix)


def thin_plate_spline_mapper(source: np.ndarray, target: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt :func:`fit_thin_plate_spline_transform` to the ``fit`` interface above."""

    return fit_thin_plate_spline_transform(source, target)


def summarize_residuals_mm(residuals_mm: np.ndarray) -> dict[str, float]:
    array = np.asarray(residuals_mm, dtype=np.float64)
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.square(array)))),
        "median_mm": float(np.median(array)),
        "p95_mm": float(np.quantile(array, 0.95)) if array.size >= 2 else float(array[0]),
        "count": int(array.size),
    }


def warp_image_thin_plate_spline(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    output_shape: Tuple[int, int],
    order: int = 1,
    cval: float = 1.0,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Warp ``image`` (defined in the *target*-point frame) into the
    *source*-point (output) frame via TPS.

    Mirrors :func:`fingerprint_reconstruction.evaluation.registration.warp_image_output_to_input`'s
    calling convention exactly: pass ``source_points`` (output/callee frame)
    and ``target_points`` (input frame, the frame ``image`` is defined in) in
    the same order you would pass to ``fit_affine_transform`` before calling
    ``warp_image_output_to_input`` with its resulting matrix. The point-fit
    functions here already produce a *direct* source->target map (not a
    matrix to invert), so no argument swap is needed -- for every
    output-frame pixel, this map gives the input-frame coordinate to sample.
    """

    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise RegistrationError("image must be a non-empty 2D array")
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0 or order not in range(6):
        raise RegistrationError("invalid output shape or interpolation order")
    inverse_map = fit_thin_plate_spline_transform(source_points, target_points, smoothing=smoothing)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    query_xy = np.column_stack((grid_x.ravel().astype(np.float64), grid_y.ravel().astype(np.float64)))
    sampled_xy = inverse_map(query_xy)
    from scipy.ndimage import map_coordinates

    # map_coordinates expects (row, col) = (y, x) order.
    coordinates = np.stack((sampled_xy[:, 1], sampled_xy[:, 0]), axis=0)
    warped = map_coordinates(
        array.astype(np.float64), coordinates, order=order, mode="constant", cval=float(cval)
    )
    return np.ascontiguousarray(warped.reshape(height, width))
