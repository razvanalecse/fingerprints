"""Geometric registration diagnostics from declared point correspondences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import affine_transform


class RegistrationError(ValueError):
    """Raised when a transform cannot be estimated without ambiguity."""


@dataclass(frozen=True)
class TransformDiagnostics:
    matrix: np.ndarray
    residuals_0_01mm: np.ndarray
    rmse_mm: float
    median_mm: float
    p95_mm: float


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> TransformDiagnostics:
    """Fit x'=a*x-b*y+tx, y'=b*x+a*y+ty by least squares."""

    source_array, target_array = _validate_points(source, target, minimum=2)
    count = source_array.shape[0]
    design = np.zeros((2 * count, 4), dtype=np.float64)
    observations = target_array.reshape(-1)
    for index, (x, y) in enumerate(source_array):
        design[2 * index] = (x, -y, 1.0, 0.0)
        design[2 * index + 1] = (y, x, 0.0, 1.0)
    parameters, _, rank, _ = np.linalg.lstsq(design, observations, rcond=None)
    if rank < 4:
        raise RegistrationError("similarity transform is rank deficient")
    a, b, tx, ty = parameters
    matrix = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)
    return _diagnostics(matrix, source_array, target_array)


def fit_affine_transform(source: np.ndarray, target: np.ndarray) -> TransformDiagnostics:
    """Fit an unconstrained 2D affine transform by least squares."""

    source_array, target_array = _validate_points(source, target, minimum=3)
    design = np.column_stack(
        (source_array, np.ones(source_array.shape[0], dtype=np.float64))
    )
    parameters, _, rank, _ = np.linalg.lstsq(design, target_array, rcond=None)
    if rank < 3:
        raise RegistrationError("affine transform is rank deficient")
    matrix = parameters.T
    return _diagnostics(matrix, source_array, target_array)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 homogeneous affine matrix to Nx2 points."""

    array = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise RegistrationError("points must have shape Nx2")
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise RegistrationError("matrix must be finite with shape 2x3")
    homogeneous = np.column_stack((array, np.ones(array.shape[0], dtype=np.float64)))
    return homogeneous @ transform.T


def warp_image_output_to_input(
    image: np.ndarray,
    output_to_input_xy: np.ndarray,
    *,
    output_shape: Tuple[int, int],
    order: int = 1,
    cval: float = 1.0,
) -> np.ndarray:
    """Warp an image when the 2x3 matrix maps output (x,y) to input (x,y).

    This convention is deliberate for exemplar-to-latent registration: the
    fitted latent->exemplar point transform directly supplies the inverse-map
    required to sample the exemplar in the latent image frame.
    """

    array = np.asarray(image)
    transform = np.asarray(output_to_input_xy, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise RegistrationError("image must be a non-empty 2D array")
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise RegistrationError("matrix must be finite with shape 2x3")
    height, width = int(output_shape[0]), int(output_shape[1])
    if height <= 0 or width <= 0 or order not in range(6):
        raise RegistrationError("invalid output shape or interpolation order")
    # scipy uses (row=y, col=x), whereas the fitted matrix uses (x, y).
    matrix_yx = np.array(
        [[transform[1, 1], transform[1, 0]], [transform[0, 1], transform[0, 0]]],
        dtype=np.float64,
    )
    offset_yx = np.array([transform[1, 2], transform[0, 2]], dtype=np.float64)
    warped = affine_transform(
        array,
        matrix_yx,
        offset=offset_yx,
        output_shape=(height, width),
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    )
    return np.ascontiguousarray(warped)


def rescale_output_to_input_transform(
    matrix: np.ndarray,
    *,
    native_output_shape: Tuple[int, int],
    native_input_shape: Tuple[int, int],
    resized_output_shape: Tuple[int, int],
    resized_input_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert a native-pixel inverse map for center-aligned resized images."""

    transform = np.asarray(matrix, dtype=np.float64)
    shapes = (
        native_output_shape,
        native_input_shape,
        resized_output_shape,
        resized_input_shape,
    )
    if transform.shape != (2, 3) or any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
        raise RegistrationError("invalid transform or image shapes")
    out_scale = np.array(
        [native_output_shape[1] / resized_output_shape[1], native_output_shape[0] / resized_output_shape[0]],
        dtype=np.float64,
    )
    in_scale = np.array(
        [native_input_shape[1] / resized_input_shape[1], native_input_shape[0] / resized_input_shape[0]],
        dtype=np.float64,
    )
    out_center_offset = (out_scale - 1.0) / 2.0
    in_center_offset = (in_scale - 1.0) / 2.0
    result = np.empty((2, 3), dtype=np.float64)
    result[:, :2] = (
        np.diag(1.0 / in_scale) @ transform[:, :2] @ np.diag(out_scale)
    )
    result[:, 2] = (
        transform[:, :2] @ out_center_offset + transform[:, 2] - in_center_offset
    ) / in_scale
    return result


def _validate_points(
    source: np.ndarray, target: np.ndarray, *, minimum: int
) -> Tuple[np.ndarray, np.ndarray]:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if source_array.shape != target_array.shape or source_array.ndim != 2:
        raise RegistrationError("source and target must have the same Nx2 shape")
    if source_array.shape[1:] != (2,) or source_array.shape[0] < minimum:
        raise RegistrationError(f"at least {minimum} corresponding 2D points are required")
    if not np.isfinite(source_array).all() or not np.isfinite(target_array).all():
        raise RegistrationError("correspondences contain NaN or infinity")
    return source_array, target_array


def _diagnostics(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> TransformDiagnostics:
    residuals = np.linalg.norm(transform_points(source, matrix) - target, axis=1)
    return TransformDiagnostics(
        matrix=matrix,
        residuals_0_01mm=residuals,
        rmse_mm=float(np.sqrt(np.mean(np.square(residuals))) / 100.0),
        median_mm=float(np.median(residuals) / 100.0),
        p95_mm=float(np.quantile(residuals, 0.95) / 100.0),
    )
