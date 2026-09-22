"""Fingerprint ridge-orientation estimation and circular error metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
)
from scipy.spatial import ConvexHull, Delaunay, QhullError


@dataclass(frozen=True)
class OrientationField:
    """Ridge tangent orientation in [0, pi) and local coherence in [0, 1]."""

    theta: np.ndarray
    coherence: np.ndarray
    valid: np.ndarray
    foreground: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 5:
        raise ValueError("orientation estimation requires a 2D image at least 5x5")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinity")
    return array


def estimate_foreground_mask(
    image: np.ndarray,
    *,
    local_sigma: float = 3.0,
    contrast_threshold: float = 0.055,
    mean_range: tuple[float, float] = (0.06, 0.97),
    ink_mean_threshold: float = 0.88,
    border_margin_fraction: float = 0.05,
    min_component_fraction: float = 0.002,
    morphology_radius_fraction: float = 0.03,
    contour_smoothing_sigma: float = 2.0,
    convex_envelope: bool = True,
) -> np.ndarray:
    """Estimate where ridge structure is present, excluding uniform background.

    The decision combines local standard deviation with a permissive local-mean
    interval. Small isolated responses are removed and the outer image frame is
    excluded because scanner/crop borders can otherwise look like coherent
    ridges to a structure tensor.
    """

    array = _validate_image(image)
    if local_sigma <= 0:
        raise ValueError("local_sigma must be positive")
    if contrast_threshold < 0:
        raise ValueError("contrast_threshold must be non-negative")
    low, high = mean_range
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("mean_range must satisfy 0 <= low < high <= 1")
    if not 0.0 <= ink_mean_threshold <= 1.0:
        raise ValueError("ink_mean_threshold must lie in [0, 1]")
    if not 0.0 <= border_margin_fraction < 0.5:
        raise ValueError("border_margin_fraction must lie in [0, 0.5)")
    if not 0.0 <= min_component_fraction <= 1.0:
        raise ValueError("min_component_fraction must lie in [0, 1]")
    if morphology_radius_fraction < 0:
        raise ValueError("morphology_radius_fraction must be non-negative")
    if contour_smoothing_sigma < 0:
        raise ValueError("contour_smoothing_sigma must be non-negative")

    local_mean = gaussian_filter(array, sigma=local_sigma, mode="reflect")
    local_second_moment = gaussian_filter(array * array, sigma=local_sigma, mode="reflect")
    local_std = np.sqrt(np.maximum(local_second_moment - local_mean * local_mean, 0.0))
    texture_support = (
        (local_std >= contrast_threshold)
        & (local_mean >= low)
        & (local_mean <= high)
    )
    # Dark saturated regions may belong to the fingerprint contact area even
    # when individual ridges are not resolvable there. Include them in the ROI
    # candidate, but later select the component with strongest texture support.
    foreground = texture_support | (local_mean <= ink_mean_threshold)

    # Remove the scanner/crop frame before topology-changing morphology. If
    # fill_holes runs first, a rectangular frame can be mistaken for the outer
    # boundary of one enormous foreground component.
    margin_y = int(np.ceil(border_margin_fraction * array.shape[0]))
    margin_x = int(np.ceil(border_margin_fraction * array.shape[1]))
    if margin_y:
        foreground[:margin_y] = False
        foreground[-margin_y:] = False
        texture_support[:margin_y] = False
        texture_support[-margin_y:] = False
    if margin_x:
        foreground[:, :margin_x] = False
        foreground[:, -margin_x:] = False
        texture_support[:, :margin_x] = False
        texture_support[:, -margin_x:] = False

    radius = max(1, int(round(morphology_radius_fraction * min(array.shape))))
    grid_y, grid_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    closing_structure = grid_x * grid_x + grid_y * grid_y <= radius * radius
    opening_radius = max(1, radius // 2)
    open_y, open_x = np.mgrid[
        -opening_radius : opening_radius + 1,
        -opening_radius : opening_radius + 1,
    ]
    opening_structure = open_x * open_x + open_y * open_y <= opening_radius**2
    foreground = binary_closing(foreground, structure=closing_structure)
    foreground = binary_opening(foreground, structure=opening_structure)

    components, count = label(foreground)
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        texture_counts = np.bincount(
            components.ravel(), weights=texture_support.ravel(), minlength=sizes.size
        )
        texture_counts[0] = 0
        largest = int(np.argmax(texture_counts))
        minimum = max(1, int(np.ceil(min_component_fraction * array.size)))
        foreground = (
            components == largest
            if texture_counts[largest] > 0 and sizes[largest] >= minimum
            else np.zeros_like(foreground)
        )

    # Fingerprint contact regions are approximately convex. The envelope is
    # used as an anatomical ROI, not as evidence that ridge orientation is
    # valid everywhere inside it; the latter remains gated by tensor quality.
    if convex_envelope and np.count_nonzero(foreground) >= 3:
        coordinates = np.column_stack(np.nonzero(foreground))
        try:
            hull = ConvexHull(coordinates)
            triangulation = Delaunay(coordinates[hull.vertices])
            grid_y, grid_x = np.indices(array.shape)
            grid = np.column_stack((grid_y.ravel(), grid_x.ravel()))
            foreground = (triangulation.find_simplex(grid) >= 0).reshape(array.shape)
        except QhullError:
            pass

    foreground = binary_fill_holes(foreground)
    if contour_smoothing_sigma > 0 and foreground.any():
        smoothed = gaussian_filter(
            foreground.astype(np.float64),
            sigma=contour_smoothing_sigma,
            mode="constant",
        )
        foreground = smoothed >= 0.5

    # Reassert frame exclusion after morphology and contour smoothing.
    if margin_y:
        foreground[:margin_y] = False
        foreground[-margin_y:] = False
    if margin_x:
        foreground[:, :margin_x] = False
        foreground[:, -margin_x:] = False
    return foreground.astype(bool, copy=False)


def smooth_axial_orientation(
    theta: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    sigma: float = 2.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Smooth a pi-periodic orientation field in doubled-angle space."""

    angles = np.asarray(theta, dtype=np.float64)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if weights is None:
        weight_array = np.ones_like(angles)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != angles.shape:
            raise ValueError("weights must have the same shape as theta")
        if np.any(weight_array < 0) or not np.isfinite(weight_array).all():
            raise ValueError("weights must be finite and non-negative")
    if sigma == 0:
        return np.mod(angles, np.pi).astype(np.float32)

    vector_x = gaussian_filter(weight_array * np.cos(2.0 * angles), sigma=sigma, mode="reflect")
    vector_y = gaussian_filter(weight_array * np.sin(2.0 * angles), sigma=sigma, mode="reflect")
    support = gaussian_filter(weight_array, sigma=sigma, mode="reflect")
    vector_x = np.divide(vector_x, support, out=np.zeros_like(vector_x), where=support > epsilon)
    vector_y = np.divide(vector_y, support, out=np.zeros_like(vector_y), where=support > epsilon)
    return np.mod(0.5 * np.arctan2(vector_y, vector_x), np.pi).astype(np.float32)


def suppress_background(
    image: np.ndarray,
    foreground: np.ndarray,
    *,
    background_value: float = 1.0,
) -> np.ndarray:
    """Preserve fingerprint pixels and replace all pixels outside its ROI."""

    array = _validate_image(image)
    mask = np.asarray(foreground, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError("foreground must have the same shape as image")
    if not 0.0 <= background_value <= 1.0:
        raise ValueError("background_value must lie in [0, 1]")
    return np.where(mask, array, background_value).astype(np.float32)


def estimate_orientation_field(
    image: np.ndarray,
    *,
    gradient_sigma: float = 1.0,
    tensor_sigma: float = 3.0,
    smooth_sigma: float = 2.0,
    coherence_threshold: float = 0.20,
    use_foreground_mask: bool = True,
    epsilon: float = 1e-8,
) -> OrientationField:
    """Estimate ridge tangents using a smoothed structure tensor.

    Image gradients describe the normal to a ridge. The principal gradient
    orientation is therefore rotated by pi/2 to obtain the ridge tangent. A
    second circular smoothing step operates on doubled angles, respecting the
    axial periodicity theta == theta + pi.
    """

    array = _validate_image(image)
    for name, value in (
        ("gradient_sigma", gradient_sigma),
        ("tensor_sigma", tensor_sigma),
        ("smooth_sigma", smooth_sigma),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 <= coherence_threshold <= 1.0:
        raise ValueError("coherence_threshold must lie in [0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    grad_y = gaussian_filter(array, sigma=gradient_sigma, order=(1, 0), mode="reflect")
    grad_x = gaussian_filter(array, sigma=gradient_sigma, order=(0, 1), mode="reflect")
    jxx = gaussian_filter(grad_x * grad_x, sigma=tensor_sigma, mode="reflect")
    jyy = gaussian_filter(grad_y * grad_y, sigma=tensor_sigma, mode="reflect")
    jxy = gaussian_filter(grad_x * grad_y, sigma=tensor_sigma, mode="reflect")

    anisotropy_x = jxx - jyy
    anisotropy_y = 2.0 * jxy
    energy = jxx + jyy
    coherence = np.sqrt(anisotropy_x**2 + anisotropy_y**2) / (energy + epsilon)
    coherence = np.clip(coherence, 0.0, 1.0)

    gradient_angle = 0.5 * np.arctan2(anisotropy_y, anisotropy_x)
    ridge_angle = np.mod(gradient_angle + np.pi / 2.0, np.pi)

    ridge_angle = smooth_axial_orientation(
        ridge_angle,
        weights=coherence,
        sigma=smooth_sigma,
        epsilon=epsilon,
    )
    foreground = estimate_foreground_mask(array) if use_foreground_mask else np.ones(array.shape, dtype=bool)
    valid = foreground & (coherence >= coherence_threshold) & (energy > epsilon)
    return OrientationField(
        theta=ridge_angle.astype(np.float32),
        coherence=coherence.astype(np.float32),
        valid=valid,
        foreground=foreground,
        energy=energy.astype(np.float32),
    )


def circular_orientation_distance(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Return 1-cos(2 delta), invariant to orientation shifts of pi."""

    reference_array = np.asarray(reference, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("reference and estimate must have identical shapes")
    return 1.0 - np.cos(2.0 * (reference_array - estimate_array))


def orientation_error(
    reference: OrientationField,
    estimate: OrientationField,
    *,
    region_mask: Optional[np.ndarray] = None,
    weighting: str = "joint_coherence",
) -> float:
    """Aggregate circular orientation error over jointly valid pixels.

    ``joint_coherence`` weights a location by the geometric mean of reference
    and estimated coherence. ``uniform`` gives all jointly valid pixels equal
    weight. The function fails rather than reporting a misleading value when no
    valid structural support remains.
    """

    if reference.theta.shape != estimate.theta.shape:
        raise ValueError("orientation fields must have identical shapes")
    valid = reference.valid & estimate.valid
    if region_mask is not None:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("region_mask shape does not match orientation fields")
        valid &= mask
    if not np.any(valid):
        raise ValueError("no jointly valid pixels for orientation error")

    if weighting == "joint_coherence":
        weights = np.sqrt(reference.coherence * estimate.coherence)
    elif weighting == "uniform":
        weights = np.ones(valid.shape, dtype=np.float32)
    else:
        raise ValueError(f"unsupported weighting: {weighting}")

    weights = np.where(valid, weights, 0.0).astype(np.float64)
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("orientation weights sum to zero")
    distances = circular_orientation_distance(reference.theta, estimate.theta)
    return float(np.sum(weights * distances) / denominator)
