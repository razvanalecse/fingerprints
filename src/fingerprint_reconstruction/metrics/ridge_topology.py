"""Skeleton-based ridge continuity and reconstruction-artifact diagnostics.

Some reconstructions contain coherent, near-horizontal bands that reduce
orientation-field error against a coarse reference without reconstructing
genuine ridge topology (bifurcations, ridge endings, natural curvature).
Orientation error alone cannot detect this failure mode: a locally uniform
stripe pattern can have low orientation error if its dominant angle happens
to match the true local orientation, while still being topologically wrong.

These tools are diagnostic rather than forensic-grade. They flag gross
artifacts for comparisons between real and reconstructed regions and must not
be used to certify minutiae accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects, skeletonize


@dataclass(frozen=True)
class RidgeTopologySummary:
    minutiae_count: int
    minutiae_density_per_1000px: float
    ridge_pixel_count: int
    orientation_coherence_mean: float
    orientation_curvature_mean: float


def binarize_ridges(
    image: np.ndarray,
    roi: np.ndarray,
    *,
    block_size: int = 15,
    offset: float = 0.02,
) -> np.ndarray:
    """Return a boolean ridge mask via local adaptive thresholding.

    Ridges are darker than their local neighbourhood in this project's pixel
    convention (ink-on-paper: low value = ridge, high value = background),
    matching ``ink_mean_threshold`` in
    :func:`fingerprint_reconstruction.preprocessing.orientation.estimate_foreground_mask`.
    """

    array = np.asarray(image, dtype=np.float64)
    region = np.asarray(roi, dtype=bool)
    if array.shape != region.shape:
        raise ValueError("image and roi must have equal shapes")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be odd and >=3")
    local_threshold = threshold_local(array, block_size=block_size, offset=offset)
    ridge = (array < local_threshold) & region
    return remove_small_objects(ridge, min_size=4)


def skeleton_from_ridges(image: np.ndarray, roi: np.ndarray, **binarize_kwargs) -> np.ndarray:
    """Binarize then thin to a 1-pixel-wide ridge skeleton."""

    ridge = binarize_ridges(image, roi, **binarize_kwargs)
    return skeletonize(ridge)


def count_minutiae(skeleton: np.ndarray, roi: np.ndarray, *, erosion_px: int = 3) -> dict[str, float]:
    """Crossing-number minutiae count, restricted to an eroded interior ROI.

    Erosion excludes skeleton pixels near the ROI boundary, where the ROI
    cut itself creates spurious ridge endings unrelated to genuine ridge
    topology -- a standard, if approximate, practical correction.
    """

    skeleton = np.asarray(skeleton, dtype=bool)
    region = np.asarray(roi, dtype=bool)
    if skeleton.shape != region.shape:
        raise ValueError("skeleton and roi must have equal shapes")
    interior = ndimage.binary_erosion(region, iterations=erosion_px) if erosion_px > 0 else region
    padded = np.pad(skeleton, 1, mode="constant", constant_values=False)
    # 8-neighbour cyclic order for the crossing-number transition count.
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ys, xs = np.nonzero(skeleton & interior)
    endings = bifurcations = 0
    for y, x in zip(ys, xs):
        py, px = y + 1, x + 1
        neighbours = [int(padded[py + dy, px + dx]) for dy, dx in offsets]
        crossing_number = sum(
            abs(neighbours[i] - neighbours[(i + 1) % 8]) for i in range(8)
        ) // 2
        if crossing_number == 1:
            endings += 1
        elif crossing_number == 3:
            bifurcations += 1
    interior_pixels = int(interior.sum())
    total = endings + bifurcations
    density = 1000.0 * total / interior_pixels if interior_pixels > 0 else float("nan")
    return {
        "endings": float(endings),
        "bifurcations": float(bifurcations),
        "total": float(total),
        "density_per_1000px": density,
        "interior_pixels": float(interior_pixels),
    }


def orientation_curvature(theta: np.ndarray, valid: np.ndarray, roi: np.ndarray) -> float:
    r"""Mean local angular change of the orientation field within the ROI.

    Real ridges curve; a degenerate "painted stripe" reconstruction has a
    near-constant orientation over large areas. Computed as the circular
    gradient magnitude of the doubled orientation angle (doubling removes
    the pi-periodicity of undirected ridge orientation before differencing),
    averaged over valid ROI pixels. Higher means more locally varying
    (curved) orientation; a suspiciously low value relative to a real
    reference region is the signal of interest, not an absolute threshold.
    """

    field = np.asarray(theta, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.asarray(roi, dtype=bool)
    if field.shape != mask.shape:
        raise ValueError("theta, valid and roi must have equal shapes")
    if not mask.any():
        return float("nan")
    doubled_cos, doubled_sin = np.cos(2.0 * field), np.sin(2.0 * field)
    gradient_magnitude = np.zeros_like(field)
    for component in (doubled_cos, doubled_sin):
        gy, gx = np.gradient(component)
        gradient_magnitude += gy**2 + gx**2
    gradient_magnitude = np.sqrt(gradient_magnitude)
    return float(gradient_magnitude[mask].mean())


def summarize_ridge_topology(
    image: np.ndarray,
    roi: np.ndarray,
    orientation_theta: np.ndarray,
    orientation_coherence: np.ndarray,
    orientation_valid: np.ndarray,
    **binarize_kwargs,
) -> RidgeTopologySummary:
    """One-call summary: skeleton, minutiae, coherence, and curvature for an ROI."""

    skeleton = skeleton_from_ridges(image, roi, **binarize_kwargs)
    minutiae = count_minutiae(skeleton, roi)
    region = np.asarray(roi, dtype=bool)
    valid_mask = np.asarray(orientation_valid, dtype=bool) & region
    coherence_mean = (
        float(np.asarray(orientation_coherence)[valid_mask].mean()) if valid_mask.any() else float("nan")
    )
    curvature_mean = orientation_curvature(orientation_theta, orientation_valid, roi)
    return RidgeTopologySummary(
        minutiae_count=int(minutiae["total"]),
        minutiae_density_per_1000px=minutiae["density_per_1000px"],
        ridge_pixel_count=int(skeleton[region].sum()),
        orientation_coherence_mean=coherence_mean,
        orientation_curvature_mean=curvature_mean,
    )
