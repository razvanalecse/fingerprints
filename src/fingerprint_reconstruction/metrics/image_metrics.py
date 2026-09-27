"""Region-aware image metrics for fingerprint reconstruction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity


def _validate_arrays(
    reference: np.ndarray, estimate: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=np.float64)
    estimate_array = np.asarray(estimate, dtype=np.float64)
    region_array = np.asarray(region, dtype=bool)
    if reference_array.shape != estimate_array.shape or reference_array.shape != region_array.shape:
        raise ValueError("reference, estimate, and region must have identical shapes")
    if reference_array.ndim != 2:
        raise ValueError("metrics require 2D grayscale images")
    if not reference_array.size or not region_array.any():
        raise ValueError("evaluation region cannot be empty")
    if not np.isfinite(reference_array).all() or not np.isfinite(estimate_array).all():
        raise ValueError("images contain NaN or infinity")
    return reference_array, estimate_array, region_array


def region_image_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    region: np.ndarray,
    *,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Compute errors over a region and average the SSIM map in that region.

    The SSIM convention is explicit: a standard full-image SSIM map is first
    computed using an 11x11 Gaussian window, then its local values are averaged
    only at pixels selected by ``region``. Windows can therefore include context
    from both sides of a region boundary; this is not claimed to be a canonical
    masked SSIM definition.
    """

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    reference_array, estimate_array, region_array = _validate_arrays(
        reference, estimate, region
    )
    errors = estimate_array[region_array] - reference_array[region_array]
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range**2 / mse))
    _, ssim_map = structural_similarity(
        reference_array,
        estimate_array,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )
    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim_map_mean": float(np.mean(ssim_map[region_array])),
        "pixels": float(np.count_nonzero(region_array)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }
