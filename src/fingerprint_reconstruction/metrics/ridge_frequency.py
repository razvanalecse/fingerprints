"""Local spectral diagnostics for fingerprint ridge spacing.

The estimator is deliberately rotation invariant: each window is transformed
to the Fourier domain and power is pooled in radial frequency bins. It is an
evaluation/EDA tool, not an identity feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates


@dataclass(frozen=True)
class RidgeFrequencyEstimate:
    """Accepted local dominant frequencies, measured in cycles per pixel."""

    frequencies: np.ndarray
    peak_prominences: np.ndarray
    centers_yx: np.ndarray

    @property
    def periods_pixels(self) -> np.ndarray:
        return 1.0 / np.maximum(self.frequencies, np.finfo(np.float32).eps)


def _oriented_profile_peak(
    patch: np.ndarray,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[float, float]:
    """Estimate periodicity along the structure-tensor ridge normal.

    Collapsing several parallel profiles suppresses non-ridge texture. This is
    more selective than radial averaging, which can confuse fingerprint-boundary
    energy and interpolation artefacts with a ridge-frequency peak.
    """

    size = patch.shape[0]
    patch = patch.astype(np.float32)
    gy, gx = np.gradient(patch)
    tensor_window = np.outer(np.hanning(size), np.hanning(size))
    jxx = float(np.sum(tensor_window * gx * gx))
    jyy = float(np.sum(tensor_window * gy * gy))
    jxy = float(np.sum(tensor_window * gx * gy))
    tensor_strength = np.hypot(jxx - jyy, 2.0 * jxy)
    if tensor_strength <= 1e-10:
        return float("nan"), 0.0
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    normal = np.asarray((np.sin(normal_angle), np.cos(normal_angle)))
    tangent = np.asarray((-normal[1], normal[0]))
    center = np.asarray(((size - 1) / 2.0, (size - 1) / 2.0))
    along = np.linspace(-(size - 2) / 2.0, (size - 2) / 2.0, size - 1)
    across = np.linspace(-size / 6.0, size / 6.0, max(5, size // 3))
    coordinates = (
        center[:, None, None]
        + normal[:, None, None] * along[None, :, None]
        + tangent[:, None, None] * across[None, None, :]
    )
    profiles = map_coordinates(
        patch, (coordinates[0], coordinates[1]), order=1, mode="reflect"
    )
    profile = profiles.mean(axis=1)
    # Remove slowly varying illumination/foreground-shape energy. Without this
    # step the first non-DC bin is often selected instead of ridge periodicity.
    profile = profile - gaussian_filter1d(profile, sigma=2.0, mode="reflect")
    profile = (profile - profile.mean()) * np.hanning(profile.size)
    # Zero padding makes the peak location less quantized, without claiming
    # additional physical resolution.
    transform_size = max(256, 8 * profile.size)
    power = np.abs(np.fft.rfft(profile, n=transform_size)) ** 2
    frequencies = np.fft.rfftfreq(transform_size)
    eligible = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    if not eligible.any() or float(power[eligible].max()) <= 0:
        return float("nan"), 0.0
    eligible_indices = np.flatnonzero(eligible)
    peak_index = int(eligible_indices[np.argmax(power[eligible])])
    baseline = float(np.median(power[eligible]))
    prominence = float(power[peak_index] / max(baseline, 1e-12))
    return float(frequencies[peak_index]), prominence


def estimate_local_ridge_frequency(
    image: np.ndarray,
    region: np.ndarray,
    *,
    patch_size: int = 32,
    stride: int = 8,
    minimum_region_fraction: float = 0.60,
    minimum_frequency: float = 0.04,
    maximum_frequency: float = 0.40,
    minimum_peak_prominence: float = 1.5,
) -> RidgeFrequencyEstimate:
    """Estimate local ridge frequencies in windows supported by ``region``.

    The broad default search interval corresponds to periods of 4--25 pixels.
    It only excludes DC/background and pixel noise; the task-specific ridge
    band must be estimated empirically from the training split.
    """

    image = np.asarray(image, dtype=np.float32)
    region = np.asarray(region, dtype=bool)
    if image.ndim != 2 or image.shape != region.shape:
        raise ValueError("image and region must be same-shaped 2D arrays")
    if patch_size < 16 or patch_size % 2:
        raise ValueError("patch_size must be an even integer >= 16")
    if stride <= 0 or not 0.0 < minimum_region_fraction <= 1.0:
        raise ValueError("invalid stride or region fraction")
    if not 0.0 < minimum_frequency < maximum_frequency <= 0.5:
        raise ValueError("frequency interval must lie inside (0, 0.5]")

    frequencies, prominences, centers = [], [], []
    height, width = image.shape
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            local_region = region[top : top + patch_size, left : left + patch_size]
            if float(local_region.mean()) < minimum_region_fraction:
                continue
            patch = image[top : top + patch_size, left : left + patch_size]
            frequency, prominence = _oriented_profile_peak(
                patch,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
            if np.isfinite(frequency) and prominence >= minimum_peak_prominence:
                frequencies.append(frequency)
                prominences.append(prominence)
                centers.append((top + patch_size // 2, left + patch_size // 2))
    return RidgeFrequencyEstimate(
        frequencies=np.asarray(frequencies, dtype=np.float32),
        peak_prominences=np.asarray(prominences, dtype=np.float32),
        centers_yx=np.asarray(centers, dtype=np.int32).reshape(-1, 2),
    )


def paired_ridge_frequency_error(
    reference: np.ndarray,
    estimate: np.ndarray,
    region: np.ndarray,
    **kwargs,
) -> dict[str, float]:
    """Compare local dominant frequency on reference-selected windows."""

    reference_result = estimate_local_ridge_frequency(reference, region, **kwargs)
    if reference_result.frequencies.size == 0:
        return {
            "ridge_frequency_mae_cpx": float("nan"),
            "ridge_period_mae_pixels": float("nan"),
            "ridge_frequency_relative_mae": float("nan"),
            "ridge_frequency_windows": 0.0,
        }
    patch_size = int(kwargs.get("patch_size", 32))
    minimum_frequency = float(kwargs.get("minimum_frequency", 0.04))
    maximum_frequency = float(kwargs.get("maximum_frequency", 0.40))
    estimated_frequencies, reference_frequencies = [], []
    half = patch_size // 2
    estimate = np.asarray(estimate, dtype=np.float32)
    for (center_y, center_x), ref_frequency in zip(
        reference_result.centers_yx, reference_result.frequencies
    ):
        patch = estimate[
            center_y - half : center_y + half,
            center_x - half : center_x + half,
        ]
        frequency, _ = _oriented_profile_peak(
            patch,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        if np.isfinite(frequency):
            estimated_frequencies.append(frequency)
            reference_frequencies.append(float(ref_frequency))
    ref = np.asarray(reference_frequencies, dtype=np.float64)
    est = np.asarray(estimated_frequencies, dtype=np.float64)
    absolute = np.abs(est - ref)
    return {
        "ridge_frequency_mae_cpx": float(absolute.mean()),
        "ridge_period_mae_pixels": float(np.abs(1.0 / est - 1.0 / ref).mean()),
        "ridge_frequency_relative_mae": float((absolute / ref).mean()),
        "ridge_frequency_windows": float(ref.size),
    }
