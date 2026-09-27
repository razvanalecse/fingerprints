"""Simulate latent-print-like degradation of a clean fingerprint exemplar.

SD302 real latent/exemplar pairs do not provide pixel-exact ground truth;
`registered_approximate` is a
*different impression*, approximately registered). This module instead
degrades a **single** clean exemplar image into a synthetic "latent-like"
version -- elastic warp (skin deformation), blur, contrast reduction, and
noise, leaving the undegraded input as exact pixel ground truth for masked
regions. The resulting synthetic track complements, rather than replaces,
the real-latent tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class DegradationConfig:
    elastic_alpha: float = 12.0
    elastic_sigma: float = 6.0
    blur_sigma: float = 0.6
    contrast_factor: float = 0.55
    noise_std: float = 0.04
    background_darkening: float = 0.12


def elastic_warp(image: np.ndarray, rng: np.random.Generator, *, alpha: float, sigma: float) -> np.ndarray:
    """Smooth random displacement field warp (Simard-style elastic transform).

    Simulates the non-rigid skin/surface deformation between how a finger
    naturally looks and how it deposits onto a surface -- the same physical
    effect motivating this project's non-rigid-registration check (master
    table section 16), applied here as a forward *simulation* instead of an
    inverse *registration* problem.
    """

    if image.ndim != 2:
        raise ValueError("elastic_warp expects a 2D image")
    if alpha < 0 or sigma <= 0:
        raise ValueError("alpha must be non-negative and sigma must be positive")
    shape = image.shape
    displacement_y = gaussian_filter(rng.uniform(-1.0, 1.0, size=shape), sigma) * alpha
    displacement_x = gaussian_filter(rng.uniform(-1.0, 1.0, size=shape), sigma) * alpha
    grid_y, grid_x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    coordinates = np.stack(
        (np.clip(grid_y + displacement_y, 0, shape[0] - 1), np.clip(grid_x + displacement_x, 0, shape[1] - 1)),
        axis=0,
    )
    return map_coordinates(image, coordinates, order=1, mode="reflect")


def simulate_latent_degradation(
    image: np.ndarray, rng: np.random.Generator, *, config: DegradationConfig = DegradationConfig()
) -> np.ndarray:
    """Turn a clean fingerprint exemplar into a synthetic "latent-like" image.

    Pipeline: elastic warp -> Gaussian blur -> contrast reduction (real
    latents are typically lower-contrast than a clean rolled/plain
    capture) -> additive noise -> mild background darkening (latents are
    often lifted against non-white backgrounds, unlike ink-on-white-card
    exemplars). Every step is intentionally mild and physically motivated,
    not adversarial; this is meant to resemble a plausible latent, not to
    maximally confuse a model.
    """

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("image must be a finite 2D array")
    if not 0.0 <= array.min() and array.max() <= 1.0 + 1e-6:
        pass  # already normalized; degradation below tolerates minor float slack
    warped = elastic_warp(array, rng, alpha=config.elastic_alpha, sigma=config.elastic_sigma)
    blurred = gaussian_filter(warped, sigma=config.blur_sigma)
    contrast_reduced = 0.5 + (blurred - 0.5) * config.contrast_factor
    background_darkened = contrast_reduced - config.background_darkening * (1.0 - contrast_reduced)
    noisy = background_darkened + rng.normal(0.0, config.noise_std, size=array.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)
