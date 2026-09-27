import numpy as np
import pytest

from fingerprint_reconstruction.preprocessing.degradation import (
    DegradationConfig,
    elastic_warp,
    simulate_latent_degradation,
)


def _synthetic_print(shape=(64, 64)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (0.5 + 0.45 * np.cos(2.0 * np.pi * yy / 8.0)).astype(np.float64)


def test_elastic_warp_with_zero_alpha_is_the_identity():
    image = _synthetic_print()
    rng = np.random.default_rng(0)
    warped = elastic_warp(image, rng, alpha=0.0, sigma=4.0)
    np.testing.assert_allclose(warped, image, atol=1e-6)


def test_elastic_warp_preserves_shape_and_stays_bounded():
    image = _synthetic_print()
    rng = np.random.default_rng(1)
    warped = elastic_warp(image, rng, alpha=8.0, sigma=5.0)
    assert warped.shape == image.shape
    assert np.isfinite(warped).all()
    assert warped.min() >= image.min() - 1e-6
    assert warped.max() <= image.max() + 1e-6


def test_elastic_warp_rejects_non_2d_input():
    with pytest.raises(ValueError):
        elastic_warp(np.zeros((4, 4, 3)), np.random.default_rng(0), alpha=1.0, sigma=1.0)


def test_simulate_latent_degradation_stays_in_unit_interval():
    image = _synthetic_print()
    rng = np.random.default_rng(2)
    degraded = simulate_latent_degradation(image, rng)
    assert degraded.min() >= 0.0
    assert degraded.max() <= 1.0
    assert degraded.shape == image.shape


def test_simulate_latent_degradation_reduces_contrast():
    image = _synthetic_print()
    rng = np.random.default_rng(3)
    config = DegradationConfig(elastic_alpha=2.0, elastic_sigma=6.0, blur_sigma=0.3, contrast_factor=0.4, noise_std=0.01, background_darkening=0.0)
    degraded = simulate_latent_degradation(image, rng, config=config)
    assert degraded.std() < image.std()


def test_simulate_latent_degradation_is_reproducible_given_the_same_seed():
    image = _synthetic_print()
    first = simulate_latent_degradation(image, np.random.default_rng(42))
    second = simulate_latent_degradation(image, np.random.default_rng(42))
    np.testing.assert_allclose(first, second)


def test_simulate_latent_degradation_rejects_non_finite_input():
    image = _synthetic_print()
    image[0, 0] = np.nan
    with pytest.raises(ValueError):
        simulate_latent_degradation(image, np.random.default_rng(0))
