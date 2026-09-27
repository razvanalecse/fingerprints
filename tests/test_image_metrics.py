import numpy as np
import pytest

from fingerprint_reconstruction.metrics.image_metrics import region_image_metrics


def test_region_metrics_use_only_selected_pixels():
    reference = np.zeros((16, 16), dtype=np.float32)
    estimate = np.zeros_like(reference)
    region = np.zeros_like(reference, dtype=bool)
    region[:, 8:] = True
    estimate[:, 8:] = 0.5
    metrics = region_image_metrics(reference, estimate, region)
    assert metrics["mse"] == pytest.approx(0.25)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["psnr"] == pytest.approx(10.0 * np.log10(4.0))
    assert metrics["pixels"] == 128


def test_perfect_region_metrics_have_zero_error_and_infinite_psnr():
    image = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    metrics = region_image_metrics(image, image.copy(), np.ones_like(image, dtype=bool))
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim_map_mean"] == pytest.approx(1.0)
