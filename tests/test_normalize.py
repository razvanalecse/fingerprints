import numpy as np
import pytest
from PIL import Image

from fingerprint_reconstruction.preprocessing.normalize import load_grayscale, normalize_grayscale


def test_uint8_normalization_uses_dtype_range():
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    result = normalize_grayscale(image)
    assert result.dtype == np.float32
    assert result[0, 0] == 0.0
    assert result[0, 2] == 1.0
    assert result[0, 1] == pytest.approx(127 / 255)


def test_float_input_is_not_minmax_scaled():
    image = np.array([[0.2, 0.6]], dtype=np.float64)
    result = normalize_grayscale(image)
    assert np.allclose(result, image)


@pytest.mark.parametrize(
    "image",
    [
        np.array([[np.nan]], dtype=np.float32),
        np.array([[-0.1, 0.5]], dtype=np.float32),
        np.array([[0.5, 1.1]], dtype=np.float32),
    ],
)
def test_invalid_float_images_are_rejected(image):
    with pytest.raises(ValueError):
        normalize_grayscale(image)


def test_load_grayscale_resizes_in_height_width_order(tmp_path):
    array = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    path = tmp_path / "image.bmp"
    Image.fromarray(array).save(path)
    result = load_grayscale(path, output_shape=(16, 24))
    assert result.shape == (16, 24)
    assert result.dtype == np.float32


def test_rgba_normalization_composites_transparency_on_white():
    image = np.array(
        [[[0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 255, 128]]],
        dtype=np.uint8,
    )
    result = normalize_grayscale(image)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[0, 2] == pytest.approx(1.0)


def test_load_grayscale_respects_alpha_channel(tmp_path):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[2:6, 2:6, :3] = 0
    rgba[2:6, 2:6, 3] = 255
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    result = load_grayscale(path)
    assert np.all(result[:2] == 1.0)
    assert np.all(result[2:6, 2:6] == 0.0)


def test_load_grayscale_preserves_16bit_png_information(tmp_path):
    values = np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
    path = tmp_path / "nist16.png"
    Image.fromarray(values).save(path)

    loaded = load_grayscale(path)
    resized = load_grayscale(path, output_shape=(4, 4), interpolation="bilinear")

    assert loaded.dtype == np.float32
    assert loaded[0, 0] == pytest.approx(0.0)
    assert loaded[-1, -1] == pytest.approx(1.0)
    assert np.unique(loaded).size == 4
    assert resized.shape == (4, 4)
    assert float(resized.max()) > float(resized.min())
