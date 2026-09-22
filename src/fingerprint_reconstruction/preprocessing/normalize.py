"""Image loading and intensity normalization with explicit contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a finite float32 grayscale image in [0, 1].

    Integer inputs are scaled by their dtype range. Floating-point inputs must
    already lie in [0, 1]; silently min-max scaling individual fingerprints
    would destroy cross-image intensity information and is therefore forbidden.
    """

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] not in (3, 4):
            raise ValueError("color images must have 3 or 4 channels")
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            color = (array.astype(np.float64) - info.min) / float(info.max - info.min)
        elif np.issubdtype(array.dtype, np.floating):
            color = array.astype(np.float64)
            if not np.isfinite(color).all():
                raise ValueError("floating-point image contains NaN or infinity")
            tolerance = 1e-6
            if float(color.min()) < -tolerance or float(color.max()) > 1.0 + tolerance:
                raise ValueError("floating-point image must already be normalized to [0, 1]")
            color = np.clip(color, 0.0, 1.0)
        else:
            raise TypeError(f"unsupported image dtype: {array.dtype}")

        rgb = color[..., :3]
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if color.shape[-1] == 4:
            alpha = color[..., 3]
            gray = alpha * gray + (1.0 - alpha)  # transparent pixels -> white
        return np.ascontiguousarray(gray, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("expected a 2D grayscale or HxWx{3,4} color image")
    if array.size == 0:
        raise ValueError("image cannot be empty")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = (array.astype(np.float32) - info.min) / float(info.max - info.min)
    elif np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("floating-point image contains NaN or infinity")
        tolerance = 1e-6
        if float(result.min()) < -tolerance or float(result.max()) > 1.0 + tolerance:
            raise ValueError("floating-point image must already be normalized to [0, 1]")
        np.clip(result, 0.0, 1.0, out=result)
    else:
        raise TypeError(f"unsupported image dtype: {array.dtype}")

    if not np.isfinite(result).all():
        raise ValueError("normalized image contains NaN or infinity")
    return np.ascontiguousarray(result, dtype=np.float32)


def load_grayscale(
    path: Path,
    *,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Load an image as normalized grayscale, optionally resizing to HxW."""

    interpolation_modes = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if interpolation not in interpolation_modes:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    with Image.open(Path(path)) as image:
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            array = np.asarray(
                Image.alpha_composite(background, rgba).convert("L"), dtype=np.uint8
            )
            normalized = normalize_grayscale(array)
        elif image.mode.startswith("I;16"):
            # Pillow's I;16 -> L conversion clips values above 255. NIST SD302
            # contains genuine 16-bit PNGs whose useful values are commonly in
            # the thousands, so that conversion would produce all-white data.
            normalized = normalize_grayscale(np.array(image, dtype=np.uint16, copy=True))
        else:
            normalized = normalize_grayscale(
                np.asarray(image.convert("L"), dtype=np.uint8)
            )

    if output_shape is None:
        return normalized

    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("output_shape dimensions must be positive")
    # Resize normalized data in floating point to preserve 16-bit information.
    float_image = Image.fromarray(normalized)
    resized = float_image.resize(
        (width, height), resample=interpolation_modes[interpolation]
    )
    result = np.asarray(resized, dtype=np.float32)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)
