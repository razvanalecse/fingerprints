"""Deterministic geometric augmentation applied before synthetic masking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    rotation_degrees: float = 25.0
    translation_fraction: float = 0.08
    scale_range: tuple[float, float] = (0.9, 1.1)
    background: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, object]]) -> Optional["AugmentationConfig"]:
        if not values or not values.get("enabled", True):
            return None
        scale = values.get("scale_range", (0.9, 1.1))
        result = cls(
            horizontal_flip=bool(values.get("horizontal_flip", True)),
            rotation_degrees=float(values.get("rotation_degrees", 25.0)),
            translation_fraction=float(values.get("translation_fraction", 0.08)),
            scale_range=(float(scale[0]), float(scale[1])),
            background=float(values.get("background", 1.0)),
        )
        if result.rotation_degrees < 0 or result.translation_fraction < 0:
            raise ValueError("augmentation ranges must be non-negative")
        if not 0 < result.scale_range[0] <= result.scale_range[1]:
            raise ValueError("scale_range must be positive and ordered")
        return result


def augmentation_rng(base_seed: int, sample_id: str, epoch: int) -> np.random.Generator:
    digest = hashlib.sha256(f"aug|{base_seed}|{sample_id}|{epoch}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def augment_image(
    image: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply a reproducible flip/rotation/scale/translation about image centre."""

    height, width = image.shape
    if config.horizontal_flip and rng.random() < 0.5:
        image = image[:, ::-1]
    angle = np.deg2rad(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    scale = rng.uniform(*config.scale_range)
    shift = rng.uniform(-config.translation_fraction, config.translation_fraction, size=2)
    shift *= np.array([height, width])
    cosine, sine = np.cos(angle) / scale, np.sin(angle) / scale
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    centre = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
    offset = centre - matrix @ centre - shift
    warped = affine_transform(
        np.ascontiguousarray(image, dtype=np.float32),
        matrix,
        offset=offset,
        order=1,
        mode="constant",
        cval=config.background,
    )
    return np.clip(warped, 0.0, 1.0).astype(np.float32)
