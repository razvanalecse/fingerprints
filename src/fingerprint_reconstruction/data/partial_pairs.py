"""Reproducible construction of full/partial fingerprint training pairs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from fingerprint_reconstruction.preprocessing.masks import MaskFamily, MaskGenerator, MaskSpec
from fingerprint_reconstruction.preprocessing.normalize import normalize_grayscale


@dataclass(frozen=True)
class PartialPair:
    full: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    metadata: Dict[str, Any]


def stable_condition_seed(
    *,
    base_seed: int,
    sample_id: str,
    family: MaskFamily,
    observed_fraction: float,
    replicate: int,
) -> int:
    """Derive a platform-independent 64-bit seed from an experiment condition."""

    if base_seed < 0 or replicate < 0:
        raise ValueError("base_seed and replicate must be non-negative")
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    canonical = (
        f"v1\0{base_seed}\0{sample_id}\0{family.value}\0"
        f"{observed_fraction:.12f}\0{replicate}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big", signed=False)


def stable_nested_mask_seed(
    *, base_seed: int, sample_id: str, family: MaskFamily, replicate: int
) -> int:
    """Derive a seed shared by every observed fraction in one nested mask set.

    Unlike :func:`stable_condition_seed`, this seed intentionally excludes
    ``observed_fraction``. It is used for repeated-measures experiments where
    mask geometry must remain fixed while the observed area changes.
    """

    if base_seed < 0 or replicate < 0:
        raise ValueError("base_seed and replicate must be non-negative")
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    canonical = (
        f"nested-v1\0{base_seed}\0{sample_id}\0{family.value}\0{replicate}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big", signed=False)


def build_partial_pair(
    image: np.ndarray,
    *,
    sample_id: str,
    family: MaskFamily,
    observed_fraction: float,
    replicate: int,
    base_seed: int = 1729,
    num_fragments: Optional[int] = None,
    missing_fill_value: float = 0.0,
) -> PartialPair:
    """Create X, Y, M while preserving exact observed pixels."""

    if not 0.0 <= missing_fill_value <= 1.0:
        raise ValueError("missing_fill_value must lie in [0, 1]")
    full = normalize_grayscale(image)
    seed = stable_condition_seed(
        base_seed=base_seed,
        sample_id=sample_id,
        family=family,
        observed_fraction=observed_fraction,
        replicate=replicate,
    )
    result = MaskGenerator(full.shape).generate(
        MaskSpec(
            family=family,
            observed_fraction=observed_fraction,
            seed=seed,
            num_fragments=num_fragments,
        )
    )
    mask = result.mask
    observed = np.where(mask == 1, full, np.float32(missing_fill_value)).astype(np.float32)
    metadata: Dict[str, Any] = {
        **result.metadata,
        "sample_id": sample_id,
        "base_seed": base_seed,
        "condition_seed": seed,
        "replicate": replicate,
        "missing_fill_value": missing_fill_value,
    }
    return PartialPair(full=full, observed=observed, mask=mask, metadata=metadata)


def enforce_data_consistency(
    reconstruction: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return M*Y + (1-M)*X_hat for arrays with identical HxW shapes."""

    reconstruction_array = np.asarray(reconstruction, dtype=np.float32)
    observed_array = np.asarray(observed, dtype=np.float32)
    mask_array = np.asarray(mask)
    if not (
        reconstruction_array.shape == observed_array.shape == mask_array.shape
        and reconstruction_array.ndim == 2
    ):
        raise ValueError("reconstruction, observed, and mask must share a 2D shape")
    if not np.isin(mask_array, (0, 1)).all():
        raise ValueError("mask must be binary")
    return np.where(mask_array == 1, observed_array, reconstruction_array).astype(np.float32)


def observed_fraction_report(mask: np.ndarray, fingerprint_roi: np.ndarray) -> Dict[str, float]:
    """Report observed/missing fractions on both canvas and fingerprint ROI."""

    mask_array = np.asarray(mask)
    roi = np.asarray(fingerprint_roi, dtype=bool)
    if mask_array.shape != roi.shape or mask_array.ndim != 2:
        raise ValueError("mask and fingerprint_roi must share a 2D shape")
    if not np.isin(mask_array, (0, 1)).all():
        raise ValueError("mask must be binary")
    roi_pixels = int(roi.sum())
    if roi_pixels == 0:
        raise ValueError("fingerprint_roi cannot be empty")
    observed_canvas = float(mask_array.mean())
    observed_roi = float(mask_array.astype(bool)[roi].mean())
    return {
        "observed_fraction_canvas": observed_canvas,
        "missing_fraction_canvas": 1.0 - observed_canvas,
        "observed_fraction_fingerprint_roi": observed_roi,
        "missing_fraction_fingerprint_roi": 1.0 - observed_roi,
        "fingerprint_roi_pixels": float(roi_pixels),
    }
