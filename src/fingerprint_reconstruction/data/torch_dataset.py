"""PyTorch dataset for deterministic SOCOFing partial-observation training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from fingerprint_reconstruction.analysis.socofing_eda import read_manifest
from fingerprint_reconstruction.data.augmentation import (
    AugmentationConfig,
    augment_image,
    augmentation_rng,
)
from fingerprint_reconstruction.data.partial_pairs import build_partial_pair
from fingerprint_reconstruction.preprocessing.masks import MaskFamily
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale
from fingerprint_reconstruction.preprocessing.orientation import (
    estimate_foreground_mask,
    estimate_orientation_field,
)


def valid_mask_conditions(
    families: Sequence[MaskFamily], observed_fractions: Sequence[float]
) -> List[Tuple[MaskFamily, float]]:
    conditions = []
    for family in families:
        for fraction in observed_fractions:
            value = float(fraction)
            if family == MaskFamily.SEVERE_PARTIAL and value > 0.30:
                continue
            conditions.append((family, value))
    if not conditions:
        raise ValueError("at least one valid mask condition is required")
    return conditions


class SocofingPartialDataset(Dataset):
    """Generate deterministic partial pairs on demand for one leakage-safe split."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        image_root: Path,
        split: str,
        output_shape: Tuple[int, int] = (128, 128),
        families: Sequence[MaskFamily] = tuple(MaskFamily),
        observed_fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        base_seed: int = 1729,
        include_foreground: bool = False,
        include_structure: bool = False,
        structure_shape: Tuple[int, int] = (32, 32),
        augmentation: Optional[AugmentationConfig] = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.rows: List[Mapping[str, str]] = [
            row for row in read_manifest(manifest_path) if row["split"] == split
        ]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split {split}")
        self.image_root = Path(image_root)
        self.output_shape = tuple(int(value) for value in output_shape)
        self.conditions = valid_mask_conditions(families, observed_fractions)
        self.base_seed = int(base_seed)
        self.include_foreground = bool(include_foreground)
        self.include_structure = bool(include_structure)
        self.structure_shape = tuple(int(value) for value in structure_shape)
        if min(self.structure_shape) <= 0:
            raise ValueError("structure_shape dimensions must be positive")
        if augmentation is not None and (include_foreground or include_structure):
            raise ValueError("augmentation is incompatible with cached foreground/structure targets")
        self.augmentation = augmentation
        self._foreground_cache: dict[int, torch.Tensor] = {}
        self._structure_cache: dict[int, torch.Tensor] = {}
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        row = self.rows[index]
        condition_index = (index + self.epoch * 104729) % len(self.conditions)
        family, observed_fraction = self.conditions[condition_index]
        image = load_grayscale(
            self.image_root / row["relative_path"], output_shape=self.output_shape
        )
        if self.augmentation is not None:
            image = augment_image(
                image,
                self.augmentation,
                augmentation_rng(self.base_seed, row["sample_id"], self.epoch),
            )
        pair = build_partial_pair(
            image,
            sample_id=row["sample_id"],
            family=family,
            observed_fraction=observed_fraction,
            replicate=self.epoch,
            base_seed=self.base_seed,
            num_fragments=4 if family == MaskFamily.DISCONNECTED_FRAGMENTS else None,
        )
        observed = torch.from_numpy(pair.observed).unsqueeze(0)
        mask = torch.from_numpy(pair.mask.astype("float32", copy=False)).unsqueeze(0)
        target = torch.from_numpy(pair.full).unsqueeze(0)
        result = {
            "conditioning": torch.cat((observed, mask), dim=0),
            "observed": observed,
            "mask": mask,
            "target": target,
            "sample_id": row["sample_id"],
            "mask_family": family.value,
            "observed_fraction": torch.tensor(observed_fraction, dtype=torch.float32),
        }
        if self.include_foreground:
            if index not in self._foreground_cache:
                foreground = estimate_foreground_mask(pair.full)
                self._foreground_cache[index] = torch.from_numpy(
                    foreground.astype("float32", copy=False)
                ).unsqueeze(0)
            result["foreground"] = self._foreground_cache[index]
        if self.include_structure:
            if index not in self._structure_cache:
                field = estimate_orientation_field(pair.full)
                support = torch.from_numpy(field.foreground.astype("float32"))[None, None]
                coherence = torch.from_numpy(field.coherence.astype("float32"))[None, None]
                valid = torch.from_numpy(field.valid.astype("float32"))[None, None]
                theta = torch.from_numpy(field.theta.astype("float32"))[None, None]
                weight = coherence * valid
                weight_down = F.interpolate(weight, self.structure_shape, mode="area")
                vector_x = F.interpolate(weight * torch.cos(2.0 * theta), self.structure_shape, mode="area")
                vector_y = F.interpolate(weight * torch.sin(2.0 * theta), self.structure_shape, mode="area")
                vector_x = vector_x / weight_down.clamp_min(1e-6)
                vector_y = vector_y / weight_down.clamp_min(1e-6)
                norm = torch.sqrt(vector_x.square() + vector_y.square()).clamp_min(1e-6)
                vector_x, vector_y = vector_x / norm, vector_y / norm
                support_down = F.interpolate(support, self.structure_shape, mode="area")
                coherence_down = F.interpolate(coherence * valid, self.structure_shape, mode="area") / F.interpolate(valid, self.structure_shape, mode="area").clamp_min(1e-6)
                valid_down = (weight_down >= 0.05).float()
                self._structure_cache[index] = torch.cat(
                    (support_down, vector_x, vector_y, coherence_down, valid_down), dim=1
                )[0]
            result["structure"] = self._structure_cache[index]
        return result
