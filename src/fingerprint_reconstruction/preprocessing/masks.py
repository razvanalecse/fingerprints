"""Controlled synthetic partial-fingerprint mask generation.

Mask semantics are fixed throughout the project:

    M[p] = 1  -> pixel p is observed
    M[p] = 0  -> pixel p is missing

All generators return exactly ``round(r * H * W)`` observed pixels. Geometry is
generated first and a deterministic rank projection enforces the requested area.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


class MaskFamily(str, Enum):
    RANDOM_RECTANGLES = "random_rectangles"
    IRREGULAR = "irregular"
    CENTRAL_MISSING = "central_missing"
    PERIPHERAL_ONLY = "peripheral_only"
    CENTRAL_ONLY = "central_only"
    DISCONNECTED_FRAGMENTS = "disconnected_fragments"
    STRIPES = "stripes"
    SEVERE_PARTIAL = "severe_partial"


@dataclass(frozen=True)
class MaskSpec:
    family: MaskFamily
    observed_fraction: float
    seed: int
    num_fragments: Optional[int] = None

    def validate(self) -> None:
        if not 0.0 < self.observed_fraction < 1.0:
            raise ValueError("observed_fraction must lie strictly between 0 and 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.family == MaskFamily.DISCONNECTED_FRAGMENTS:
            if self.num_fragments is None or self.num_fragments < 2:
                raise ValueError("disconnected_fragments requires num_fragments >= 2")
        if self.family == MaskFamily.SEVERE_PARTIAL and self.observed_fraction > 0.30:
            raise ValueError("severe_partial is defined only for observed_fraction <= 0.30")


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    metadata: Dict[str, Any]


class MaskGenerator:
    """Generate exact-area masks with reproducible geometry."""

    def __init__(self, shape: Tuple[int, int]):
        if len(shape) != 2 or min(shape) < 8:
            raise ValueError("shape must be a 2D image of at least 8 x 8 pixels")
        self.height, self.width = (int(shape[0]), int(shape[1]))
        self.yy, self.xx = np.mgrid[: self.height, : self.width]

    def generate(self, spec: MaskSpec) -> MaskResult:
        spec.validate()
        rng = np.random.default_rng(spec.seed)
        observed_score = self._generate_score(
            spec.family, spec.observed_fraction, spec.num_fragments, rng
        )
        target_observed = int(round(spec.observed_fraction * self.height * self.width))
        mask = self._project_exact_area(observed_score, target_observed, rng)
        return MaskResult(mask=mask, metadata=self._metadata(spec, mask))

    def _generate_score(
        self,
        family: MaskFamily,
        observed_fraction: float,
        num_fragments: Optional[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if family == MaskFamily.CENTRAL_MISSING:
            observed_score = self._radial_score(peripheral=True, rng=rng)
        elif family == MaskFamily.PERIPHERAL_ONLY:
            observed_score = self._peripheral_fragment_score(rng)
        elif family == MaskFamily.CENTRAL_ONLY:
            observed_score = self._radial_score(peripheral=False, rng=rng)
        elif family == MaskFamily.RANDOM_RECTANGLES:
            observed_score = self._rectangle_observed_score(
                rng, missing_fraction=1.0 - observed_fraction
            )
        elif family == MaskFamily.IRREGULAR:
            observed_score = self._irregular_observed_score(
                rng, missing_fraction=1.0 - observed_fraction
            )
        elif family == MaskFamily.DISCONNECTED_FRAGMENTS:
            observed_score = self._fragment_score(rng, int(num_fragments))
        elif family == MaskFamily.STRIPES:
            observed_score = self._stripe_observed_score(
                rng, missing_fraction=1.0 - observed_fraction
            )
        elif family == MaskFamily.SEVERE_PARTIAL:
            observed_score = self._fragment_score(rng, int(num_fragments or 3))
        else:  # pragma: no cover - Enum prevents this in normal use.
            raise ValueError(f"unsupported mask family: {family}")

        return observed_score

    def _metadata(self, spec: MaskSpec, mask: np.ndarray) -> Dict[str, Any]:
        target_observed = int(round(spec.observed_fraction * self.height * self.width))
        observed = int(mask.sum())
        return {
            **asdict(spec),
            "family": spec.family.value,
            "height": self.height,
            "width": self.width,
            "target_observed_pixels": target_observed,
            "actual_observed_pixels": observed,
            "actual_observed_fraction": observed / mask.size,
            "actual_missing_fraction": 1.0 - observed / mask.size,
            "semantics": "1=observed,0=missing",
            "algorithm_version": 1,
        }

    def generate_nested(
        self,
        *,
        family: MaskFamily,
        observed_fractions: Sequence[float],
        seed: int,
        num_fragments: Optional[int] = None,
    ) -> Dict[float, MaskResult]:
        """Generate exact-area masks with one fixed geometry across all r values."""

        fractions = sorted({float(value) for value in observed_fractions})
        if not fractions:
            raise ValueError("observed_fractions cannot be empty")
        specifications = [
            MaskSpec(family, value, seed=seed, num_fragments=num_fragments)
            for value in fractions
        ]
        for specification in specifications:
            specification.validate()
        geometry_fraction = float(np.median(fractions))
        rng = np.random.default_rng(seed)
        score = self._generate_score(
            family, geometry_fraction, num_fragments, rng
        ).astype(np.float64, copy=False).ravel()
        order = np.argsort(score + rng.uniform(0.0, 1e-9, score.size))
        nested_id = f"{family.value}:{seed}:{geometry_fraction:.6f}"
        results: Dict[float, MaskResult] = {}
        for specification in specifications:
            target = int(round(specification.observed_fraction * score.size))
            mask = np.zeros(score.size, dtype=np.uint8)
            mask[order[-target:]] = 1
            mask = mask.reshape(self.height, self.width)
            metadata = self._metadata(specification, mask)
            metadata.update(
                {
                    "nested": True,
                    "nested_mask_set_id": nested_id,
                    "geometry_reference_fraction": geometry_fraction,
                }
            )
            results[specification.observed_fraction] = MaskResult(mask, metadata)
        return results

    def generate_nested_in_roi(
        self,
        *,
        family: MaskFamily,
        observed_fractions: Sequence[float],
        seed: int,
        roi: np.ndarray,
        num_fragments: Optional[int] = None,
    ) -> Dict[float, MaskResult]:
        """Generate nested masks with exact observed area inside an evaluation ROI.

        Pixels outside ``roi`` remain unobserved. This is intended for controlled
        synthetic experiments that compare mask geometry at equal amounts of
        fingerprint information. The ROI must never be supplied to the model as
        target-derived conditioning.
        """

        region = np.asarray(roi, dtype=bool)
        if region.shape != (self.height, self.width):
            raise ValueError("roi must match the mask generator shape")
        roi_size = int(region.sum())
        if roi_size < 2:
            raise ValueError("roi must contain at least two pixels")
        fractions = sorted({float(value) for value in observed_fractions})
        if not fractions:
            raise ValueError("observed_fractions cannot be empty")
        specifications = [
            MaskSpec(family, value, seed=seed, num_fragments=num_fragments)
            for value in fractions
        ]
        for specification in specifications:
            specification.validate()
        geometry_fraction = float(np.median(fractions))
        rng = np.random.default_rng(seed)
        score = self._generate_score(
            family, geometry_fraction, num_fragments, rng
        ).astype(np.float64, copy=False)
        roi_scores = score[region] + rng.uniform(0.0, 1e-9, roi_size)
        order = np.argsort(roi_scores)
        roi_flat_indices = np.flatnonzero(region.ravel())
        nested_id = f"roi:{family.value}:{seed}:{geometry_fraction:.6f}"
        results: Dict[float, MaskResult] = {}
        for specification in specifications:
            target = int(round(specification.observed_fraction * roi_size))
            if not 0 < target < roi_size:
                raise ValueError("ROI target area must be non-empty and non-complete")
            mask_flat = np.zeros(self.height * self.width, dtype=np.uint8)
            selected_roi_positions = order[-target:]
            mask_flat[roi_flat_indices[selected_roi_positions]] = 1
            mask = mask_flat.reshape(self.height, self.width)
            metadata = self._metadata(specification, mask)
            metadata.update(
                {
                    "nested": True,
                    "fraction_domain": "fingerprint_roi",
                    "roi_pixels": roi_size,
                    "target_observed_roi_pixels": target,
                    "actual_observed_roi_pixels": int(mask[region].sum()),
                    "actual_observed_roi_fraction": float(mask[region].mean()),
                    "nested_mask_set_id": nested_id,
                    "geometry_reference_fraction": geometry_fraction,
                }
            )
            results[specification.observed_fraction] = MaskResult(mask, metadata)
        return results

    def _project_exact_area(
        self, score: np.ndarray, target: int, rng: np.random.Generator
    ) -> np.ndarray:
        if score.shape != (self.height, self.width):
            raise RuntimeError("internal score shape mismatch")
        if not 0 < target < score.size:
            raise ValueError("target area must be non-empty and non-complete")

        # Random jitter is used only as a deterministic tie-breaker; its scale is
        # too small to alter non-tied score ordering.
        flat_score = score.astype(np.float64, copy=False).ravel()
        jitter = rng.uniform(0.0, 1e-9, flat_score.size)
        selected = np.argpartition(flat_score + jitter, -target)[-target:]
        mask = np.zeros(flat_score.size, dtype=np.uint8)
        mask[selected] = 1
        return mask.reshape(self.height, self.width)

    def _radial_score(
        self, *, peripheral: bool, rng: np.random.Generator
    ) -> np.ndarray:
        cy = (self.height - 1) / 2.0 + rng.uniform(-0.04, 0.04) * self.height
        cx = (self.width - 1) / 2.0 + rng.uniform(-0.04, 0.04) * self.width
        aspect = rng.uniform(0.85, 1.15)
        distance = np.sqrt(
            ((self.yy - cy) / self.height) ** 2
            + aspect * ((self.xx - cx) / self.width) ** 2
        )
        return distance if peripheral else -distance

    def _signed_observed_score(self, missing_region: np.ndarray) -> np.ndarray:
        """Rank pixels by signed distance from a proposed missing region.

        Positive values lie in observed candidates and negative values in holes.
        Exact-area projection can therefore adjust a boundary without creating
        salt-and-pepper pixels when a primitive mask has excess area.
        """

        missing = np.asarray(missing_region, dtype=bool)
        if not missing.any() or missing.all():
            raise RuntimeError("proposed missing geometry must contain both classes")
        distance_outside = distance_transform_edt(~missing)
        distance_inside = distance_transform_edt(missing)
        return distance_outside - distance_inside

    def _rectangle_observed_score(
        self, rng: np.random.Generator, *, missing_fraction: float
    ) -> np.ndarray:
        region = np.zeros((self.height, self.width), dtype=bool)
        target = min(0.98, missing_fraction + 0.05)
        for _ in range(64):
            rh = int(rng.integers(max(2, self.height // 8), max(3, self.height // 2)))
            rw = int(rng.integers(max(2, self.width // 8), max(3, self.width // 2)))
            y0 = int(rng.integers(0, self.height - rh + 1))
            x0 = int(rng.integers(0, self.width - rw + 1))
            region[y0 : y0 + rh, x0 : x0 + rw] = True
            if float(region.mean()) >= target:
                break
        return self._signed_observed_score(region)

    def _paint_disk(
        self, canvas: np.ndarray, cy: float, cx: float, radius: float, value: float
    ) -> None:
        y0 = max(0, int(np.floor(cy - radius)))
        y1 = min(self.height, int(np.ceil(cy + radius + 1)))
        x0 = max(0, int(np.floor(cx - radius)))
        x1 = min(self.width, int(np.ceil(cx + radius + 1)))
        local_y = self.yy[y0:y1, x0:x1]
        local_x = self.xx[y0:y1, x0:x1]
        inside = (local_y - cy) ** 2 + (local_x - cx) ** 2 <= radius**2
        canvas[y0:y1, x0:x1][inside] += value

    def _irregular_observed_score(
        self, rng: np.random.Generator, *, missing_fraction: float
    ) -> np.ndarray:
        score = np.zeros((self.height, self.width), dtype=np.float64)
        scale = min(self.height, self.width)
        target = min(0.98, missing_fraction + 0.05)
        for _stroke in range(80):
            y = rng.uniform(0, self.height - 1)
            x = rng.uniform(0, self.width - 1)
            angle = rng.uniform(0, 2 * np.pi)
            steps = int(rng.integers(4, 13))
            for _ in range(steps):
                radius = rng.uniform(0.025, 0.09) * scale
                self._paint_disk(score, y, x, radius, rng.uniform(0.8, 1.2))
                angle += rng.normal(0.0, 0.65)
                length = rng.uniform(0.02, 0.10) * scale
                y = np.clip(y + np.sin(angle) * length, 0, self.height - 1)
                x = np.clip(x + np.cos(angle) * length, 0, self.width - 1)
            if float(np.mean(score > 0)) >= target:
                break
        return self._signed_observed_score(score > 0)

    def _fragment_score(self, rng: np.random.Generator, count: int) -> np.ndarray:
        score = np.full((self.height, self.width), -np.inf, dtype=np.float64)
        margin_y = 0.08 * self.height
        margin_x = 0.08 * self.width
        centers = []
        min_separation = 0.18 * min(self.height, self.width)
        for _ in range(count):
            candidate = None
            for _attempt in range(200):
                proposal = (
                    rng.uniform(margin_y, self.height - 1 - margin_y),
                    rng.uniform(margin_x, self.width - 1 - margin_x),
                )
                if all(
                    np.hypot(proposal[0] - y, proposal[1] - x) >= min_separation
                    for y, x in centers
                ):
                    candidate = proposal
                    break
            if candidate is None:
                candidate = proposal
            centers.append(candidate)

        for cy, cx in centers:
            ry = rng.uniform(0.08, 0.18) * self.height
            rx = rng.uniform(0.08, 0.18) * self.width
            local = -(((self.yy - cy) / ry) ** 2 + ((self.xx - cx) / rx) ** 2)
            score = np.maximum(score, local)
        return score

    def _peripheral_fragment_score(self, rng: np.random.Generator) -> np.ndarray:
        """Favor several distributed patches near the image boundary."""

        score = np.full((self.height, self.width), -np.inf, dtype=np.float64)
        cy, cx = (self.height - 1) / 2.0, (self.width - 1) / 2.0
        count = int(rng.integers(5, 9))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for index in range(count):
            angle = phase + 2.0 * np.pi * index / count + rng.normal(0.0, 0.12)
            center_y = cy + 0.40 * self.height * np.sin(angle)
            center_x = cx + 0.40 * self.width * np.cos(angle)
            radius_y = rng.uniform(0.10, 0.18) * self.height
            radius_x = rng.uniform(0.10, 0.18) * self.width
            local = -(
                ((self.yy - center_y) / radius_y) ** 2
                + ((self.xx - center_x) / radius_x) ** 2
            )
            score = np.maximum(score, local)
        return score

    def _stripe_observed_score(
        self, rng: np.random.Generator, *, missing_fraction: float
    ) -> np.ndarray:
        score = np.zeros((self.height, self.width), dtype=np.float64)
        diagonal = np.hypot(self.height, self.width)
        target = min(0.98, missing_fraction + 0.05)
        for _ in range(32):
            angle = rng.uniform(0, np.pi)
            normal_y, normal_x = np.sin(angle), np.cos(angle)
            offset = rng.uniform(-0.35, 0.35) * diagonal
            centered_y = self.yy - (self.height - 1) / 2.0
            centered_x = self.xx - (self.width - 1) / 2.0
            distance = np.abs(normal_y * centered_y + normal_x * centered_x - offset)
            width = rng.uniform(0.025, 0.10) * min(self.height, self.width)
            score += np.exp(-0.5 * (distance / width) ** 2)
            if float(np.mean(score > np.exp(-4.5))) >= target:
                break
        # A fixed threshold can cover the full canvas after many overlapping
        # stripes. Select the strongest responses by rank instead, preserving
        # stripe geometry while guaranteeing both observed and missing support.
        missing_pixels = int(np.clip(round(target * score.size), 1, score.size - 1))
        selected = np.argpartition(score.ravel(), -missing_pixels)[-missing_pixels:]
        missing_region = np.zeros(score.size, dtype=bool)
        missing_region[selected] = True
        return self._signed_observed_score(missing_region.reshape(score.shape))
