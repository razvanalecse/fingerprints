"""Quality-aware PyTorch dataset joining SD302e images with SD302h EFS data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fingerprint_reconstruction.datasets.nist302 import parse_lffs_finger_position_filename
from fingerprint_reconstruction.preprocessing.normalize import load_grayscale


ASSESSMENT_CODES = {"NONPRINT": 0, "NOVALUE": 1, "LIMITED": 2, "VALUE": 3}


class Nist302DatasetError(ValueError):
    """Raised when prepared SD302e/SD302h artifacts cannot be joined safely."""


@dataclass(frozen=True)
class Nist302AnnotatedRow:
    sample_id: str
    png_sample_id: str
    subject_id: str
    split: str
    image_relative_path: str
    lffs_relative_path: str
    lffs_filename: str
    impression: int
    fgp: Optional[int]
    fgp_status: str
    source_code: int
    native_ppi: float
    native_width: int
    native_height: int
    assessment: str
    quality_map_relative_path: str
    quality_grid_size_0_01mm: int
    roi_horizontal_offset_0_01mm: int
    roi_vertical_offset_0_01mm: int
    quality_format_recovered: bool
    minutiae_count: int
    errata_mentioned: bool


def rasterize_ridge_quality(
    quality: np.ndarray,
    *,
    image_shape: Tuple[int, int],
    ppi: float,
    grid_size_0_01mm: int,
    horizontal_offset_0_01mm: int = 0,
    vertical_offset_0_01mm: int = 0,
) -> np.ndarray:
    """Project an EFS 9.308 grid onto the associated image pixel lattice.

    EFS coordinates use 0.01 mm units with an upper-left origin. Pixels not
    covered by the finite quality grid are background (code 0), rather than a
    stretched copy of the last grid cell.
    """

    array = np.asarray(quality)
    if array.ndim != 2 or array.size == 0:
        raise Nist302DatasetError("quality map must be a non-empty 2D array")
    if not np.issubdtype(array.dtype, np.integer):
        raise Nist302DatasetError("quality map must contain integer EFS codes")
    if int(array.min()) < 0 or int(array.max()) > 5:
        raise Nist302DatasetError("quality map codes must lie in [0, 5]")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0 or ppi <= 0 or grid_size_0_01mm <= 0:
        raise Nist302DatasetError("image shape, ppi, and grid size must be positive")
    if horizontal_offset_0_01mm < 0 or vertical_offset_0_01mm < 0:
        raise Nist302DatasetError("ROI offsets cannot be negative")

    units_per_pixel = 2540.0 / float(ppi)
    row_indices = np.floor(
        (
            np.arange(height, dtype=np.float64) * units_per_pixel
            - vertical_offset_0_01mm
        )
        / grid_size_0_01mm
    ).astype(np.int64)
    col_indices = np.floor(
        (
            np.arange(width, dtype=np.float64) * units_per_pixel
            - horizontal_offset_0_01mm
        )
        / grid_size_0_01mm
    ).astype(np.int64)
    valid_rows = (row_indices >= 0) & (row_indices < array.shape[0])
    valid_cols = (col_indices >= 0) & (col_indices < array.shape[1])
    result = np.zeros((height, width), dtype=np.uint8)
    if valid_rows.any() and valid_cols.any():
        result[np.ix_(valid_rows, valid_cols)] = array[
            np.ix_(row_indices[valid_rows], col_indices[valid_cols])
        ].astype(np.uint8, copy=False)
    return result


def resize_label_map(label_map: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize categorical annotations with nearest-neighbor interpolation."""

    height, width = (int(output_shape[0]), int(output_shape[1]))
    if height <= 0 or width <= 0:
        raise Nist302DatasetError("output shape must be positive")
    image = Image.fromarray(np.asarray(label_map, dtype=np.uint8))
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def build_annotated_rows(
    *,
    manifest_path: Path,
    annotation_csv: Path,
    split: str,
    assessments: Sequence[str],
    source_codes: Sequence[int],
    exclude_errata: bool,
    finger_positions_csv: Optional[Path] = None,
    strict: bool = True,
) -> Tuple[Nist302AnnotatedRow, ...]:
    """Join each LFFS impression to exactly one SD302e PNG manifest row."""

    if split not in {"train", "validation", "test"}:
        raise Nist302DatasetError("split must be train, validation, or test")
    allowed_assessments = {value.upper() for value in assessments}
    unknown = allowed_assessments - set(ASSESSMENT_CODES)
    if unknown:
        raise Nist302DatasetError(f"unknown assessments: {sorted(unknown)}")
    allowed_sources = {int(value) for value in source_codes}
    if not allowed_sources or not allowed_sources <= {1, 2, 3, 4}:
        raise Nist302DatasetError("source_codes must be a non-empty subset of {1,2,3,4}")

    manifest_rows = _read_csv(manifest_path)
    lffs_fgp: Dict[str, Optional[int]] = {}
    if finger_positions_csv is not None:
        for row in _read_csv(finger_positions_csv):
            filename = row.get("filename", "")
            raw_fgp = row.get("fgp", "").strip().upper()
            if not filename or filename in lffs_fgp:
                raise Nist302DatasetError(
                    f"invalid or duplicate filename in finger positions: {filename!r}"
                )
            if raw_fgp == "NA":
                lffs_fgp[filename] = None
            else:
                fgp = int(raw_fgp)
                if not 1 <= fgp <= 10:
                    raise Nist302DatasetError(f"invalid distal FGP {fgp} for {filename}")
                lffs_fgp[filename] = fgp
    by_lffs: Dict[str, Mapping[str, str]] = {}
    for row in manifest_rows:
        try:
            filenames = json.loads(row["lffs_filenames"])
        except (KeyError, json.JSONDecodeError) as error:
            raise Nist302DatasetError("invalid lffs_filenames in manifest") from error
        for filename in filenames:
            if filename in by_lffs:
                raise Nist302DatasetError(f"LFFS filename maps to multiple PNGs: {filename}")
            by_lffs[filename] = row

    joined = []
    unmatched = []
    for annotation in _read_csv(annotation_csv):
        relative_lffs = annotation["relative_path"]
        filename = Path(relative_lffs).name
        manifest = by_lffs.get(filename)
        if manifest is None:
            unmatched.append(filename)
            continue
        assessment = annotation["assessment"].upper()
        source_code = int(manifest["source_code"])
        if manifest["split"] != split:
            continue
        if assessment not in allowed_assessments or source_code not in allowed_sources:
            continue
        errata = _parse_bool(manifest["errata_mentioned"])
        if exclude_errata and errata:
            continue
        quality_path = annotation.get("quality_map_path", "")
        if not quality_path:
            if strict and annotation.get("quality_encoding", "").upper() == "UNC":
                raise Nist302DatasetError(
                    f"UNC annotation has no prepared quality map: {relative_lffs}"
                )
            continue
        _, impression = parse_lffs_finger_position_filename(filename)
        if finger_positions_csv is not None:
            if filename not in lffs_fgp:
                raise Nist302DatasetError(
                    f"LFFS filename missing from finger positions: {filename}"
                )
            fgp = lffs_fgp[filename]
            fgp_status = "lffs_known" if fgp is not None else "lffs_unknown"
        else:
            raw_fgp = manifest.get("fgp", "").strip()
            fgp = int(raw_fgp) if raw_fgp else None
            fgp_status = manifest.get("fgp_status", "manifest_unknown")
        joined.append(
            Nist302AnnotatedRow(
                sample_id=f"{manifest['sample_id']}:lffs:{Path(filename).stem}",
                png_sample_id=manifest["sample_id"],
                subject_id=manifest["subject_id"],
                split=manifest["split"],
                image_relative_path=manifest["original_masked_path"],
                lffs_relative_path=relative_lffs,
                lffs_filename=filename,
                impression=impression,
                fgp=fgp,
                fgp_status=fgp_status,
                source_code=source_code,
                native_ppi=float(manifest["native_ppi"]),
                native_width=int(manifest["width"]),
                native_height=int(manifest["height"]),
                assessment=assessment,
                quality_map_relative_path=quality_path,
                quality_grid_size_0_01mm=int(annotation["quality_grid_size_0_01mm"]),
                roi_horizontal_offset_0_01mm=int(
                    annotation["roi_horizontal_offset_0_01mm"]
                ),
                roi_vertical_offset_0_01mm=int(
                    annotation["roi_vertical_offset_0_01mm"]
                ),
                quality_format_recovered=_parse_bool(annotation["quality_format_recovered"]),
                minutiae_count=int(annotation["minutiae_count"]),
                errata_mentioned=errata,
            )
        )
    if strict and unmatched:
        raise Nist302DatasetError(
            f"{len(unmatched)} annotation records do not match the PNG manifest; "
            f"first: {unmatched[:3]}"
        )
    joined.sort(key=lambda row: (row.subject_id, row.lffs_filename))
    sample_ids = [row.sample_id for row in joined]
    if len(sample_ids) != len(set(sample_ids)):
        raise Nist302DatasetError("joined LFFS sample identifiers are not unique")
    if not joined:
        raise Nist302DatasetError("no rows remain after SD302 filters")
    return tuple(joined)


class Nist302AnnotatedDataset(Dataset):
    """Load aligned SD302e pixels and prepared SD302h quality annotations."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        annotation_csv: Path,
        image_root: Path,
        annotation_root: Path,
        split: str,
        output_shape: Tuple[int, int] = (256, 256),
        assessments: Sequence[str] = ("VALUE", "LIMITED"),
        source_codes: Sequence[int] = (1,),
        exclude_errata: bool = True,
        finger_positions_csv: Optional[Path] = None,
        include_model_conditioning: bool = True,
        missing_fill_value: float = 0.0,
        strict: bool = True,
    ) -> None:
        self.rows = build_annotated_rows(
            manifest_path=manifest_path,
            annotation_csv=annotation_csv,
            split=split,
            assessments=assessments,
            source_codes=source_codes,
            exclude_errata=exclude_errata,
            finger_positions_csv=finger_positions_csv,
            strict=strict,
        )
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.include_model_conditioning = bool(include_model_conditioning)
        self.missing_fill_value = float(missing_fill_value)
        if min(self.output_shape) <= 0:
            raise Nist302DatasetError("output_shape dimensions must be positive")
        if not 0.0 <= self.missing_fill_value <= 1.0:
            raise Nist302DatasetError("missing_fill_value must lie in [0, 1]")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        row = self.rows[index]
        image_path = self.image_root / row.image_relative_path
        quality_path = self.annotation_root / row.quality_map_relative_path
        if not image_path.is_file():
            raise Nist302DatasetError(f"image not found: {image_path}")
        if not quality_path.is_file():
            raise Nist302DatasetError(f"quality map not found: {quality_path}")

        with np.load(quality_path, allow_pickle=False) as archive:
            quality_grid = np.asarray(archive["quality"], dtype=np.uint8)
            stored_grid_size = int(archive["grid_size_0_01mm"])
        if stored_grid_size != row.quality_grid_size_0_01mm:
            raise Nist302DatasetError(
                f"grid size mismatch for {row.lffs_filename}: "
                f"CSV={row.quality_grid_size_0_01mm}, NPZ={stored_grid_size}"
            )
        quality_native = rasterize_ridge_quality(
            quality_grid,
            image_shape=(row.native_height, row.native_width),
            ppi=row.native_ppi,
            grid_size_0_01mm=stored_grid_size,
            horizontal_offset_0_01mm=row.roi_horizontal_offset_0_01mm,
            vertical_offset_0_01mm=row.roi_vertical_offset_0_01mm,
        )
        quality = resize_label_map(quality_native, self.output_shape)
        image = load_grayscale(image_path, output_shape=self.output_shape)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        quality_tensor = torch.from_numpy(quality.astype(np.int64, copy=False)).unsqueeze(0)
        foreground = (quality_tensor >= 1).to(torch.float32)
        reliable_ridge = (quality_tensor >= 2).to(torch.float32)
        reliable_minutiae = (quality_tensor >= 3).to(torch.float32)
        result = {
            "image": image_tensor,
            "quality": quality_tensor,
            "foreground": foreground,
            "reliable_ridge": reliable_ridge,
            "reliable_minutiae": reliable_minutiae,
            "sample_id": row.sample_id,
            "png_sample_id": row.png_sample_id,
            "subject_id": row.subject_id,
            "assessment": row.assessment,
            "assessment_code": torch.tensor(ASSESSMENT_CODES[row.assessment], dtype=torch.int64),
            "source_code": torch.tensor(row.source_code, dtype=torch.int64),
            "fgp": torch.tensor(row.fgp or 0, dtype=torch.int64),
            "fgp_known": row.fgp is not None,
            "fgp_status": row.fgp_status,
            "native_ppi": torch.tensor(row.native_ppi, dtype=torch.float32),
            "minutiae_count": torch.tensor(row.minutiae_count, dtype=torch.int64),
            "quality_format_recovered": row.quality_format_recovered,
            "pixel_aligned_exemplar_ground_truth": False,
        }
        if self.include_model_conditioning:
            mask = reliable_ridge
            observed = torch.where(
                mask.bool(), image_tensor, torch.full_like(image_tensor, self.missing_fill_value)
            )
            foreground_pixels = foreground.sum().clamp_min(1.0)
            result.update(
                {
                    "conditioning": torch.cat((observed, mask), dim=0),
                    "observed": observed,
                    "mask": mask,
                    "observed_fraction_canvas": mask.mean(),
                    "observed_fraction_annotated_roi": (mask * foreground).sum()
                    / foreground_pixels,
                    "conditioning_semantics": "M=(official EFS quality>=2); no pixel target",
                }
            )
        return result


def _read_csv(path: Path) -> List[Mapping[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise Nist302DatasetError(f"invalid Boolean value: {value!r}")
