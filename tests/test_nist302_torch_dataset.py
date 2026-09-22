import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_reconstruction.data.nist302_torch_dataset import (
    Nist302AnnotatedDataset,
    Nist302DatasetError,
    build_annotated_rows,
    rasterize_ridge_quality,
)


def test_rasterize_quality_uses_physical_grid_without_stretching() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    result = rasterize_ridge_quality(
        grid, image_shape=(5, 5), ppi=2540, grid_size_0_01mm=2
    )
    assert np.array_equal(result[:4, :4], np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1))
    assert np.all(result[4, :] == 0)
    assert np.all(result[:, 4] == 0)


def test_rasterize_quality_respects_roi_offset() -> None:
    result = rasterize_ridge_quality(
        np.array([[4]], dtype=np.uint8),
        image_shape=(3, 3),
        ppi=2540.0,
        grid_size_0_01mm=1,
        horizontal_offset_0_01mm=1,
        vertical_offset_0_01mm=1,
    )
    expected = np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def _write_fixture(root: Path, *, quality_map_path: str = "quality_maps/map.npz") -> tuple[Path, Path]:
    manifest = root / "manifest.csv"
    annotation = root / "annotations.csv"
    image_relative = "latent/png/original/masked/full_resolution/00000001/image.png"
    image_path = root / "images" / image_relative
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.arange(25, dtype=np.uint8).reshape(5, 5) * 10).save(image_path)
    if quality_map_path:
        map_path = root / "annotations" / quality_map_path
        map_path.parent.mkdir(parents=True)
        np.savez_compressed(
            map_path,
            quality=np.array([[0, 2], [3, 5]], dtype=np.uint8),
            grid_size_0_01mm=np.uint16(2),
        )
    lffs = "00000001_1A_R_L01_BP_S04_1000PPI_8BPC_1CH_LP01-1_1.lffs"
    manifest_row = {
        "sample_id": "nist302:00000001:1A:R:L01:BP:S04:1:1:1",
        "subject_id": "00000001",
        "split": "train",
        "source_code": "1",
        "native_ppi": "2540",
        "width": "5",
        "height": "5",
        "original_masked_path": image_relative,
        "lffs_filenames": json.dumps([lffs]),
        "errata_mentioned": "False",
    }
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_row))
        writer.writeheader()
        writer.writerow(manifest_row)
    annotation_row = {
        "relative_path": f"00000001/{lffs}",
        "assessment": "VALUE",
        "quality_map_path": quality_map_path,
        "quality_encoding": "UNC",
        "quality_grid_size_0_01mm": "2",
        "roi_horizontal_offset_0_01mm": "0",
        "roi_vertical_offset_0_01mm": "0",
        "quality_format_recovered": "False",
        "minutiae_count": "7",
    }
    with annotation.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annotation_row))
        writer.writeheader()
        writer.writerow(annotation_row)
    return manifest, annotation


def _write_finger_positions(root: Path, filename: str) -> Path:
    path = root / "finger_positions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "fgp"])
        writer.writeheader()
        writer.writerow({"filename": filename, "fgp": "7"})
    return path


def test_dataset_joins_lffs_to_png_and_builds_hierarchical_masks(tmp_path: Path) -> None:
    manifest, annotation = _write_fixture(tmp_path)
    filename = json.loads(next(csv.DictReader(manifest.open()))["lffs_filenames"])[0]
    finger_positions = _write_finger_positions(tmp_path, filename)
    dataset = Nist302AnnotatedDataset(
        manifest_path=manifest,
        annotation_csv=annotation,
        image_root=tmp_path / "images",
        annotation_root=tmp_path / "annotations",
        split="train",
        output_shape=(5, 5),
        finger_positions_csv=finger_positions,
    )
    item = dataset[0]
    assert item["image"].shape == (1, 5, 5)
    assert item["quality"].shape == (1, 5, 5)
    assert item["quality"].unique().tolist() == [0, 2, 3, 5]
    assert float(item["foreground"].sum()) == 12
    assert float(item["reliable_ridge"].sum()) == 12
    assert float(item["reliable_minutiae"].sum()) == 8
    assert ":lffs:" in item["sample_id"]
    assert int(item["fgp"]) == 7
    assert item["fgp_known"] is True
    assert item["pixel_aligned_exemplar_ground_truth"] is False
    assert item["conditioning"].shape == (2, 5, 5)
    assert item["mask"].equal(item["reliable_ridge"])
    assert item["observed"][item["mask"] == 0].eq(0).all()
    assert "target" not in item


def test_missing_prepared_quality_map_is_rejected(tmp_path: Path) -> None:
    manifest, annotation = _write_fixture(tmp_path, quality_map_path="")
    with pytest.raises(Nist302DatasetError, match="no prepared quality map"):
        build_annotated_rows(
            manifest_path=manifest,
            annotation_csv=annotation,
            split="train",
            assessments=("VALUE",),
            source_codes=(1,),
            exclude_errata=True,
        )
