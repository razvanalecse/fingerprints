import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_reconstruction.data.nist302_synthetic_dataset import (
    Sd302SyntheticPartialDataset,
    load_subject_split_map,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily


def _write_grating(path: Path, size: int = 64) -> None:
    yy, xx = np.mgrid[:size, :size]
    image = (127 + 110 * np.cos(2.0 * np.pi * yy / 8.0)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    exemplar_root = root / "sd302a"
    _write_grating(exemplar_root / "challengers" / "A" / "roll" / "00000001_A_roll_01.png")
    _write_grating(exemplar_root / "challengers" / "A" / "roll" / "00000001_A_roll_02.png")
    _write_grating(exemplar_root / "challengers" / "A" / "roll" / "00000002_A_roll_01.png")

    exemplar_manifest = root / "exemplars.csv"
    with exemplar_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset_part", "relative_path", "subject_id"])
        writer.writeheader()
        writer.writerow({"dataset_part": "sd302a", "relative_path": "challengers/A/roll/00000001_A_roll_01.png", "subject_id": "00000001"})
        writer.writerow({"dataset_part": "sd302a", "relative_path": "challengers/A/roll/00000001_A_roll_02.png", "subject_id": "00000001"})
        writer.writerow({"dataset_part": "sd302a", "relative_path": "challengers/A/roll/00000002_A_roll_01.png", "subject_id": "00000002"})

    registered_manifest = root / "registered_manifest.csv"
    with registered_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "split"])
        writer.writeheader()
        writer.writerow({"subject_id": "00000001", "split": "train"})
        writer.writerow({"subject_id": "00000002", "split": "validation"})

    return exemplar_manifest, registered_manifest


def test_load_subject_split_map_reads_the_registered_manifest(tmp_path: Path) -> None:
    _, registered_manifest = _write_fixture(tmp_path)
    mapping = load_subject_split_map(registered_manifest)
    assert mapping == {"00000001": "train", "00000002": "validation"}


def test_load_subject_split_map_rejects_inconsistent_splits(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "split"])
        writer.writeheader()
        writer.writerow({"subject_id": "s1", "split": "train"})
        writer.writerow({"subject_id": "s1", "split": "validation"})
    with pytest.raises(ValueError, match="multiple splits"):
        load_subject_split_map(path)


def test_dataset_only_includes_rows_for_the_requested_split(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path)
    dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=exemplar_manifest,
        registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"},
        split="train",
        output_shape=(32, 32),
    )
    assert len(dataset) == 2  # both rows for subject 00000001
    assert all(item["subject_id"] == "00000001" for item in dataset)


def test_dataset_item_has_expected_shapes_and_exact_ground_truth_semantics(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path)
    dataset = Sd302SyntheticPartialDataset(
        exemplar_manifest_path=exemplar_manifest,
        registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"},
        split="validation",
        output_shape=(32, 32),
        families=(MaskFamily.RANDOM_RECTANGLES,),
        observed_fractions=(0.5,),
    )
    item = dataset[0]
    assert item["observed"].shape == (1, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
    assert item["target"].shape == (1, 32, 32)
    # Observed pixels must exactly equal the target wherever mask==1 (data consistency
    # by construction, since observed = mask * target).
    mask_bool = item["mask"].bool()
    assert (item["observed"][mask_bool] == item["target"][mask_bool]).all()
    assert item["subject_id"] == "00000002"


def test_dataset_is_deterministic_across_construction(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path)
    kwargs = dict(
        exemplar_manifest_path=exemplar_manifest, registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"}, split="train", output_shape=(32, 32),
    )
    first = Sd302SyntheticPartialDataset(**kwargs)[0]
    second = Sd302SyntheticPartialDataset(**kwargs)[0]
    np.testing.assert_allclose(first["target"].numpy(), second["target"].numpy())
    np.testing.assert_array_equal(first["mask"].numpy(), second["mask"].numpy())


def test_dataset_raises_for_a_split_with_no_matching_subjects(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="no exemplar rows"):
        Sd302SyntheticPartialDataset(
            exemplar_manifest_path=exemplar_manifest, registered_manifest_path=registered_manifest,
            exemplar_roots={"sd302a": tmp_path / "sd302a"}, split="test", output_shape=(32, 32),
        )
