import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_reconstruction.data.nist302_patch_dataset import Sd302SyntheticPatchDataset


def _write_grating(path: Path, size: int) -> None:
    yy, xx = np.mgrid[:size, :size]
    image = (127 + 110 * np.cos(2.0 * np.pi * yy / 8.0)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _write_fixture(root: Path, *, native_size: int) -> tuple[Path, Path]:
    exemplar_root = root / "sd302a"
    _write_grating(exemplar_root / "challengers" / "A" / "roll" / "00000001_A_roll_01.png", native_size)

    exemplar_manifest = root / "exemplars.csv"
    with exemplar_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset_part", "relative_path", "subject_id"])
        writer.writeheader()
        writer.writerow({"dataset_part": "sd302a", "relative_path": "challengers/A/roll/00000001_A_roll_01.png", "subject_id": "00000001"})

    registered_manifest = root / "registered_manifest.csv"
    with registered_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "split"])
        writer.writeheader()
        writer.writerow({"subject_id": "00000001", "split": "train"})

    return exemplar_manifest, registered_manifest


def test_patch_dataset_crops_without_resizing_when_native_is_large_enough(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path, native_size=400)
    dataset = Sd302SyntheticPatchDataset(
        exemplar_manifest_path=exemplar_manifest, registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"}, split="train", patch_size=64,
        minimum_foreground_fraction=0.0,
    )
    item = dataset[0]
    assert item["target"].shape == (1, 64, 64)
    assert item["observed"].shape == (1, 64, 64)


def test_patch_dataset_falls_back_to_resize_when_native_is_smaller_than_patch(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path, native_size=32)
    dataset = Sd302SyntheticPatchDataset(
        exemplar_manifest_path=exemplar_manifest, registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"}, split="train", patch_size=64,
        minimum_foreground_fraction=0.0,
    )
    item = dataset[0]
    assert item["target"].shape == (1, 64, 64)


def test_patch_dataset_is_deterministic(tmp_path: Path) -> None:
    exemplar_manifest, registered_manifest = _write_fixture(tmp_path, native_size=400)
    kwargs = dict(
        exemplar_manifest_path=exemplar_manifest, registered_manifest_path=registered_manifest,
        exemplar_roots={"sd302a": tmp_path / "sd302a"}, split="train", patch_size=64,
    )
    first = Sd302SyntheticPatchDataset(**kwargs)[0]
    second = Sd302SyntheticPatchDataset(**kwargs)[0]
    np.testing.assert_allclose(first["target"].numpy(), second["target"].numpy())
