import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_reconstruction.datasets.socofing import (
    SocofingValidationError,
    build_socofing_manifest,
    parse_socofing_filename,
    write_socofing_artifacts,
)


FINGERS = ("thumb", "index", "middle", "ring", "little")


def _write_fixture(root: Path, subjects: int = 6) -> Path:
    real = root / "SOCOFing" / "Real"
    real.mkdir(parents=True)
    for subject in range(1, subjects + 1):
        gender = "M" if subject % 2 else "F"
        for hand_index, hand in enumerate(("Left", "Right")):
            for finger_index, finger in enumerate(FINGERS):
                base = subject * 10 + hand_index * 5 + finger_index
                yy, xx = np.mgrid[:24, :20]
                array = ((3 * yy + 5 * xx + base) % 256).astype(np.uint8)
                Image.fromarray(array).save(
                    real / f"{subject}__{gender}_{hand}_{finger}_finger.BMP"
                )
    return real


def test_filename_parser_normalizes_identity():
    parsed = parse_socofing_filename("17__f_right_RING_finger.bmp")
    assert parsed.subject_id == "000017"
    assert parsed.gender == "F"
    assert parsed.hand == "Right"
    assert parsed.finger == "ring"
    assert parsed.finger_id == "right_ring"
    assert parsed.is_original


def test_filename_parser_rejects_derived_suffix():
    with pytest.raises(SocofingValidationError, match="derived/altered"):
        parse_socofing_filename("1__M_Left_index_finger_CR.BMP")


def test_manifest_audits_images_and_prevents_subject_leakage(tmp_path):
    _write_fixture(tmp_path, subjects=6)
    manifest = build_socofing_manifest(
        tmp_path,
        split_fractions=(0.5, 0.25, 0.25),
        split_seed=41,
        expected_images=60,
        expected_subjects=6,
        expected_fingers_per_subject=10,
    )
    assert manifest.audit.num_images == 60
    assert manifest.audit.num_subjects == 6
    assert manifest.audit.num_unique_fingers == 60
    assert manifest.audit.image_shapes == {"24x20": 60}
    assert len(manifest.audit.dataset_sha256) == 64

    split_by_subject = {}
    for record in manifest.records:
        split_by_subject.setdefault(record.subject_id, set()).add(record.split)
        assert record.grayscale_max >= record.grayscale_min
        assert len(record.sha256) == 64
    assert all(len(splits) == 1 for splits in split_by_subject.values())


def test_manifest_detects_missing_fingers_in_strict_mode(tmp_path):
    real = _write_fixture(tmp_path, subjects=3)
    (real / "1__M_Left_thumb_finger.BMP").unlink()
    with pytest.raises(SocofingValidationError, match="expected 30 images"):
        build_socofing_manifest(
            tmp_path,
            expected_images=30,
            expected_subjects=3,
            expected_fingers_per_subject=10,
        )


def test_incomplete_dataset_can_be_audited_with_warnings(tmp_path):
    _write_fixture(tmp_path, subjects=4)
    manifest = build_socofing_manifest(
        tmp_path,
        expected_images=6000,
        expected_subjects=600,
        strict_expected_counts=False,
    )
    assert len(manifest.audit.warnings) == 2


def test_shape_and_mode_heterogeneity_are_recorded_as_warnings(tmp_path):
    real = _write_fixture(tmp_path, subjects=4)
    Image.fromarray(np.zeros((30, 22, 3), dtype=np.uint8)).save(
        real / "1__M_Left_thumb_finger.BMP"
    )
    manifest = build_socofing_manifest(
        tmp_path,
        expected_images=40,
        expected_subjects=4,
        strict_expected_counts=True,
    )
    assert any("multiple decoded image shapes" in item for item in manifest.audit.warnings)
    assert any("multiple decoded source modes" in item for item in manifest.audit.warnings)


def test_artifacts_round_trip(tmp_path):
    _write_fixture(tmp_path, subjects=6)
    manifest = build_socofing_manifest(
        tmp_path,
        split_fractions=(0.5, 0.25, 0.25),
        expected_images=60,
        expected_subjects=6,
    )
    paths = write_socofing_artifacts(manifest, tmp_path / "processed")
    with paths["csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with paths["audit"].open(encoding="utf-8") as stream:
        audit = json.load(stream)
    assert len(rows) == 60
    assert audit["dataset_sha256"] == manifest.audit.dataset_sha256
    assert paths["jsonl"].read_text(encoding="utf-8").count("\n") == 60


def test_corrupt_bmp_is_rejected(tmp_path):
    real = tmp_path / "Real"
    real.mkdir()
    (real / "1__M_Left_thumb_finger.BMP").write_bytes(b"not-a-bitmap")
    with pytest.raises(SocofingValidationError, match="cannot decode"):
        build_socofing_manifest(
            real,
            expected_images=None,
            expected_subjects=None,
            expected_fingers_per_subject=None,
        )
