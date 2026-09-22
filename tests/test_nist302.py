import csv
from pathlib import Path

import pytest
from PIL import Image

from fingerprint_reconstruction.datasets.nist302 import (
    Nist302ValidationError,
    build_nist302_manifest,
    load_finger_positions,
    parse_latent_png_filename,
)


def test_parse_real_sd302_latent_name() -> None:
    parsed = parse_latent_png_filename(
        "00002551_4E_X_243_IN_D800_1112PPI_16BPC_1CH_LP01_1.png"
    )
    assert parsed.subject_id == "00002551"
    assert parsed.ppi == 1112
    assert parsed.bits_per_channel == 16
    assert parsed.latent_number == 1
    assert parsed.source_code == 1


def test_rejects_non_contract_filename() -> None:
    with pytest.raises(Nist302ValidationError):
        parse_latent_png_filename("latent.png")


def test_multiple_lffs_impressions_remain_explicit(tmp_path: Path) -> None:
    positions = tmp_path / "finger_positions.csv"
    with positions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "fgp"])
        writer.writerow(["00002306_1H_R_L01_BP_S22_1000PPI_8BPC_1CH_LP14-1_1.lffs", "3"])
        writer.writerow(["00002306_1H_R_L01_BP_S22_1000PPI_8BPC_1CH_LP14-2_1.lffs", "3"])
    grouped = load_finger_positions(positions)
    assert len(grouped) == 1
    assert [entry.impression for entry in next(iter(grouped.values()))] == [1, 2]


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (24, 32), 128).save(path)


def test_manifest_is_subject_disjoint_and_not_pixel_ground_truth(tmp_path: Path) -> None:
    sd302e = tmp_path / "e"
    positions = tmp_path / "finger_positions.csv"
    exemplar_roots = {name: tmp_path / name for name in ("a", "b", "d")}
    for root in exemplar_roots.values():
        root.mkdir()

    rows = []
    for index, subject in enumerate(("00000001", "00000002", "00000003"), start=1):
        filename = f"{subject}_1A_R_L01_BP_S04_1200PPI_8BPC_1CH_LP01_1.png"
        _write_png(sd302e / "latent/png/original/masked/full_resolution" / subject / filename)
        rows.append([f"{subject}_1A_R_L01_BP_S04_1000PPI_8BPC_1CH_LP01-1_1.lffs", str(index)])
        _write_png(exemplar_roots["a"] / f"{subject}_A_roll_{index:02d}.png")

    with positions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "fgp"])
        writer.writerows(rows)

    manifest = build_nist302_manifest(
        sd302e_root=sd302e,
        finger_positions_csv=positions,
        exemplar_roots=exemplar_roots,
        fractions=(1 / 3, 1 / 3, 1 / 3),
    )
    assert len(manifest.records) == 3
    assert {record.split for record in manifest.records} == {"train", "validation", "test"}
    assert all(not record.pixel_aligned_ground_truth for record in manifest.records)
    assert all(len(record.exemplar_paths) == 1 for record in manifest.records)
