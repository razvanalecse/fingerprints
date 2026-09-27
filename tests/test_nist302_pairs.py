import csv
from pathlib import Path

import pytest

from fingerprint_reconstruction.data.nist302_torch_dataset import Nist302AnnotatedRow
from fingerprint_reconstruction.datasets.nist302_pairs import (
    Nist302ExemplarRow,
    Nist302PairError,
    build_latent_exemplar_pairs,
    write_pair_manifest,
)


def _latent(*, fgp=7) -> Nist302AnnotatedRow:
    return Nist302AnnotatedRow(
        sample_id="latent-1",
        png_sample_id="png-1",
        subject_id="00000001",
        split="test",
        image_relative_path="latent.png",
        lffs_relative_path="latent.lffs",
        lffs_filename="latent.lffs",
        impression=1,
        fgp=fgp,
        fgp_status="lffs_known" if fgp is not None else "lffs_unknown",
        source_code=1,
        native_ppi=1000,
        native_width=10,
        native_height=10,
        assessment="VALUE",
        quality_map_relative_path="quality.npz",
        quality_grid_size_0_01mm=20,
        roi_horizontal_offset_0_01mm=0,
        roi_vertical_offset_0_01mm=0,
        quality_format_recovered=False,
        minutiae_count=12,
        errata_mentioned=False,
    )


def test_pair_builder_keeps_all_same_finger_candidates(tmp_path: Path) -> None:
    exemplars = (
        Nist302ExemplarRow("sd302a", "a.png", "00000001", "A", None, "roll", 7),
        Nist302ExemplarRow("sd302b", "b.png", "00000001", "U", 1000, "roll", 7),
        Nist302ExemplarRow("sd302b", "wrong.png", "00000001", "U", 1000, "roll", 8),
    )
    pairs = build_latent_exemplar_pairs((_latent(),), exemplars)
    assert len(pairs) == 2
    assert {pair.exemplar_relative_path for pair in pairs} == {"a.png", "b.png"}
    assert all(pair.pixel_aligned_ground_truth is False for pair in pairs)
    destination = tmp_path / "pairs.csv"
    write_pair_manifest(pairs, destination)
    assert len(list(csv.DictReader(destination.open()))) == 2


def test_known_fgp_without_candidate_is_rejected() -> None:
    with pytest.raises(Nist302PairError, match="no exemplar candidate"):
        build_latent_exemplar_pairs((_latent(),), ())


def test_unknown_fgp_is_not_guessed() -> None:
    assert build_latent_exemplar_pairs((_latent(fgp=None),), (), require_candidate=True) == ()
