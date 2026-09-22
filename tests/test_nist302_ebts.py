from pathlib import Path

import pytest

from fingerprint_reconstruction.datasets.nist302_ebts import (
    Nist302EbtsError,
    efs_to_pixel,
    match_correspondences,
    parse_formatted_text,
)


LFFS_TEXT = "\n".join(
    (
        "1.1.1.1 [1.001]=100\x1f",
        "1.2.1.1 [1.002]=0502\x1f",
        "1.3.1.1 [1.004]=LFFS\x1f",
        "2.1.1.1 [2.001]=10\x1f",
        "3.1.1.1 [9.001]=100\x1f",
        "3.2.1.1 [9.002]=1\x1f",
        "3.3.1.1 [9.300]=2540\x1f",
        "3.3.1.2 [9.300]=1270\x1f",
        "3.3.1.3 [9.300]=100\x1f",
        "3.3.1.4 [9.300]=200\x1f",
        "3.3.1.5 [9.300]=0,0-2540,0-2540,1270-0,1270\x1f",
        "3.4.1.1 [9.308]=012\x1f",
        "3.4.2.1 [9.308]=345\x1f",
        "3.5.1.1 [9.309]=20\x1f",
        "3.5.1.2 [9.309]=UNC\x1f",
        "3.6.1.1 [9.331]=354\x1f",
        "3.6.1.2 [9.331]=454\x1f",
        "3.6.1.3 [9.331]=098\x1f",
        "3.6.1.4 [9.331]=B\x1f",
        "3.7.1.1 [9.353]=VALUE\x1f",
    )
)


def test_parse_lffs_quality_roi_and_minutia() -> None:
    transaction = parse_formatted_text(LFFS_TEXT, source_path="sample.lffs")
    assert transaction.transaction_type == "LFFS"
    assert transaction.version == "0502"
    record = transaction.efs_records[0]
    assert record.roi.horizontal_offset_0_01mm == 100
    assert record.ridge_quality is not None
    assert record.ridge_quality.shape == (2, 3)
    assert record.ridge_quality.counts() == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert record.assessment == "VALUE"
    assert record.minutiae[0].direction_degrees == 98
    assert record.minutiae[0].position_uncertainty_radius_0_01mm is None


def test_coordinate_conversion_accounts_for_roi_offset() -> None:
    x, y = efs_to_pixel(
        354,
        454,
        ppi=1000,
        horizontal_offset_0_01mm=100,
        vertical_offset_0_01mm=200,
    )
    assert x == pytest.approx((354 + 100) * 1000 / 2540)
    assert y == pytest.approx((454 + 200) * 1000 / 2540)


def test_rejects_nonrectangular_quality_map() -> None:
    with pytest.raises(Nist302EbtsError, match="non-rectangular"):
        parse_formatted_text(LFFS_TEXT.replace("[9.308]=345", "[9.308]=34"))


def test_narrow_recovery_for_malformed_release_quality_format() -> None:
    malformed = LFFS_TEXT.replace(
        "3.4.2.1 [9.308]=345\x1f\n3.5.1.1 [9.309]=20\x1f\n3.5.1.2 [9.309]=UNC\x1f",
        "3.4.2.1 [9.308]=20\x1f\n3.4.2.2 [9.308]=UNC\x1f\n3.4.2.3 [9.308]=345\x1f",
    )
    record = parse_formatted_text(malformed).efs_records[0]
    assert record.quality_format_recovered
    assert record.ridge_quality is not None
    assert record.ridge_quality.rows == ("012", "345")


def test_narrow_recovery_for_duplicated_quality_format_items() -> None:
    malformed = LFFS_TEXT.replace(
        "3.5.1.2 [9.309]=UNC\x1f",
        "3.5.1.2 [9.309]=UNC\x1f\n"
        "3.5.1.3 [9.309]=20\x1f\n"
        "3.5.1.4 [9.309]=UNC\x1f",
    )
    record = parse_formatted_text(malformed).efs_records[0]
    assert record.quality_format_recovered
    assert record.ridge_quality is not None
    assert record.ridge_quality.grid_size_0_01mm == 20


def test_parse_comp_sources_and_links() -> None:
    text = LFFS_TEXT.replace("[1.004]=LFFS", "[1.004]=COMP") + "\n" + "\n".join(
        (
            "2.8.1.1 [2.1406]=1\x1f",
            "2.8.1.2 [2.1406]=LFFS\x1f",
            "2.8.1.3 [2.1406]=1\x1f",
            "2.8.1.4 [2.1406]=latent-id\x1f",
            "2.8.1.5 [2.1406]=0\x1f",
            "2.8.1.6 [2.1406]=9,13\x1f",
            "2.8.1.7 [2.1406]=latent.lffs\x1f",
            "2.8.1.8 [2.1406]=NIST SD 302\x1f",
            "3.8.1.1 [9.361]=97\x1f",
            "3.8.1.2 [9.361]=F\x1f",
            "3.8.1.3 [9.361]=331\x1f",
            "3.8.1.4 [9.361]=1\x1f",
            "3.8.1.5 [9.361]=354\x1f",
            "3.8.1.6 [9.361]=454\x1f",
            "3.9.1.1 [9.363]=2\x1f",
            "3.9.1.2 [9.363]=-112\x1f",
            "3.10.1.1 [9.362]=2\x1f",
            "3.10.1.2 [9.362]=INDIV\x1f",
            "3.10.1.3 [9.362]=FINAL\x1f",
        )
    )
    transaction = parse_formatted_text(text)
    assert transaction.comp_sources[0].filename == "latent.lffs"
    assert transaction.comp_feature_references[3][0].label == "97"
    assert transaction.comp_feature_references[3][0].feature_index == 1
    assert transaction.examiner_comparisons[3][0].determination == "INDIV"
    assert transaction.relative_rotations[3][0].degrees == -112


def test_unrecognized_nonempty_line_is_not_silently_ignored() -> None:
    with pytest.raises(Nist302EbtsError, match="unrecognized"):
        parse_formatted_text("not an an2k2txt line")


def test_correspondences_match_by_label_not_by_list_position() -> None:
    text = LFFS_TEXT + "\n" + "\n".join(
        (
            "3.8.1.1 [9.361]=M7\x1f",
            "3.8.1.2 [9.361]=F\x1f",
            "3.8.1.3 [9.361]=331\x1f",
            "3.8.1.4 [9.361]=1\x1f",
            "3.8.1.5 [9.361]=354\x1f",
            "3.8.1.6 [9.361]=454\x1f",
            "4.1.1.1 [9.001]=100\x1f",
            "4.2.1.1 [9.002]=2\x1f",
            "4.3.1.1 [9.300]=2540\x1f",
            "4.3.1.2 [9.300]=1270\x1f",
            "4.4.1.1 [9.361]=M7\x1f",
            "4.4.1.2 [9.361]=F\x1f",
            "4.4.1.3 [9.361]=331\x1f",
            "4.4.1.4 [9.361]=9\x1f",
            "4.4.1.5 [9.361]=800\x1f",
            "4.4.1.6 [9.361]=900\x1f",
        )
    )
    transaction = parse_formatted_text(text)
    matches = match_correspondences(transaction, 3, 4)
    assert len(matches) == 1
    assert matches[0].label == "M7"
    assert matches[0].first.feature_index == 1
    assert matches[0].second.feature_index == 9
