import pytest

from fingerprint_reconstruction.analysis.socofing_eda import summarize_manifest


def test_summary_counts_unique_subjects_fingers_and_normalized_quantiles():
    rows = [
        {
            "subject_id": "s1", "finger_id": "left_thumb", "split": "train",
            "gender": "M", "hand": "Left", "height": "103", "width": "96",
            "source_mode": "RGBA", "grayscale_mean": "127.5", "grayscale_std": "25.5",
        },
        {
            "subject_id": "s1", "finger_id": "right_thumb", "split": "train",
            "gender": "M", "hand": "Right", "height": "103", "width": "96",
            "source_mode": "RGBA", "grayscale_mean": "255", "grayscale_std": "51",
        },
        {
            "subject_id": "s2", "finger_id": "left_thumb", "split": "test",
            "gender": "F", "hand": "Left", "height": "298", "width": "241",
            "source_mode": "RGB", "grayscale_mean": "0", "grayscale_std": "0",
        },
    ]
    summary = summarize_manifest(rows)
    assert summary.num_images == 3
    assert summary.num_subjects == 2
    assert summary.num_fingers == 3
    assert summary.split_subjects == {"test": 1, "train": 1}
    assert summary.intensity_mean_quantiles["q50"] == pytest.approx(0.5)
    assert summary.image_shapes == {"103x96": 2, "298x241": 1}


def test_summary_rejects_empty_rows():
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_manifest([])
