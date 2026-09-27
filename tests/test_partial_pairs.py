import numpy as np
import pytest

from fingerprint_reconstruction.data.partial_pairs import (
    build_partial_pair,
    enforce_data_consistency,
    observed_fraction_report,
    stable_condition_seed,
    stable_nested_mask_seed,
)
from fingerprint_reconstruction.preprocessing.masks import MaskFamily


def test_condition_seed_is_stable_and_condition_specific():
    kwargs = dict(
        base_seed=1729,
        sample_id="socofing:000001:left_thumb",
        family=MaskFamily.IRREGULAR,
        observed_fraction=0.3,
        replicate=0,
    )
    assert stable_condition_seed(**kwargs) == stable_condition_seed(**kwargs)
    assert stable_condition_seed(**kwargs) != stable_condition_seed(**{**kwargs, "replicate": 1})


def test_nested_mask_seed_is_fraction_independent_but_geometry_specific():
    kwargs = dict(
        base_seed=1729,
        sample_id="socofing:000001:left_thumb",
        family=MaskFamily.IRREGULAR,
        replicate=0,
    )
    seed = stable_nested_mask_seed(**kwargs)
    assert seed == stable_nested_mask_seed(**kwargs)
    assert seed != stable_nested_mask_seed(
        **{**kwargs, "family": MaskFamily.STRIPES}
    )
    assert seed != stable_nested_mask_seed(**{**kwargs, "replicate": 1})


def test_partial_pair_preserves_observed_pixels_and_exact_ratio():
    image = np.linspace(0, 1, 64 * 80, dtype=np.float32).reshape(64, 80)
    pair = build_partial_pair(
        image,
        sample_id="sample-1",
        family=MaskFamily.STRIPES,
        observed_fraction=0.3,
        replicate=2,
    )
    observed_region = pair.mask == 1
    missing_region = ~observed_region
    assert int(pair.mask.sum()) == round(0.3 * image.size)
    assert np.array_equal(pair.observed[observed_region], pair.full[observed_region])
    assert np.all(pair.observed[missing_region] == 0.0)
    assert pair.metadata["sample_id"] == "sample-1"


def test_data_consistency_replaces_only_observed_region():
    reconstruction = np.full((8, 8), 0.75, dtype=np.float32)
    observed = np.full((8, 8), 0.25, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, :3] = 1
    result = enforce_data_consistency(reconstruction, observed, mask)
    assert np.all(result[:, :3] == 0.25)
    assert np.all(result[:, 3:] == 0.75)


def test_data_consistency_rejects_nonbinary_mask():
    with pytest.raises(ValueError, match="binary"):
        enforce_data_consistency(
            np.zeros((4, 4)), np.zeros((4, 4)), np.full((4, 4), 0.5)
        )


def test_observed_fraction_report_separates_canvas_and_fingerprint_roi():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    roi = np.zeros((4, 4), dtype=bool)
    roi[:2, :] = True
    report = observed_fraction_report(mask, roi)
    assert report["observed_fraction_canvas"] == 0.25
    assert report["observed_fraction_fingerprint_roi"] == 0.5
    assert report["missing_fraction_fingerprint_roi"] == 0.5
