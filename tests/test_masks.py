import numpy as np
import pytest

from fingerprint_reconstruction.preprocessing.masks import (
    MaskFamily,
    MaskGenerator,
    MaskSpec,
)


FAMILIES = list(MaskFamily)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("ratio", [0.10, 0.30, 0.50, 0.80])
def test_exact_observed_area(family, ratio):
    if family == MaskFamily.SEVERE_PARTIAL and ratio > 0.30:
        pytest.skip("outside the declared severe-partial regime")
    generator = MaskGenerator((64, 80))
    spec = MaskSpec(
        family=family,
        observed_fraction=ratio,
        seed=13,
        num_fragments=4 if family == MaskFamily.DISCONNECTED_FRAGMENTS else None,
    )
    result = generator.generate(spec)
    expected = round(ratio * 64 * 80)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 1})
    assert int(result.mask.sum()) == expected
    assert result.metadata["actual_observed_pixels"] == expected
    assert result.metadata["semantics"] == "1=observed,0=missing"


@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic(family):
    ratio = 0.2 if family == MaskFamily.SEVERE_PARTIAL else 0.4
    fragments = 3 if family == MaskFamily.DISCONNECTED_FRAGMENTS else None
    generator = MaskGenerator((48, 48))
    spec = MaskSpec(family, ratio, seed=1234, num_fragments=fragments)
    first = generator.generate(spec).mask
    second = generator.generate(spec).mask
    assert np.array_equal(first, second)


def test_different_seeds_change_stochastic_mask():
    generator = MaskGenerator((64, 64))
    first = generator.generate(
        MaskSpec(MaskFamily.IRREGULAR, 0.5, seed=1)
    ).mask
    second = generator.generate(
        MaskSpec(MaskFamily.IRREGULAR, 0.5, seed=2)
    ).mask
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "family",
    [MaskFamily.RANDOM_RECTANGLES, MaskFamily.IRREGULAR, MaskFamily.STRIPES],
)
def test_geometric_masks_do_not_degenerate_into_salt_and_pepper(family):
    mask = MaskGenerator((128, 128)).generate(
        MaskSpec(family, 0.30, seed=91)
    ).mask.astype(np.float32)
    horizontal = np.abs(np.diff(mask, axis=1)).mean()
    vertical = np.abs(np.diff(mask, axis=0)).mean()
    assert horizontal + vertical < 0.25


@pytest.mark.parametrize("ratio", [0.10, 0.20, 0.80])
def test_stripe_masks_remain_defined_across_seeds(ratio):
    generator = MaskGenerator((128, 128))
    target = round(ratio * 128 * 128)
    for seed in range(20):
        result = generator.generate(MaskSpec(MaskFamily.STRIPES, ratio, seed=seed))
        assert int(result.mask.sum()) == target


def _mean_radius(mask):
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    cy, cx = (np.asarray(mask.shape) - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx)
    return float(radius[mask.astype(bool)].mean())


def test_central_information_is_more_central_than_peripheral_information():
    generator = MaskGenerator((96, 96))
    central = generator.generate(
        MaskSpec(MaskFamily.CENTRAL_ONLY, 0.3, seed=7)
    ).mask
    peripheral = generator.generate(
        MaskSpec(MaskFamily.PERIPHERAL_ONLY, 0.3, seed=7)
    ).mask
    assert _mean_radius(central) < _mean_radius(peripheral)


def test_central_missing_and_peripheral_fragments_are_not_duplicate_families():
    generator = MaskGenerator((96, 96))
    central_missing = generator.generate(
        MaskSpec(MaskFamily.CENTRAL_MISSING, 0.3, seed=7)
    ).mask
    peripheral = generator.generate(
        MaskSpec(MaskFamily.PERIPHERAL_ONLY, 0.3, seed=7)
    ).mask
    assert not np.array_equal(central_missing, peripheral)


def test_severe_partial_rejects_non_severe_ratio():
    with pytest.raises(ValueError, match="observed_fraction <= 0.30"):
        MaskGenerator((32, 32)).generate(
            MaskSpec(MaskFamily.SEVERE_PARTIAL, 0.5, seed=0)
        )


def test_disconnected_fragments_requires_count():
    with pytest.raises(ValueError, match="num_fragments"):
        MaskGenerator((32, 32)).generate(
            MaskSpec(MaskFamily.DISCONNECTED_FRAGMENTS, 0.2, seed=0)
        )


@pytest.mark.parametrize(
    "family",
    [MaskFamily.CENTRAL_ONLY, MaskFamily.IRREGULAR, MaskFamily.STRIPES],
)
def test_nested_masks_hold_geometry_fixed_and_are_exact(family):
    fractions = (0.1, 0.2, 0.4, 0.8)
    results = MaskGenerator((64, 80)).generate_nested(
        family=family, observed_fractions=fractions, seed=19
    )
    previous = np.zeros((64, 80), dtype=bool)
    for fraction in fractions:
        current = results[fraction].mask.astype(bool)
        assert int(current.sum()) == round(fraction * 64 * 80)
        assert np.all(previous <= current)
        assert results[fraction].metadata["nested"] is True
        previous = current


@pytest.mark.parametrize(
    "family", [MaskFamily.CENTRAL_ONLY, MaskFamily.PERIPHERAL_ONLY]
)
def test_nested_roi_masks_have_equal_roi_information_and_no_outside_pixels(family):
    yy, xx = np.mgrid[:64, :80]
    roi = ((yy - 31.5) / 27.0) ** 2 + ((xx - 39.5) / 31.0) ** 2 <= 1.0
    fractions = (0.1, 0.3, 0.8)
    results = MaskGenerator((64, 80)).generate_nested_in_roi(
        family=family,
        observed_fractions=fractions,
        seed=23,
        roi=roi,
    )
    previous = np.zeros((64, 80), dtype=bool)
    for fraction in fractions:
        current = results[fraction].mask.astype(bool)
        assert int(current[roi].sum()) == round(fraction * int(roi.sum()))
        assert not current[~roi].any()
        assert np.all(previous <= current)
        assert results[fraction].metadata["fraction_domain"] == "fingerprint_roi"
        previous = current
