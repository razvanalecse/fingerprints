import numpy as np

from fingerprint_reconstruction.metrics.ridge_markov import (
    marginal_entropy_bits,
    orientation_transition_counts,
    quantize_orientation,
    summarize_ridge_markov,
    transition_entropy_bits,
)


def test_quantize_orientation_is_pi_periodic():
    # Axial orientation: theta and theta + pi are the same physical ridge direction.
    theta = np.array([[0.1, 0.1 + np.pi]], dtype=np.float64)
    states = quantize_orientation(theta, bins=12)
    assert states[0, 0] == states[0, 1]


def test_quantize_orientation_keeps_states_in_range_for_the_closed_upper_edge():
    theta = np.array([[0.0, np.pi, 2 * np.pi - 1e-12]], dtype=np.float64)
    states = quantize_orientation(theta, bins=8)
    assert states.min() >= 0
    assert states.max() <= 7


def test_constant_field_has_zero_entropy_and_full_self_transition():
    theta = np.full((32, 32), 0.7, dtype=np.float64)
    region = np.ones_like(theta, dtype=bool)
    summary = summarize_ridge_markov(theta, region, bins=12, step=3)
    assert summary.transition_entropy_bits == 0.0
    assert summary.self_transition_rate == 1.0
    assert summary.marginal_entropy_bits == 0.0
    assert summary.transition_count > 0


def test_random_field_is_less_predictable_than_a_smooth_one():
    rng = np.random.default_rng(0)
    smooth = np.linspace(0.0, 0.3, 40)[None, :].repeat(40, axis=0)
    noisy = rng.uniform(0.0, np.pi, size=(40, 40))
    region = np.ones((40, 40), dtype=bool)

    smooth_summary = summarize_ridge_markov(smooth, region, bins=12, step=3)
    noisy_summary = summarize_ridge_markov(noisy, region, bins=12, step=3)

    # This is the whole point of the diagnostic: an over-smoothed field is
    # more predictable, so it carries lower conditional entropy.
    assert smooth_summary.transition_entropy_bits < noisy_summary.transition_entropy_bits
    assert smooth_summary.self_transition_rate > noisy_summary.self_transition_rate


def test_region_and_validity_masks_restrict_the_counted_transitions():
    theta = np.zeros((20, 20), dtype=np.float64)
    region = np.zeros((20, 20), dtype=bool)
    region[:10, :10] = True
    unrestricted = orientation_transition_counts(theta, np.ones_like(region), step=1).sum()
    restricted = orientation_transition_counts(theta, region, step=1).sum()
    assert restricted < unrestricted

    valid = np.ones_like(region)
    valid[:5, :] = False
    masked = orientation_transition_counts(theta, region, valid=valid, step=1).sum()
    assert masked < restricted


def test_entropy_helpers_agree_with_a_hand_computed_case():
    # Two states, each transitioning to the other half the time: H(next|cur)=1 bit.
    counts = np.array([[5, 5], [5, 5]], dtype=np.int64)
    assert transition_entropy_bits(counts) == 1.0
    assert marginal_entropy_bits(counts) == 1.0

    deterministic = np.array([[10, 0], [0, 10]], dtype=np.int64)
    assert transition_entropy_bits(deterministic) == 0.0
    assert marginal_entropy_bits(deterministic) == 1.0


def test_empty_region_reports_nan_rather_than_raising():
    theta = np.zeros((8, 8), dtype=np.float64)
    region = np.zeros((8, 8), dtype=bool)
    summary = summarize_ridge_markov(theta, region, step=1)
    assert summary.transition_count == 0
    assert np.isnan(summary.transition_entropy_bits)
    assert np.isnan(summary.self_transition_rate)
