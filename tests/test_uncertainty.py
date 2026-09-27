import numpy as np

from fingerprint_reconstruction.metrics.uncertainty import (
    empirical_interval_coverage,
    interval_score,
    pairwise_diversity_mae,
    sample_based_crps,
    sparsification_error,
    uncertainty_error_spearman,
)


def test_pairwise_diversity_matches_two_sample_mae():
    samples = np.stack((np.zeros((3, 3)), np.ones((3, 3))))
    assert pairwise_diversity_mae(samples, np.ones((3, 3), bool)) == 1.0


def test_empirical_coverage_and_width_are_explicit():
    samples = np.stack(
        (np.zeros((2, 2)), np.full((2, 2), 0.5), np.ones((2, 2)))
    )
    reference = np.full((2, 2), 0.5)
    coverage, width = empirical_interval_coverage(
        samples, reference, np.ones((2, 2), bool), nominal_coverage=0.8
    )
    assert coverage == 1.0
    assert np.isclose(width, 0.8)


def test_uncertainty_error_correlation_detects_monotone_pattern():
    # Across pixels, both sample SD and mean error increase monotonically.
    samples = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.2, 0.4]],
            [[0.0, 0.4, 0.8]],
        ]
    )
    reference = np.zeros((1, 3))
    correlation = uncertainty_error_spearman(samples, reference, np.ones((1, 3), bool))
    assert np.isclose(correlation, 1.0)


def test_crps_of_a_point_mass_ensemble_equals_absolute_error():
    # All samples identical -> zero ensemble spread -> CRPS degenerates to |v-y|.
    samples = np.full((5, 2, 2), 0.7)
    reference = np.full((2, 2), 0.3)
    crps = sample_based_crps(samples, reference, np.ones((2, 2), bool))
    assert np.isclose(crps, 0.4)


def test_crps_is_nonnegative_for_a_spread_ensemble():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.5, scale=0.1, size=(20, 4, 4))
    reference = np.full((4, 4), 0.5)
    crps = sample_based_crps(samples, reference, np.ones((4, 4), bool))
    assert crps >= 0.0


def test_interval_score_with_target_inside_interval_equals_width():
    # Point-mass-per-quantile ensemble: known [lower, upper] with the target
    # strictly inside -> no coverage penalty, score is exactly the width.
    samples = np.asarray([[[0.0]], [[0.5]], [[1.0]]])  # shape [K=3,H=1,W=1]
    reference = np.asarray([[0.5]])
    score = interval_score(samples, reference, np.ones((1, 1), bool), nominal_coverage=0.5)
    # tail=0.25 -> quantiles at 0.25/0.75 of [0,0.5,1.0] via linear interpolation.
    lower, upper = np.quantile(samples[:, 0, 0], [0.25, 0.75])
    assert np.isclose(score, upper - lower)


def test_interval_score_penalizes_target_outside_interval():
    samples = np.asarray([[[0.4]], [[0.5]], [[0.6]]])
    reference_inside = np.asarray([[0.5]])
    reference_outside = np.asarray([[10.0]])
    score_inside = interval_score(samples, reference_inside, np.ones((1, 1), bool), nominal_coverage=0.5)
    score_outside = interval_score(samples, reference_outside, np.ones((1, 1), bool), nominal_coverage=0.5)
    assert score_outside > score_inside


def test_sparsification_error_is_zero_for_a_perfect_ranking():
    error = np.asarray([0.1, 0.5, 0.2, 0.9, 0.05])
    # Using the true error itself as the predicted uncertainty guarantees an
    # identical sort order, so predicted and oracle curves coincide exactly.
    assert sparsification_error(error, error) == 0.0


def test_sparsification_error_is_positive_for_an_inverted_ranking():
    error = np.asarray([0.1, 0.5, 0.2, 0.9, 0.05, 0.7])
    inverted_std = -error  # highest true error gets the lowest predicted std
    assert sparsification_error(inverted_std, error) > 0.0
