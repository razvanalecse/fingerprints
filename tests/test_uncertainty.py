import numpy as np

from fingerprint_reconstruction.metrics.uncertainty import (
    empirical_interval_coverage,
    pairwise_diversity_mae,
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
