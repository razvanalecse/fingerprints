"""Sample-based uncertainty and diversity metrics for probabilistic reconstructions."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _roi_samples(samples: np.ndarray, region: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if array.ndim != 3 or array.shape[1:] != mask.shape:
        raise ValueError("samples must have shape [K,H,W] matching region [H,W]")
    if array.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if not mask.any():
        raise ValueError("region cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError("samples contain NaN or infinity")
    return array[:, mask]


def pairwise_diversity_mae(samples: np.ndarray, region: np.ndarray) -> float:
    r"""Mean pairwise L1 distance: 2/[K(K-1)] sum_{i<j} d(x_i,x_j)."""

    selected = _roi_samples(samples, region)
    differences = []
    for left in range(selected.shape[0]):
        for right in range(left + 1, selected.shape[0]):
            differences.append(np.mean(np.abs(selected[left] - selected[right])))
    return float(np.mean(differences))


def uncertainty_error_spearman(
    samples: np.ndarray, reference: np.ndarray, region: np.ndarray
) -> float:
    """Spearman association between predictive SD and predictive-mean error."""

    selected = _roi_samples(samples, region)
    truth = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("reference and region must have equal shapes")
    uncertainty = selected.std(axis=0, ddof=1)
    error = np.abs(selected.mean(axis=0) - truth[mask])
    if np.all(uncertainty == uncertainty[0]) or np.all(error == error[0]):
        return float("nan")
    return float(spearmanr(uncertainty, error).statistic)


def empirical_interval_coverage(
    samples: np.ndarray,
    reference: np.ndarray,
    region: np.ndarray,
    *,
    nominal_coverage: float,
) -> tuple[float, float]:
    """Return empirical central-interval coverage and mean interval width."""

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie strictly in (0,1)")
    selected = _roi_samples(samples, region)
    truth = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("reference and region must have equal shapes")
    tail = (1.0 - nominal_coverage) / 2.0
    lower, upper = np.quantile(selected, [tail, 1.0 - tail], axis=0)
    values = truth[mask]
    coverage = np.mean((values >= lower) & (values <= upper))
    return float(coverage), float(np.mean(upper - lower))
