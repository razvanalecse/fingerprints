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
    # For sorted values x_(i), sum_{i<j}|x_j-x_i| equals
    # sum_i (2i-K+1)x_(i). This is exact and avoids O(K^2) temporary arrays,
    # which matters for K=50 over thousands of pixels.
    ordered = np.sort(selected, axis=0)
    count = ordered.shape[0]
    coefficients = (2 * np.arange(count) - count + 1).astype(np.float64)
    pair_sum_per_pixel = np.sum(coefficients[:, None] * ordered, axis=0)
    return float(np.mean(pair_sum_per_pixel) / (count * (count - 1) / 2.0))


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


def standardized_residual_scale(
    absolute_error: np.ndarray,
    predictive_std: np.ndarray,
    *,
    nominal_coverage: float,
    epsilon: float = 1e-6,
) -> float:
    """Finite-sample upper quantile of |error|/(predictive SD + epsilon).

    The returned multiplier calibrates symmetric intervals around the predictive
    mean. It repairs marginal interval width, not spatial uncertainty ranking.
    """

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie strictly in (0,1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    std = np.asarray(predictive_std, dtype=np.float64).reshape(-1)
    if error.shape != std.shape or error.size == 0:
        raise ValueError("error and predictive_std must be non-empty and equally shaped")
    valid = np.isfinite(error) & np.isfinite(std) & (error >= 0) & (std >= 0)
    if not valid.any():
        raise ValueError("no valid residuals")
    ratios = error[valid] / (std[valid] + epsilon)
    # Split-conformal finite-sample rank, capped at the largest observation.
    level = min(1.0, np.ceil((ratios.size + 1) * nominal_coverage) / ratios.size)
    return float(np.quantile(ratios, level, method="higher"))


def scaled_std_interval_coverage(
    absolute_error: np.ndarray,
    predictive_std: np.ndarray,
    *,
    scale: float,
) -> tuple[float, float]:
    """Coverage and full width of mean ± scale * predictive_std."""

    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    std = np.asarray(predictive_std, dtype=np.float64).reshape(-1)
    if error.shape != std.shape or error.size == 0:
        raise ValueError("error and predictive_std must be non-empty and equally shaped")
    valid = np.isfinite(error) & np.isfinite(std) & (error >= 0) & (std >= 0)
    if not valid.any():
        raise ValueError("no valid residuals")
    half_width = scale * std[valid]
    return float(np.mean(error[valid] <= half_width)), float(np.mean(2.0 * half_width))


def sample_based_crps(samples: np.ndarray, reference: np.ndarray, region: np.ndarray) -> float:
    r"""Mean empirical CRPS over an ensemble, per-pixel then averaged over the ROI.

    Uses the standard NRG (Gneiting-Raftery) ensemble estimator for a single
    scalar target y and ensemble x_1..x_K:

        CRPS ~= (1/K) sum_i |x_i - y| - (1/(2K^2)) sum_i sum_j |x_i - x_j|

    which reduces to the pinball-loss-consistent proper score without
    assuming a Gaussian predictive distribution -- appropriate here since
    diffusion/CVAE sample ensembles are not generally Gaussian. Lower is
    better; a point estimate (K=1, or all samples identical) degenerates to
    plain absolute error, which is the correct limiting behaviour.
    """

    selected = _roi_samples(samples, region)
    truth = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("reference and region must have equal shapes")
    target = truth[mask]
    count = selected.shape[0]
    mean_absolute_error_term = np.mean(np.abs(selected - target[None, :]), axis=0)
    ordered = np.sort(selected, axis=0)
    coefficients = (2 * np.arange(count) - count + 1).astype(np.float64)
    pairwise_spread_term = np.sum(coefficients[:, None] * ordered, axis=0) / (count * count)
    crps_per_pixel = mean_absolute_error_term - 0.5 * pairwise_spread_term
    return float(np.mean(crps_per_pixel))


def interval_score(
    samples: np.ndarray,
    reference: np.ndarray,
    region: np.ndarray,
    *,
    nominal_coverage: float,
) -> float:
    r"""Mean Gneiting-Raftery interval score for the empirical central interval.

        IS = (u - l) + (2/alpha)(l - y) 1[y < l] + (2/alpha)(y - u) 1[y > u]

    where [l, u] is the (1 - alpha) empirical central interval from the
    sample ensemble. A single number that trades off interval width against
    coverage failures -- unlike coverage or width reported alone, a model
    cannot game this score by simply widening every interval. Lower is
    better.
    """

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie strictly in (0,1)")
    selected = _roi_samples(samples, region)
    truth = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("reference and region must have equal shapes")
    alpha = 1.0 - nominal_coverage
    tail = alpha / 2.0
    lower, upper = np.quantile(selected, [tail, 1.0 - tail], axis=0)
    target = truth[mask]
    width = upper - lower
    below = target < lower
    above = target > upper
    penalty = np.zeros_like(width)
    penalty[below] += (2.0 / alpha) * (lower[below] - target[below])
    penalty[above] += (2.0 / alpha) * (target[above] - upper[above])
    return float(np.mean(width + penalty))


def sparsification_error(predictive_std: np.ndarray, absolute_error: np.ndarray) -> float:
    r"""Area between the oracle and predicted-uncertainty sparsification curves.

    Sorts pixels by predicted uncertainty (descending) and, at each removal
    fraction, records the mean absolute error remaining among the pixels not
    yet discarded -- the "sparsification curve". A well-calibrated ranking
    should discard its highest-error pixels first, so this curve should
    closely track the oracle curve obtained by sorting by *true* absolute
    error instead. The area between the two curves (AUSE, area under the
    sparsification error) is 0 for a perfect ranking and grows with worse
    ranking; unlike Spearman correlation, it is sensitive to the *shape* of
    the whole curve, not just monotonic association.
    """

    error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    std = np.asarray(predictive_std, dtype=np.float64).reshape(-1)
    if error.shape != std.shape or error.size < 2:
        raise ValueError("predictive_std and absolute_error must be non-empty, equal length, size>=2")
    if not (np.isfinite(error).all() and np.isfinite(std).all()):
        raise ValueError("predictive_std and absolute_error must be finite")
    count = error.size

    def _remaining_mean_curve(order: np.ndarray) -> np.ndarray:
        # order[0] is discarded first; remaining-mean after discarding the
        # first k is the mean of error[order[k:]], for k=0..count-1.
        sorted_error = error[order]
        suffix_sum = np.cumsum(sorted_error[::-1])[::-1]
        remaining_count = np.arange(count, 0, -1)
        return suffix_sum / remaining_count

    predicted_order = np.argsort(-std, kind="stable")
    oracle_order = np.argsort(-error, kind="stable")
    predicted_curve = _remaining_mean_curve(predicted_order)
    oracle_curve = _remaining_mean_curve(oracle_order)
    return float(np.mean(predicted_curve - oracle_curve))


def bounded_scaled_interval_coverage(
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float]:
    """Coverage and width after clipping a symmetric interval to valid bounds."""

    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    mean = np.asarray(predictive_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(predictive_std, dtype=np.float64).reshape(-1)
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    if mean.shape != std.shape or mean.shape != truth.shape or mean.size == 0:
        raise ValueError("mean, std and reference must be non-empty and equally shaped")
    valid = np.isfinite(mean) & np.isfinite(std) & np.isfinite(truth) & (std >= 0)
    if not valid.any():
        raise ValueError("no valid interval inputs")
    mean, std, truth = mean[valid], std[valid], truth[valid]
    lower = np.clip(mean - scale * std, lower_bound, upper_bound)
    upper = np.clip(mean + scale * std, lower_bound, upper_bound)
    return float(np.mean((truth >= lower) & (truth <= upper))), float(np.mean(upper - lower))
