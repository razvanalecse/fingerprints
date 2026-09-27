"""Paired model comparisons for repeated fingerprint reconstructions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.stats import t, ttest_1samp, wilcoxon


@dataclass(frozen=True)
class PairedTestResult:
    n: int
    mean_improvement: float
    ci95_low: float
    ci95_high: float
    cohen_dz: float
    paired_t_statistic: float
    paired_t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    wilcoxon_pvalue_holm: float = float("nan")


def paired_comparison(
    baseline: Sequence[float], candidate: Sequence[float], *, higher_is_better: bool
) -> PairedTestResult:
    """Return paired inference with improvement always coded as positive."""

    base = np.asarray(baseline, dtype=np.float64)
    model = np.asarray(candidate, dtype=np.float64)
    if base.shape != model.shape or base.ndim != 1:
        raise ValueError("paired samples must be one-dimensional and equally sized")
    finite = np.isfinite(base) & np.isfinite(model)
    base, model = base[finite], model[finite]
    if base.size < 2:
        raise ValueError("at least two finite pairs are required")
    improvement = model - base if higher_is_better else base - model
    mean = float(improvement.mean())
    standard_deviation = float(improvement.std(ddof=1))
    half_width = float(
        t.ppf(0.975, improvement.size - 1)
        * standard_deviation
        / np.sqrt(improvement.size)
    )
    t_result = ttest_1samp(improvement, popmean=0.0)
    w_result = wilcoxon(improvement, alternative="two-sided", zero_method="wilcox")
    return PairedTestResult(
        n=int(improvement.size),
        mean_improvement=mean,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
        cohen_dz=mean / standard_deviation if standard_deviation > 0 else float("inf"),
        paired_t_statistic=float(t_result.statistic),
        paired_t_pvalue=float(t_result.pvalue),
        wilcoxon_statistic=float(w_result.statistic),
        wilcoxon_pvalue=float(w_result.pvalue),
    )


def holm_adjust(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm family-wise error correction with monotonic adjusted p-values."""

    ordered = sorted(pvalues, key=pvalues.get)
    total = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * float(pvalues[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_metric_suite(
    baseline: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    directions: Mapping[str, bool],
) -> Dict[str, Dict[str, float]]:
    results = {
        name: paired_comparison(
            baseline[name], candidate[name], higher_is_better=directions[name]
        )
        for name in directions
    }
    adjusted = holm_adjust({name: result.wilcoxon_pvalue for name, result in results.items()})
    return {
        name: asdict(
            PairedTestResult(
                **{
                    **asdict(result),
                    "wilcoxon_pvalue_holm": adjusted[name],
                }
            )
        )
        for name, result in results.items()
    }
