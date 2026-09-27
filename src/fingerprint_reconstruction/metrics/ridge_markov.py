"""A Markov model of ridge-orientation transitions, used as a realism test.

`ridge_topology.py` answers "does this reconstruction have the right ridge
statistics?" with three descriptive numbers (coherence, curvature, minutiae
density). This module turns the same intuition into a *statistical* one: treat
the orientation field, read along a spatial step, as a first-order Markov
chain over quantized orientation states, and ask how predictable the next
state is given the current one.

A field that has been over-smoothed by a generative model -- the "painted
stripe" artifact -- is unusually predictable: its conditional entropy
H(theta_{t+1} | theta_t) is low and its self-transition probability is high,
because the orientation simply does not change as often as it does in a real
ridge field. Genuine ridges curve, bifurcate and end, all of which inject
transitions between orientation states.

Two properties make this comparable across images:

* Orientation is axial, so states are bins of [0, pi), not [0, 2*pi).
* Entropy estimates are biased downward when few transitions are counted, so
  real and reconstructed fields must be scored on the *same* region with the
  same number of transitions. The diagnostic script does exactly that, which
  is why the bias largely cancels in the paired comparison rather than having
  to be corrected explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeMarkovSummary:
    """First-order Markov statistics of a quantized orientation field."""

    transition_entropy_bits: float
    self_transition_rate: float
    marginal_entropy_bits: float
    transition_count: int


def quantize_orientation(theta: np.ndarray, bins: int) -> np.ndarray:
    """Map axial orientations in radians onto ``bins`` states covering [0, pi)."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    wrapped = np.mod(theta, np.pi)
    states = np.floor(wrapped / np.pi * bins).astype(np.int64)
    # guard the closed upper edge: theta exactly pi would land one past the last bin
    return np.clip(states, 0, bins - 1)


def orientation_transition_counts(
    theta: np.ndarray,
    region: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    bins: int = 12,
    step: int = 3,
) -> np.ndarray:
    """Count state transitions between pixels ``step`` apart, within ``region``.

    Transitions are pooled over both axes (down a column and along a row). Both
    endpoints must lie inside the region and, if supplied, be marked valid.
    """
    if step < 1:
        raise ValueError("step must be at least 1")
    if theta.shape != region.shape:
        raise ValueError("theta and region must have the same shape")

    usable = region.astype(bool)
    if valid is not None:
        if valid.shape != theta.shape:
            raise ValueError("valid must have the same shape as theta")
        usable = usable & valid.astype(bool)

    states = quantize_orientation(theta, bins)
    counts = np.zeros((bins, bins), dtype=np.int64)

    for axis in (0, 1):
        if theta.shape[axis] <= step:
            continue
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis] = slice(None, -step)
        tail[axis] = slice(step, None)
        pair_ok = usable[tuple(head)] & usable[tuple(tail)]
        if not pair_ok.any():
            continue
        source = states[tuple(head)][pair_ok]
        target = states[tuple(tail)][pair_ok]
        np.add.at(counts, (source, target), 1)

    return counts


def transition_entropy_bits(counts: np.ndarray) -> float:
    """Conditional entropy H(next | current) in bits, from a count matrix."""
    totals = counts.sum(axis=1, keepdims=True)
    grand_total = counts.sum()
    if grand_total == 0:
        return float("nan")
    conditional = counts / np.maximum(totals, 1)
    logs = np.zeros_like(conditional)
    positive = conditional > 0
    logs[positive] = np.log2(conditional[positive])
    row_entropy = -(conditional * logs).sum(axis=1)
    row_weight = totals.ravel() / grand_total
    return float((row_weight * row_entropy).sum())


def marginal_entropy_bits(counts: np.ndarray) -> float:
    """Entropy of the state distribution itself, in bits."""
    totals = counts.sum(axis=1)
    grand_total = totals.sum()
    if grand_total == 0:
        return float("nan")
    probabilities = totals / grand_total
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def summarize_ridge_markov(
    theta: np.ndarray,
    region: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    bins: int = 12,
    step: int = 3,
) -> RidgeMarkovSummary:
    """Full first-order Markov summary of an orientation field over a region."""
    counts = orientation_transition_counts(
        theta, region, valid=valid, bins=bins, step=step
    )
    total = int(counts.sum())
    self_rate = float(np.trace(counts) / total) if total else float("nan")
    return RidgeMarkovSummary(
        transition_entropy_bits=transition_entropy_bits(counts),
        self_transition_rate=self_rate,
        marginal_entropy_bits=marginal_entropy_bits(counts),
        transition_count=total,
    )
