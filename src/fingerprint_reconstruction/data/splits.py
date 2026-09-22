"""Leakage-safe dataset splitting at subject or finger level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class GroupRecord:
    """Minimal manifest record required for leakage-safe splitting."""

    sample_id: str
    subject_id: str
    finger_id: str

    def group_key(self, level: Literal["subject", "finger"]) -> str:
        if level == "subject":
            return self.subject_id
        if level == "finger":
            return f"{self.subject_id}::{self.finger_id}"
        raise ValueError(f"unsupported grouping level: {level}")


@dataclass(frozen=True)
class GroupSplit:
    assignments: Mapping[str, SplitName]
    group_level: Literal["subject", "finger"]
    seed: int

    def samples(self, split: SplitName) -> Tuple[str, ...]:
        return tuple(k for k, value in self.assignments.items() if value == split)


def _validate_fractions(fractions: Sequence[float]) -> Tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("fractions must contain train, validation, and test")
    values = tuple(float(x) for x in fractions)
    if any(x <= 0.0 for x in values):
        raise ValueError("all split fractions must be strictly positive")
    if not np.isclose(sum(values), 1.0, atol=1e-8):
        raise ValueError("split fractions must sum to one")
    return values  # type: ignore[return-value]


def _largest_remainder_counts(n: int, fractions: Sequence[float]) -> np.ndarray:
    raw = n * np.asarray(fractions, dtype=np.float64)
    counts = np.floor(raw).astype(np.int64)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1

    # With very small numbers of groups, standard largest-remainder allocation
    # can assign zero groups to a positive-probability split. Preserve the total
    # while enforcing the declared contract that train/validation/test are all
    # represented. This branch does not affect normal-sized datasets.
    for empty_index in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts > 1)
        if len(donors) == 0:
            break
        donor_index = donors[np.argmax(counts[donors] - raw[donors])]
        counts[donor_index] -= 1
        counts[empty_index] += 1
    return counts


def make_group_split(
    records: Iterable[GroupRecord],
    *,
    group_level: Literal["subject", "finger"] = "subject",
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 1729,
) -> GroupSplit:
    """Assign all samples in a group to exactly one split.

    The function is deterministic for a fixed manifest, grouping level, and
    seed. Input order does not change the result.
    """

    fraction_values = _validate_fractions(fractions)
    records_list = list(records)
    if not records_list:
        raise ValueError("records cannot be empty")

    sample_ids = [record.sample_id for record in records_list]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    by_group: Dict[str, List[str]] = {}
    for record in records_list:
        key = record.group_key(group_level)
        by_group.setdefault(key, []).append(record.sample_id)

    groups = np.asarray(sorted(by_group), dtype=object)
    if len(groups) < 3:
        raise ValueError("at least three independent groups are required")

    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]
    counts = _largest_remainder_counts(len(groups), fraction_values)
    if np.any(counts == 0):
        raise ValueError("each split must contain at least one group")

    boundaries = np.cumsum(counts)
    split_groups = {
        "train": groups[: boundaries[0]],
        "validation": groups[boundaries[0] : boundaries[1]],
        "test": groups[boundaries[1] :],
    }

    assignments: Dict[str, SplitName] = {}
    for split_name, assigned_groups in split_groups.items():
        for group in assigned_groups:
            for sample_id in by_group[str(group)]:
                assignments[sample_id] = split_name  # type: ignore[assignment]

    if set(assignments) != set(sample_ids):
        raise RuntimeError("internal error: not all samples were assigned")

    return GroupSplit(assignments, group_level, seed)
