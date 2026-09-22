from collections import defaultdict

import pytest

from fingerprint_reconstruction.data.splits import GroupRecord, make_group_split


def _records(subjects=20, fingers=3, impressions=2):
    return [
        GroupRecord(
            sample_id=f"s{s}_f{f}_i{i}",
            subject_id=f"s{s}",
            finger_id=f"f{f}",
        )
        for s in range(subjects)
        for f in range(fingers)
        for i in range(impressions)
    ]


def test_subject_split_has_no_subject_leakage():
    records = _records()
    split = make_group_split(records, group_level="subject", seed=99)
    observed = defaultdict(set)
    for record in records:
        observed[record.subject_id].add(split.assignments[record.sample_id])
    assert all(len(names) == 1 for names in observed.values())
    assert set(split.assignments.values()) == {"train", "validation", "test"}


def test_input_order_does_not_change_split():
    records = _records()
    forward = make_group_split(records, seed=11)
    reverse = make_group_split(reversed(records), seed=11)
    assert forward.assignments == reverse.assignments


def test_finger_level_keeps_impressions_together():
    records = _records(subjects=8)
    split = make_group_split(records, group_level="finger", seed=3)
    observed = defaultdict(set)
    for record in records:
        key = (record.subject_id, record.finger_id)
        observed[key].add(split.assignments[record.sample_id])
    assert all(len(names) == 1 for names in observed.values())


def test_duplicate_sample_ids_are_rejected():
    records = [
        GroupRecord("x", "s1", "f1"),
        GroupRecord("x", "s2", "f2"),
        GroupRecord("z", "s3", "f3"),
    ]
    with pytest.raises(ValueError, match="unique"):
        make_group_split(records)
