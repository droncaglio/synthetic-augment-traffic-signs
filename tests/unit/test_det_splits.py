"""Unit tests for detection.splits — anti-leak is the highest-priority invariant."""
from collections import Counter

import pytest

from detection.splits import (
    _hamming, build_groups, assign_groups, assert_no_leak, get_donor_pool,
)


# ---- perceptual-hash grouping ------------------------------------------------
def test_hamming():
    assert _hamming(0b0000, 0b0000) == 0
    assert _hamming(0b0000, 0b0001) == 1
    assert _hamming(0b0000, 0b1111) == 4


def test_build_groups_near_dup_and_singletons():
    # a,b within T=1; c far -> {a,b}, {c}
    groups = build_groups({"a": 0b0000, "b": 0b0001, "c": 0b1111}, T=1)
    assert ["a", "b"] in groups
    assert ["c"] in groups
    assert len(groups) == 2


def test_build_groups_transitive():
    # a-b dist1, b-c dist1, a-c dist2; with T=1 union is transitive -> one group
    groups = build_groups({"a": 0b0000, "b": 0b0001, "c": 0b0011}, T=1)
    assert groups == [["a", "b", "c"]]


def test_build_groups_deterministic_order():
    h = {"z": 0, "a": 0b111111, "m": 1}
    assert build_groups(h, T=1) == [["a"], ["m", "z"]]  # sorted by first id


# ---- stratified assignment ---------------------------------------------------
def _records(spec):
    """spec: {pid: {class: n_instances}} -> rec_by_id with objects."""
    rec = {}
    for pid, classes in spec.items():
        objs = []
        for c, n in classes.items():
            objs += [{"category": c, "xyxy": [0, 0, 1, 1]} for _ in range(n)]
        rec[pid] = {"id": pid, "objects": objs}
    return rec


def _class_counts(ids, rec_by_id, cls):
    ctr = Counter()
    for pid in ids:
        for o in rec_by_id[pid]["objects"]:
            ctr[o["category"]] += 1
    return ctr[cls]


def test_assign_meets_min_test_support():
    # A in p0..p14 (15 inst), B in p15..p19 (2 each = 10 inst)
    spec = {f"p{i}": {"A": 1} for i in range(15)}
    spec.update({f"p{i}": {"B": 2} for i in range(15, 20)})
    rec = _records(spec)
    groups = [[pid] for pid in spec]
    out = assign_groups(groups, rec, ["A", "B"], {"train": .7, "val": .15, "test": .15},
                        min_test_support=3, seed=42)
    assert _class_counts(out["test"], rec, "A") >= 3
    assert _class_counts(out["test"], rec, "B") >= 3
    # partition is complete and disjoint
    allids = set(out["train"]) | set(out["val"]) | set(out["test"])
    assert allids == set(spec)
    assert len(out["train"]) + len(out["val"]) + len(out["test"]) == 20


def test_assign_deterministic():
    spec = {f"p{i}": {"A": 1, "B": 1} for i in range(30)}
    rec = _records(spec)
    groups = [[pid] for pid in spec]
    args = (groups, rec, ["A", "B"], {"train": .7, "val": .15, "test": .15}, 3, 7)
    assert assign_groups(*args) == assign_groups(*args)


def test_assign_keeps_groups_intact():
    # a 2-panorama near-dup group must stay together
    spec = {f"p{i}": {"A": 5} for i in range(10)}
    rec = _records(spec)
    groups = [["p0", "p1"]] + [[f"p{i}"] for i in range(2, 10)]
    out = assign_groups(groups, rec, ["A"], {"train": .7, "val": .15, "test": .15}, 3, 42)
    split_of = {pid: s for s in ("train", "val", "test") for pid in out[s]}
    assert split_of["p0"] == split_of["p1"]


def test_assign_raises_when_support_impossible():
    spec = {f"p{i}": {"A": 1} for i in range(10)}  # A=10 total, need 20 in test
    rec = _records(spec)
    groups = [[pid] for pid in spec]
    with pytest.raises(ValueError):
        assign_groups(groups, rec, ["A"], {"train": .7, "val": .15, "test": .15},
                      min_test_support=20, seed=42)


# ---- anti-leak assertions ----------------------------------------------------
def test_assert_no_leak_passes_and_detects():
    good = {"train": ["a", "b"], "val": ["c"], "test": ["d"],
            "groups": [["a", "b"], ["c"], ["d"]]}
    assert_no_leak(good)  # no raise

    dup = {"train": ["a"], "val": ["a"], "test": ["d"], "groups": [["a"], ["d"]]}
    with pytest.raises(AssertionError):
        assert_no_leak(dup)

    split_group = {"train": ["a"], "val": ["b"], "test": [],
                   "groups": [["a", "b"]]}  # group spans train+val
    with pytest.raises(AssertionError):
        assert_no_leak(split_group)


def test_get_donor_pool_is_train_only():
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    assert get_donor_pool(splits) == ["a", "b"]


def test_get_donor_pool_rejects_leak():
    leaky = {"train": ["a", "b"], "val": ["a"], "test": []}  # 'a' in train AND val
    with pytest.raises(AssertionError):
        get_donor_pool(leaky)


def test_assign_warns_on_ratio_drift():
    # every panorama has A,B; a high min_test_support forces ~half into test,
    # blowing the 15% target -> a ratio-drift warning (support is reachable).
    spec = {f"p{i}": {"A": 1, "B": 1} for i in range(20)}
    rec = _records(spec)
    groups = [[pid] for pid in spec]
    out = assign_groups(groups, rec, ["A", "B"], {"train": .7, "val": .15, "test": .15},
                        min_test_support=10, seed=1)
    assert _class_counts(out["test"], rec, "A") >= 10
    assert any("ratio" in w for w in out["_warnings"])
