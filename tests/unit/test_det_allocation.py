"""Unit tests for detection.allocation (frequency water-filling)."""
import pytest

from detection.allocation import (
    train_instance_counts, budget_from_K, compute_water_fill, build_allocation,
)


def test_water_fill_sums_to_B_and_levels_rarest():
    # counts a=100, b=10, c=50; B=80 -> b:60 (10->70), c:20 (50->70), a:0
    alloc = compute_water_fill({0: 100, 1: 10, 2: 50}, B=80)
    assert sum(alloc.values()) == 80
    assert alloc[0] == 0                      # already above the floor
    assert alloc[1] == 60 and alloc[2] == 20  # both reach effective 70
    # rarer class never gets less budget than a more common one
    assert alloc[1] > alloc[2] > alloc[0]


def test_water_fill_zero_budget():
    assert compute_water_fill({0: 5, 1: 3}, 0) == {0: 0, 1: 0}


def test_water_fill_alpha_cap_limits_tiny_classes():
    # class 1 has 10 real -> alpha=3 caps it at 30; without cap it would get 60
    alloc = compute_water_fill({0: 100, 1: 10, 2: 50}, B=80, alpha=3.0)
    assert alloc[1] == 30                 # capped at 3*10 (was 60 uncapped)
    assert sum(alloc.values()) == 80      # remainder redistributed to class 2
    assert alloc[2] == 50 and alloc[0] == 0


def test_water_fill_all_capped_budget_shortfall():
    # every class capped at 2x -> max total = 40 < B=100; sum reflects the cap
    alloc = compute_water_fill({0: 10, 1: 10}, B=100, alpha=2.0)
    assert alloc == {0: 20, 1: 20}
    assert sum(alloc.values()) == 40      # < B, effective budget limited by caps


def test_water_fill_g_max_absolute_cap():
    alloc = compute_water_fill({0: 5, 1: 5}, B=100, g_max=8)
    assert alloc == {0: 8, 1: 8}          # absolute cap per class


def test_water_fill_is_deterministic_and_arm_independent():
    counts = {0: 100, 1: 7, 2: 7, 3: 40}
    a = compute_water_fill(counts, 33)
    b = compute_water_fill(counts, 33)
    assert a == b                              # identical across calls (arms)
    assert sum(a.values()) == 33
    # tie between classes 1 and 2 (equal counts) resolved deterministically, near-equal
    assert abs(a[1] - a[2]) <= 1


def test_water_fill_floor_invariant():
    counts = {0: 100, 1: 10, 2: 50}
    alloc = compute_water_fill(counts, 80)
    eff = {c: counts[c] + alloc[c] for c in counts}
    filled = [c for c in counts if alloc[c] > 0]
    floor = max(eff[c] for c in filled)
    # every unfilled class sits at or above the floor
    assert all(counts[c] >= floor for c in counts if alloc[c] == 0)


def test_budget_from_K():
    assert budget_from_K({0: 100, 1: 100}, 0.5) == 100
    assert budget_from_K({0: 3}, 0.5) == 2  # round(1.5) -> 2


def test_train_instance_counts_train_only():
    rec = {
        "p_train": {"objects": [{"category": "A"}, {"category": "A"}, {"category": "Z"}]},
        "p_test":  {"objects": [{"category": "A"}]},  # must NOT be counted
    }
    counts = train_instance_counts(rec, ["p_train"], {"A": 0, "B": 1})
    assert counts == {0: 2, 1: 0}  # A=2 (train only), B=0, Z ignored (not subset)


def test_build_allocation_end_to_end():
    rec = {f"p{i}": {"objects": [{"category": "A"}]} for i in range(10)}
    rec.update({f"q{i}": {"objects": [{"category": "B"}]} for i in range(2)})
    train_ids = list(rec)
    spec = build_allocation(rec, train_ids, {"A": 0, "B": 1}, K=0.5)
    assert spec["B"] == 6                      # round(0.5 * 12)
    assert sum(spec["alloc"].values()) == 6
    assert spec["alloc"]["1"] >= spec["alloc"]["0"]  # B rarer -> more budget
