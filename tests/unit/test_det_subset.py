"""Unit tests for detection.subset (deterministic class selection)."""
import statistics

import pytest

from detection.subset import select_subset, _pick_evenly, _three_bands


def _catalog(counts):
    """Build a catalog dict (categories ordered by -instances) from a count list."""
    names = [f"c{i:02d}" for i in range(len(counts))]
    cats = {n: {"instances": c, "images": c} for n, c in zip(names, counts)}
    # ensure desc order (prepare guarantees this)
    cats = dict(sorted(cats.items(), key=lambda kv: (-kv[1]["instances"], kv[0])))
    return {"n_panoramas": sum(counts), "n_categories": len(counts), "categories": cats}


COUNTS = [1000, 500, 400, 300, 250, 200, 180, 160, 140, 120,
          100, 90, 85, 82, 81, 50, 40, 30, 20, 10]  # first 15 are >= 80


def test_pick_evenly_spreads():
    band = [f"c{i}" for i in range(5)]
    assert _pick_evenly(band, 3) == ["c0", "c2", "c4"]
    assert _pick_evenly(band, 1) == ["c0"]
    assert _pick_evenly(band, 9) == band  # k>=n returns all


def test_three_bands_tail_not_largest():
    h, m, t = _three_bands([f"c{i}" for i in range(14)])  # 14 -> 5/5/4
    assert (len(h), len(m), len(t)) == (5, 5, 4)


def test_determinism():
    cat = _catalog(COUNTS)
    a = select_subset(cat, n_classes=9, min_instances=80)
    b = select_subset(cat, n_classes=9, min_instances=80)
    assert a == b


def test_respects_min_instances_and_count():
    sub = select_subset(_catalog(COUNTS), n_classes=9, min_instances=80)
    assert sub["n_classes"] == 9
    assert all(c["instances"] >= 80 for c in sub["classes"])


def test_contiguous_ids_desc_order():
    sub = select_subset(_catalog(COUNTS), n_classes=9, min_instances=80)
    assert [c["id"] for c in sub["classes"]] == list(range(9))
    insts = [c["instances"] for c in sub["classes"]]
    assert insts == sorted(insts, reverse=True)


def test_tiers_cover_head_to_tail():
    sub = select_subset(_catalog(COUNTS), n_classes=9, min_instances=80)
    for t in ("head", "mid", "tail"):
        assert sub["by_tier"][t], f"tier {t} empty"
    head_i = [c["instances"] for c in sub["classes"] if c["tier"] == "head"]
    tail_i = [c["instances"] for c in sub["classes"] if c["tier"] == "tail"]
    assert statistics.median(head_i) > statistics.median(tail_i)


def test_min_instances_changes_eligible():
    strict = select_subset(_catalog(COUNTS), n_classes=6, min_instances=200)
    assert all(c["instances"] >= 200 for c in strict["classes"])
    # only 6 classes have >= 200 instances
    assert strict["n_classes"] <= 6


def test_too_few_eligible_raises():
    with pytest.raises(ValueError):
        select_subset(_catalog([1000, 500, 10, 5]), n_classes=9, min_instances=100)
