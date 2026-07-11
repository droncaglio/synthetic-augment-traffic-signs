"""Unit tests for the shared source/placement manifests (Stage 2 keystone)."""
from detection.generators.manifests import (
    index_instances_by_class, select_sources, assign_placements, per_class_counts,
)


def _write_labels(d, files):
    d.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (d / f"{name}.txt").write_text("\n".join(lines))


def test_index_instances_all(tmp_path):
    _write_labels(tmp_path, {
        "t0": ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.05 0.05"],
        "t1": ["0 0.3 0.3 0.1 0.1"],
        "t2": [""],  # background tile, no labels
    })
    idx = index_instances_by_class(tmp_path, single_sign_only=False)
    assert len(idx[0]) == 2 and len(idx[1]) == 1
    assert ("t0", [0.5, 0.5, 0.1, 0.1]) in idx[0]


def test_index_single_sign_only_excludes_multi(tmp_path):
    _write_labels(tmp_path, {
        "t0": ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.05 0.05"],  # 2 signs -> excluded
        "t1": ["0 0.3 0.3 0.1 0.1"],                         # 1 sign -> kept
    })
    idx = index_instances_by_class(tmp_path, single_sign_only=True)
    # t0 dropped (co-occurring) -> only t1 remains for class 0; class 1 has no source
    assert idx.get(0) == [("t1", [0.3, 0.3, 0.1, 0.1])]
    assert 1 not in idx


def test_select_sources_counts_and_determinism():
    index = {0: [("a", [0.5, 0.5, 0.1, 0.1]), ("b", [0.3, 0.3, 0.1, 0.1])],
             1: [("c", [0.2, 0.2, 0.05, 0.05])]}
    alloc = {"0": 3, "1": 2}
    s1 = select_sources(alloc, index, seed=42)
    s2 = select_sources(alloc, index, seed=42)
    assert s1 == s2                                  # SHARED across arms (deterministic)
    assert per_class_counts(s1) == {0: 3, 1: 2}      # exactly alloc per class
    assert all(e["source_tile"] in ("a", "b") for e in s1 if e["class_id"] == 0)


def test_select_sources_skips_absent_class():
    index = {0: [("a", [0.5, 0.5, 0.1, 0.1])]}       # class 1 has no pool
    out = select_sources({"0": 2, "1": 5}, index, seed=1)
    assert per_class_counts(out) == {0: 2}           # class 1 skipped (no source)


def test_assign_placements_within_bounds_and_deterministic():
    sources = [{"class_id": 0, "source_tile": "a", "bbox": [0.5, 0.5, 0.1, 0.1]}]
    bgs = ["bg0", "bg1", "bg2"]
    p1 = assign_placements(sources, bgs, seed=7)
    p2 = assign_placements(sources, bgs, seed=7)
    assert p1 == p2
    e = p1[0]
    assert e["recipient_tile"] in bgs
    cx, cy, w, h = e["place"]
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0 and w <= 0.9 and h <= 0.9
