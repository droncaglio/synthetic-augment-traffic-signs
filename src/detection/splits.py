"""Panorama-level train/val/test split with near-duplicate grouping (anti-leak).

CRITICAL for validity: the split is done over full 2048 panoramas BEFORE any
tiling, and near-duplicate panoramas (perceptual hash) are grouped so a whole
group lands in a single split. This prevents crops of the same scene leaking
across train/val/test.

Pure, unit-testable core (no image I/O):
  * build_groups(id_to_hash, T)      -> Union-Find grouping by Hamming <= T
  * assign_groups(...)               -> stratified split honoring min_test_support
  * assert_no_leak(splits)           -> raises on any panorama/group leak
  * get_donor_pool(splits)           -> train-only ids (synthetic generation gateway)
I/O layer:
  * compute_phashes(records, raw_dir) -> {id: hash_int}  (PIL + imagehash)
  * make_splits(...)                  -> orchestrates the above -> splits dict

Mirrors the anti-leak contract of Paper 1 (synthetic-allocation-derma splits.py),
with grouping key = perceptual-hash component instead of lesion_id.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Pure core
# --------------------------------------------------------------------------- #
def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_groups(id_to_hash: dict[str, int], T: int) -> list[list[str]]:
    """Union-Find grouping: connect ids whose perceptual hashes differ by <= T bits.

    O(N^2) pairwise (one-time prepare step). Returns groups as sorted lists,
    ordered by their smallest id — fully deterministic.
    """
    ids = sorted(id_to_hash)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(len(ids)):
        hi = id_to_hash[ids[i]]
        for j in range(i + 1, len(ids)):
            if _hamming(hi, id_to_hash[ids[j]]) <= T:
                union(ids[i], ids[j])

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def _group_class_counts(group: list[str], rec_by_id: dict[str, dict],
                        subset_set: set[str]) -> Counter:
    ctr: Counter = Counter()
    for pid in group:
        for o in rec_by_id[pid]["objects"]:
            if o["category"] in subset_set:
                ctr[o["category"]] += 1
    return ctr


def assign_groups(groups: list[list[str]], rec_by_id: dict[str, dict],
                  subset_names: Iterable[str], ratios: dict[str, float],
                  min_test_support: int, seed: int) -> dict:
    """Assign whole groups to train/val/test.

    Phase 1: seeded shuffle, greedy fill toward panorama-count ratios.
    Phase 2 (repair): move train groups into test until every subset class has
    >= min_test_support instances in test. Raises if a class cannot reach it.
    Deterministic given (groups, records, ratios, seed).
    """
    subset_set = set(subset_names)
    gcc = [_group_class_counts(g, rec_by_id, subset_set) for g in groups]
    sizes = [len(g) for g in groups]
    total = sum(sizes)
    targets = {s: ratios[s] * total for s in _SPLITS}

    assign: dict[int, str] = {}
    cur = {s: 0 for s in _SPLITS}
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    for gi in order:
        s = max(_SPLITS, key=lambda k: targets[k] - cur[k])
        assign[gi] = s
        cur[s] += sizes[gi]

    # --- Phase 2: repair test min-support (deterministic) ---
    def test_counts() -> Counter:
        ctr: Counter = Counter()
        for gi, s in assign.items():
            if s == "test":
                ctr.update(gcc[gi])
        return ctr

    tc = test_counts()
    for cls in sorted(subset_set):
        while tc[cls] < min_test_support:
            # candidate train groups containing cls, richest first (fewest moves)
            cands = sorted(
                (gi for gi, s in assign.items() if s == "train" and gcc[gi][cls] > 0),
                key=lambda gi: (-gcc[gi][cls], gi),
            )
            if not cands:
                raise ValueError(
                    f"assign_groups: class {cls!r} cannot reach min_test_support="
                    f"{min_test_support} in test (only {tc[cls]} available in train). "
                    f"Raise subset min_instances or lower min_test_support."
                )
            gi = cands[0]
            assign[gi] = "test"
            tc = test_counts()

    # build result deterministically (explicit sorted group order, not dict order)
    result: dict = {}
    for s in _SPLITS:
        pids: list[str] = []
        for gi in sorted(assign):
            if assign[gi] == s:
                pids.extend(groups[gi])
        result[s] = sorted(pids)

    warnings: list[str] = []
    # ratio drift (the repair pass can pull groups into test)
    for s in _SPLITS:
        if total and abs(len(result[s]) / total - ratios[s]) > 0.05:
            warnings.append(
                f"{s} ratio {len(result[s]) / total:.1%} deviates >5% from target {ratios[s]:.0%}"
            )
    # subset classes absent from val (val is used for checkpoint selection)
    val_cls: Counter = Counter()
    for gi, s in assign.items():
        if s == "val":
            val_cls.update(gcc[gi])
    warnings += [f"class {c} has 0 instances in val"
                 for c in sorted(subset_set) if val_cls[c] == 0]
    result["_warnings"] = warnings
    return result


def assert_no_leak(splits: dict) -> None:
    """Raise if any panorama is in >1 split or any group spans multiple splits."""
    sets = {s: set(splits[s]) for s in _SPLITS}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise AssertionError("split leakage: panorama present in more than one split")
    id2split = {i: s for s in _SPLITS for i in sets[s]}
    for g in splits.get("groups", []):
        spans = {id2split.get(i) for i in g if i in id2split}
        if len(spans) > 1:
            raise AssertionError(f"near-duplicate group split across splits: {g}")


def get_donor_pool(splits: dict) -> list[str]:
    """Train-only panorama ids — the ONLY source allowed for synthetic generation.

    Fail-fast: refuses to return if train overlaps val/test (would leak a val/test
    scene into synthetic training data, invalidating the split).
    """
    tr = set(splits["train"])
    if tr & set(splits.get("val", [])) or tr & set(splits.get("test", [])):
        raise AssertionError("donor pool (train) overlaps val/test — refusing to leak")
    return list(splits["train"])


# --------------------------------------------------------------------------- #
# I/O layer
# --------------------------------------------------------------------------- #
def compute_phashes(records: list[dict], raw_dir: str | Path, size: int = 256) -> dict[str, int]:
    """Perceptual hash (int) per panorama. Requires the image files under raw_dir."""
    from PIL import Image
    import imagehash

    raw_dir = Path(raw_dir)
    out: dict[str, int] = {}
    for rec in records:
        img_path = raw_dir / rec["path"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB").resize((size, size))
            out[rec["id"]] = int(str(imagehash.phash(im)), 16)
    return out


def make_splits(records: list[dict], subset: dict, raw_dir: str | Path,
                seed: int = 42, phash_T: int = 5,
                ratios: dict[str, float] | None = None,
                min_test_support: int = 10) -> dict:
    """Full pipeline: phash -> groups -> stratified assignment -> splits dict."""
    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    rec_by_id = {r["id"]: r for r in records}
    id_to_hash = compute_phashes(records, raw_dir)
    # Group only panoramas with a real hash; panoramas whose image is missing
    # become their OWN singleton group (never near-dup-grouped with each other).
    present = {r["id"]: id_to_hash[r["id"]] for r in records if r["id"] in id_to_hash}
    missing = sorted(r["id"] for r in records if r["id"] not in id_to_hash)
    groups = build_groups(present, phash_T) + [[pid] for pid in missing]
    groups = sorted(groups, key=lambda g: g[0])  # stable group indices
    assigned = assign_groups(groups, rec_by_id, subset["names"], ratios, min_test_support, seed)
    splits = {
        "meta": {
            "seed": seed, "phash_T": phash_T, "ratios": ratios,
            "min_test_support": min_test_support, "subset_names": subset["names"],
            "n_groups": len(groups), "n_missing_hash": len(missing),
            "warnings": assigned.pop("_warnings", []),
        },
        "groups": groups,
        **{s: assigned[s] for s in _SPLITS},
    }
    assert_no_leak(splits)
    return splits


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(splits, indent=2))


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
