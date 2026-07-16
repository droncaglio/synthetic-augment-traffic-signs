"""Shared source-selection + placement manifests for the content arms.

The ALLOCATION manifest (allocation.json, from allocation.py) says HOW MANY synthetic
instances per class. The SOURCE manifest picks WHICH real train instances feed those
slots — seeded and **shared by every content arm** — so the only thing that varies
between real-duplicate / bg-photometric / diffusion-bg is the background treatment
(same source instance, same in-tile position). Copy-paste consumes the same source
manifest but ADDS a placement manifest (recipient background tile + new position).

This shared selection is what makes the comparison paired: the arms never pick source
instances independently, so a ΔAP is attributable to the technique, not to which
instances happened to be augmented. Anti-leak: labels_dir must be the TRAIN tiles only.
"""
from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

Box = list  # [cx, cy, w, h] normalized to the tile


def index_instances_by_class(labels_dir: str | Path, single_sign_only: bool = True
                             ) -> dict[int, list[tuple[str, Box]]]:
    """Scan train tile labels -> {class_id: [(tile_stem, [cx,cy,w,h]), ...]} (train-only pool).

    single_sign_only (default): only index instances from tiles that contain EXACTLY ONE
    subset sign. This makes every generated tile add exactly one target instance, so the
    in-place arms (whole-tile) are instance-matched with copy-paste (single pasted sign) —
    removing the co-occurring-signal asymmetry between the arms. Sparse TT100K tiles are
    almost all single-sign; classes left without any source are reported by select_sources.
    """
    index: dict[int, list[tuple[str, Box]]] = defaultdict(list)
    for txt in sorted(Path(labels_dir).glob("*.txt")):
        rows = [ln.split() for ln in txt.read_text().splitlines() if len(ln.split()) >= 5]
        if single_sign_only and len(rows) != 1:
            continue
        for parts in rows:
            index[int(parts[0])].append((txt.stem, [float(v) for v in parts[1:5]]))
    return dict(index)


def select_sources(alloc: dict[str, int], index: dict[int, list], seed: int) -> list[dict]:
    """Pick alloc[c] source instances of class c (seeded, with replacement if needed).

    Returns the SHARED source manifest: [{class_id, source_tile, bbox}]. Deterministic
    given (alloc, index, seed) — every content arm consumes this identical list.
    """
    rng = random.Random(seed)
    sources: list[dict] = []
    starved = []
    for cid_str in sorted(alloc, key=lambda k: int(k)):
        cid, n = int(cid_str), int(alloc[cid_str])
        pool = index.get(cid, [])
        if n > 0 and not pool:
            starved.append((cid, n))   # allocated budget but no single-sign source tiles
            continue
        if not pool or n <= 0:
            continue
        for _ in range(n):
            tile, box = pool[rng.randrange(len(pool))]
            sources.append({"class_id": cid, "source_tile": tile, "bbox": list(box)})
    if starved:
        warnings.warn(f"select_sources: {len(starved)} class(es) with allocated budget but no "
                      f"single-sign source tiles (class_id, allocated): {starved} — these get "
                      f"0 synthetic tiles (under-fill).")
    return sources


def assign_placements(sources: list[dict], background_tiles: list[str], seed: int,
                      scale_jitter: float = 0.2, margin: float = 0.15) -> list[dict]:
    """Copy-paste placement manifest: recipient background tile + new (cx,cy,w,h) per source.

    Seeded deterministically: IDENTICAL (recipient, place) tuples for every arm with the same
    (sources, background_tiles, seed). This is what makes signgen_controlnet 1:1-paired with
    copy_paste — the ONLY variable between them is the pasted crop (synthetic vs real sign).
    The iteration order + rng sequence are load-bearing; don't reorder.
    """
    rng = random.Random(seed)
    placements: list[dict] = []
    for s in sources:
        recipient = background_tiles[rng.randrange(len(background_tiles))]
        jitter = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        w = min(0.9, s["bbox"][2] * jitter)
        h = min(0.9, s["bbox"][3] * jitter)
        cx = rng.uniform(margin, 1 - margin)
        cy = rng.uniform(margin, 1 - margin)
        placements.append({**s, "recipient_tile": recipient,
                           "place": [cx, cy, w, h]})
    return placements


def assign_placements_realistic(sources: list[dict], donors: list[tuple[str, Box]], seed: int
                                ) -> list[dict]:
    """REALISTIC placement: paste each source sign where a REAL sign was — recipient = a real
    single-sign tile, place = that sign's bbox (position + scale from real data), replacing it.

    donors: pooled real single-sign instances [(tile_stem, [cx,cy,w,h])] (all classes). Fixes the
    naive random placement (sign in the sky/trees) that handicaps the paste arms. Deterministic:
    same (sources, donors, seed) -> IDENTICAL placements for every arm (copy_paste/signgen stay
    1:1 paired). The recipient's own sign is dropped in make_tile (covered by the pasted sign)."""
    rng = random.Random(seed)
    placements: list[dict] = []
    for s in sources:
        dtile, dbox = donors[rng.randrange(len(donors))]
        placements.append({**s, "recipient_tile": dtile, "place": list(dbox)})
    return placements


def per_class_counts(entries: list[dict]) -> dict[int, int]:
    """Audit: realized instances per class in a manifest."""
    ctr: dict[int, int] = defaultdict(int)
    for e in entries:
        ctr[e["class_id"]] += 1
    return dict(ctr)


def save_manifest(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def load_manifest(path: str | Path):
    return json.loads(Path(path).read_text())
