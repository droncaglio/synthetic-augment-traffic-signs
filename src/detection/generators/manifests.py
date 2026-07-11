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
from collections import defaultdict
from pathlib import Path

Box = list  # [cx, cy, w, h] normalized to the tile


def index_instances_by_class(labels_dir: str | Path) -> dict[int, list[tuple[str, Box]]]:
    """Scan train tile labels -> {class_id: [(tile_stem, [cx,cy,w,h]), ...]} (train-only pool)."""
    index: dict[int, list[tuple[str, Box]]] = defaultdict(list)
    for txt in sorted(Path(labels_dir).glob("*.txt")):
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            index[cid].append((txt.stem, [float(v) for v in parts[1:5]]))
    return dict(index)


def select_sources(alloc: dict[str, int], index: dict[int, list], seed: int) -> list[dict]:
    """Pick alloc[c] source instances of class c (seeded, with replacement if needed).

    Returns the SHARED source manifest: [{class_id, source_tile, bbox}]. Deterministic
    given (alloc, index, seed) — every content arm consumes this identical list.
    """
    rng = random.Random(seed)
    sources: list[dict] = []
    for cid_str in sorted(alloc, key=lambda k: int(k)):
        cid, n = int(cid_str), int(alloc[cid_str])
        pool = index.get(cid, [])
        if not pool or n <= 0:
            continue
        for _ in range(n):
            tile, box = pool[rng.randrange(len(pool))]
            sources.append({"class_id": cid, "source_tile": tile, "bbox": list(box)})
    return sources


def assign_placements(sources: list[dict], background_tiles: list[str], seed: int,
                      scale_jitter: float = 0.2, margin: float = 0.15) -> list[dict]:
    """Copy-paste placement manifest: recipient background tile + new (cx,cy,w,h) per source.

    Seeded and independent of the arm — records WHERE each source sign is relocated.
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
