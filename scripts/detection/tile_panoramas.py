#!/usr/bin/env python
"""CLI: tile the panoramas of one split into 640 crops (+ labels + ignore sidecars).

Usage:
  python scripts/detection/tile_panoramas.py --split train \
      --prepared data/tt100k/prepared --raw data/tt100k/tt100k_2021 \
      --out data/tt100k/tiles [--size 640] [--overlap 128] [--neg-fraction 0.1]

Negative (empty) tiles are kept deterministically for a fraction of them
(seeded by tile name hash) to provide background negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.tiling import tile_panorama  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _keep_negative(name: str, frac: float) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 1000
    return h < frac * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--raw", default="data/tt100k/tt100k_2021")
    ap.add_argument("--out", default="data/tt100k/tiles")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--neg-fraction", type=float, default=0.1)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    records = {r["id"]: r for r in _load_jsonl(prepared / "panoramas.jsonl")}
    subset = json.loads((prepared / "subset.json").read_text())
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    splits = json.loads((prepared / "splits.json").read_text())

    out_dir = Path(args.out) / args.split
    # negatives only for train (val/test evaluated at panorama level anyway)
    neg_fn = (lambda name: _keep_negative(name, args.neg_fraction)) if args.split == "train" else None
    index: list[dict] = []
    for pid in splits[args.split]:
        rec = records[pid]
        img = Path(args.raw) / rec["path"]
        if not img.exists():
            continue
        index += tile_panorama(img, rec, subset_ids, out_dir, args.size, args.overlap,
                               neg_keep_fn=neg_fn)
    (out_dir / "tile_index.json").write_text(json.dumps(index))
    print(f"[{args.split}] wrote {len(index)} non-empty tiles -> {out_dir}")


if __name__ == "__main__":
    main()
