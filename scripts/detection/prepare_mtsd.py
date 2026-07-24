#!/usr/bin/env python
"""Prepare MTSD (Mapillary Traffic Sign Dataset) into the TT100K prepared spine — 3rd
dataset for the reproducibility study. COCO format, like DFG, but:
  * split: MTSD ships train + val (its test is held out by Mapillary). We use MTSD **val
    as our TEST** and carve our VAL from MTSD train (seeded). Mirrors the DFG treatment.
  * images are HIGHLY variable (320..13312 px wide, median ~3840) -> per-image size is
    essential (handled by the isotropic R=max(W,H) normalization already in the pipeline).

Reuses build_catalog. make_full_subset + build_allocation run unchanged on the output.

Outputs: data/mtsd/prepared/{panoramas.jsonl, catalog.json, splits.json}

Usage:
  python scripts/detection/prepare_mtsd.py --raw data/mtsd --out data/mtsd/prepared --val-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from detection.prepare import build_catalog  # noqa: E402


def _coco_records(coco: dict, split_orig: str) -> list[dict]:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict = {}
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)
    recs = []
    for im in coco["images"]:
        objs = []
        for a in anns_by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            objs.append({"category": cats[a["category_id"]],
                         "xyxy": [float(x), float(y), float(x + w), float(y + h)]})
        recs.append({
            "id": Path(im["file_name"]).stem,
            "path": im["file_name"],
            "split_orig": split_orig,
            "width": int(im["width"]),
            "height": int(im["height"]),
            "objects": objs,
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/mtsd")
    ap.add_argument("--out", default="data/mtsd/prepared")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path, catalog_path, splits_path = (out / "panoramas.jsonl", out / "catalog.json",
                                             out / "splits.json")
    if jsonl_path.exists() and splits_path.exists() and not args.force:
        print(f"[skip] {out} already prepared (use --force)")
        return

    train = json.loads((raw / "train_coco.json").read_text())
    val = json.loads((raw / "val_coco.json").read_text())
    # MTSD val -> our TEST; MTSD train -> our train + carved val
    recs = _coco_records(train, "train") + _coco_records(val, "test")

    with jsonl_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    catalog = build_catalog(recs)
    catalog_path.write_text(json.dumps(catalog, indent=2))

    train_ids = [r["id"] for r in recs if r["split_orig"] == "train"]
    test_ids = [r["id"] for r in recs if r["split_orig"] == "test"]
    rng = random.Random(args.seed)
    rng.shuffle(train_ids)
    n_val = int(round(args.val_frac * len(train_ids)))
    val_ids, tr_ids = train_ids[:n_val], train_ids[n_val:]
    splits = {"train": sorted(tr_ids), "val": sorted(val_ids), "test": sorted(test_ids),
              "meta": {"dataset": "mtsd", "seed": args.seed, "val_frac": args.val_frac,
                       "source_split": "MTSD val as TEST; our val carved from MTSD train",
                       "n_train": len(tr_ids), "n_val": len(val_ids), "n_test": len(test_ids)}}
    splits_path.write_text(json.dumps(splits, indent=2))

    print(f"[ok] MTSD prepared -> {out}")
    print(f"  panoramas: {len(recs)} ({catalog['n_categories']} categorias, "
          f"{sum(c['instances'] for c in catalog['categories'].values())} objetos)")
    print(f"  splits: train {len(tr_ids)} | val {len(val_ids)} | test {len(test_ids)}")
    ws = sorted(r["width"] for r in recs)
    print(f"  img width: min {ws[0]}, mediana {ws[len(ws)//2]}, max {ws[-1]} (MUITO variável -> per-image)")


if __name__ == "__main__":
    main()
