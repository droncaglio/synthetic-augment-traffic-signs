#!/usr/bin/env python
"""Prepare DFG-TSD (Slovenian traffic-sign detection) into the SAME prepared/ spine as
TT100K, so the full-201-style pipeline (make_full_subset -> tile -> allocation -> train ->
eval) runs on it as a 2nd-dataset reproducibility check.

DFG differs from TT100K in two ways, both handled here:
  1. COCO format (bbox = [x,y,w,h] list, category_id -> name), not the TT100K imgs-dict.
  2. **Variable per-image size** (not fixed 2048² panoramas): DFG images are ~1920×1080
     (some 720w). We store the ACTUAL width/height per record so tiling/reconstruct/eval
     can denormalize per image (the downstream per-image-size change).

DFG ships a fixed train/test split; we keep it and carve a seeded val slice out of train
(no pHash needed — DFG scenes are curated/distinct; the official split is the benchmark).

Outputs (mirrors prepare.py):
  data/dfg/prepared/panoramas.jsonl  — {id, path, split_orig, width, height, objects:[{category,xyxy}]}
  data/dfg/prepared/catalog.json
  data/dfg/prepared/splits.json      — {train:[...], val:[...], test:[...], meta:{...}}

Then reuse (unchanged): make_full_subset.py --prepared data/dfg/prepared,
build_allocation.py --prepared data/dfg/prepared.

Usage:
  python scripts/detection/prepare_dfg.py --raw data/dfg --out data/dfg/prepared --val-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from detection.prepare import build_catalog  # noqa: E402  (reused, dataset-agnostic)


def _coco_records(coco: dict, split_orig: str) -> list[dict]:
    """One record per image; bbox [x,y,w,h] -> xyxy; per-image width/height preserved."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        if a.get("ignore") or a.get("iscrowd"):
            continue  # DFG marks none in practice, but be defensive
        anns_by_img.setdefault(a["image_id"], []).append(a)
    recs = []
    for im in coco["images"]:
        objs = []
        for a in anns_by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            objs.append({"category": cats[a["category_id"]],
                         "xyxy": [float(x), float(y), float(x + w), float(y + h)]})
        recs.append({
            "id": Path(im["file_name"]).stem,          # file stem = stable id across splits
            "path": im["file_name"],                    # FLAT: images/<file_name> (see fetch_dfg)
            "split_orig": split_orig,
            "width": int(im["width"]),                  # PER-IMAGE (not fixed 2048)
            "height": int(im["height"]),
            "objects": objs,
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/dfg", help="dir with train.json + test.json")
    ap.add_argument("--out", default="data/dfg/prepared")
    ap.add_argument("--val-frac", type=float, default=0.15, help="val carved from DFG train")
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

    train = json.loads((raw / "train.json").read_text())
    test = json.loads((raw / "test.json").read_text())
    recs = _coco_records(train, "train") + _coco_records(test, "test")

    with jsonl_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    catalog = build_catalog(recs)
    catalog_path.write_text(json.dumps(catalog, indent=2))

    # splits: test = DFG test; train/val = seeded carve of DFG train (image-level, no leak
    # within DFG since each image is an independent scene).
    train_ids = [r["id"] for r in recs if r["split_orig"] == "train"]
    test_ids = [r["id"] for r in recs if r["split_orig"] == "test"]
    rng = random.Random(args.seed)
    rng.shuffle(train_ids)
    n_val = int(round(args.val_frac * len(train_ids)))
    val_ids, tr_ids = train_ids[:n_val], train_ids[n_val:]
    splits = {"train": sorted(tr_ids), "val": sorted(val_ids), "test": sorted(test_ids),
              "meta": {"dataset": "dfg", "seed": args.seed, "val_frac": args.val_frac,
                       "source_split": "official DFG train/test; val carved from train",
                       "n_train": len(tr_ids), "n_val": len(val_ids), "n_test": len(test_ids)}}
    splits_path.write_text(json.dumps(splits, indent=2))

    print(f"[ok] DFG prepared -> {out}")
    print(f"  panoramas: {len(recs)} ({catalog['n_categories']} categorias, "
          f"{sum(c['instances'] for c in catalog['categories'].values())} objetos)")
    print(f"  splits: train {len(tr_ids)} | val {len(val_ids)} | test {len(test_ids)}")
    ws = sorted(r["width"] for r in recs)
    print(f"  img width: min {ws[0]}, mediana {ws[len(ws)//2]}, max {ws[-1]} (variável -> per-image size)")


if __name__ == "__main__":
    main()
