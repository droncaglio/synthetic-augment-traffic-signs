"""Parse TT100K annotations into normalized per-panorama records + class catalog.

Pure parsing, **subset-agnostic** and **without image I/O** (so it is unit-testable
without the ~36 GB dataset). Downstream:
  * subset.py  reads the catalog to pick the 15-25 class subset.
  * splits.py  reads panoramas.jsonl (panorama ids + objects) to split by panorama.
  * tiling.py  reads panoramas.jsonl to emit tiles + labels + ignore regions.
  * evaluate.py reads panoramas.jsonl for panorama-level ground truth.

TT100K annotations_all.json shape:
  {"types": [...], "imgs": {id: {"path": "train/62627.jpg", "id": "62627",
     "objects": [{"bbox": {"xmin","ymin","xmax","ymax"}, "category": "pn"}, ...]}}}

Panoramas are 2048x2048 (uniform in TT100K 2021); we take the size from config
rather than opening every image. bbox coords are absolute pixels.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

PANORAMA_SIZE_DEFAULT = 2048
_KNOWN_SPLITS = ("train", "test", "other")


def split_from_path(path: str) -> str:
    """Original TT100K split inferred from the image path prefix.

    Note: we RE-SPLIT by panorama later (splits.py); this is kept only for
    provenance/audit, never used to assign train/val/test.
    """
    head = path.split("/", 1)[0] if "/" in path else ""
    return head if head in _KNOWN_SPLITS else "unknown"


def parse_bbox(bbox: dict) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) absolute pixels from either TT100K bbox shape.

    Supports {xmin,ymin,xmax,ymax} (TT100K 2021) and {x,y,w,h} (defensive).
    """
    if "xmin" in bbox:
        return (float(bbox["xmin"]), float(bbox["ymin"]),
                float(bbox["xmax"]), float(bbox["ymax"]))
    if "x" in bbox and "w" in bbox:
        x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
        return (x, y, x + w, y + h)
    raise ValueError(f"parse_bbox: unrecognized bbox keys {sorted(bbox)}")


def xyxy_to_yolo(xyxy: tuple[float, float, float, float], w: int, h: int
                 ) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) pixels -> (cx,cy,bw,bh) normalized to [0,1]. Mirrors the
    legacy copy_paste_offline.abs_to_yolo convention."""
    if w <= 0 or h <= 0:
        raise ValueError(f"xyxy_to_yolo: invalid image size {w}x{h}")
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (cx, cy, bw, bh)


def iter_records(annotations: dict, panorama_size: int = PANORAMA_SIZE_DEFAULT
                 ) -> Iterator[dict]:
    """Yield one normalized record per annotated panorama.

    Record: {id, path, split_orig, width, height,
             objects: [{category, xyxy: [x1,y1,x2,y2]}]}
    """
    imgs = annotations.get("imgs", {})
    for pid, entry in imgs.items():
        path = entry.get("path", "")
        objects = []
        for obj in entry.get("objects", []):
            cat = obj.get("category")
            if cat is None or cat == "?":
                continue  # malformed / missing category
            x1, y1, x2, y2 = parse_bbox(obj["bbox"])
            objects.append({"category": cat, "xyxy": [x1, y1, x2, y2]})
        yield {
            "id": str(entry.get("id", pid)),
            "path": path,
            "split_orig": split_from_path(path),
            "width": panorama_size,
            "height": panorama_size,
            "objects": objects,
        }


def build_catalog(records: Iterable[dict]) -> dict:
    """Full class catalog: per-category instance and image counts (all splits).

    Image count = number of distinct panoramas containing >=1 instance of the class.
    """
    inst = defaultdict(int)
    imgs = defaultdict(int)
    n_pan = 0
    split_counts = defaultdict(int)
    for rec in records:
        n_pan += 1
        split_counts[rec["split_orig"]] += 1
        seen = set()
        for o in rec["objects"]:
            inst[o["category"]] += 1
            seen.add(o["category"])
        for c in seen:
            imgs[c] += 1
    categories = {
        c: {"instances": inst[c], "images": imgs[c]}
        for c in sorted(inst, key=lambda k: (-inst[k], k))
    }
    return {
        "n_panoramas": n_pan,
        "n_categories": len(categories),
        "split_orig_counts": dict(split_counts),
        "categories": categories,
    }


def prepare(annotations_path: str | Path, out_dir: str | Path,
            panorama_size: int = PANORAMA_SIZE_DEFAULT, force: bool = False) -> dict:
    """Parse annotations -> prepared/panoramas.jsonl + prepared/catalog.json.

    Idempotent: skips if catalog.json exists and force is False. Returns the catalog.
    """
    out_dir = Path(out_dir)
    catalog_path = out_dir / "catalog.json"
    jsonl_path = out_dir / "panoramas.jsonl"
    if catalog_path.exists() and jsonl_path.exists() and not force:
        return json.loads(catalog_path.read_text())

    annotations = json.loads(Path(annotations_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_records(annotations, panorama_size))
    with jsonl_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    catalog = build_catalog(records)
    catalog_path.write_text(json.dumps(catalog, indent=2))
    return catalog
