"""Tile 2048 panoramas into 640 crops WITHOUT resizing the signs (small-object safe).

Per tile we emit, for each object intersecting the tile (visible fraction vf):
  * vf <= drop_thresh                  -> DROP  (negligible sliver, nothing written)
  * subset class AND vf >= keep_thresh -> LABEL (clipped bbox, tile-local YOLO coords)
  * everything else visible            -> IGNORE (non-subset sign, or subset sign only
                                          partly visible): painted out + recorded in a
                                          sidecar so evaluate.py excludes predictions there.

Pure, unit-testable core (no image I/O): tile_grid, clip_visibility, classify, tile_objects.
I/O layer: paint_out, tile_panorama (crop + paint + write labels/ignores + tile_index).

Ignore is realized as **paint-out** (mean-fill) — pragmatic and framework-compatible
with stock Ultralytics; the metric-side rigor is the eval-time exclusion (reconstruct.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from detection.prepare import xyxy_to_yolo

Box = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Pure core
# --------------------------------------------------------------------------- #
def _offsets(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    xs = list(range(0, length - size + 1, stride))
    if xs[-1] != length - size:
        xs.append(length - size)
    return xs


def tile_grid(w: int, h: int, size: int = 640, overlap: int = 128) -> list[Box]:
    """Grid of (x1,y1,x2,y2) tiles covering the panorama; edge tiles clamped."""
    if overlap >= size:
        raise ValueError("overlap must be < size")
    stride = size - overlap
    return [(x, y, x + size, y + size)
            for y in _offsets(h, size, stride)
            for x in _offsets(w, size, stride)]


def clip_visibility(obj_xyxy: Box, tile_xyxy: Box) -> tuple[Box | None, float]:
    """Clip an object bbox to a tile. Returns (clipped_xyxy_panorama, visible_fraction).

    visible_fraction = intersection_area / original_object_area. (None, 0.0) if disjoint.
    """
    ox1, oy1, ox2, oy2 = obj_xyxy
    tx1, ty1, tx2, ty2 = tile_xyxy
    ix1, iy1 = max(ox1, tx1), max(oy1, ty1)
    ix2, iy2 = min(ox2, tx2), min(oy2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    orig = (ox2 - ox1) * (oy2 - oy1)
    vf = inter / orig if orig > 0 else 0.0
    return (ix1, iy1, ix2, iy2), vf


def classify(vf: float, in_subset: bool, keep_thresh: float, drop_thresh: float) -> str:
    """One of 'label' | 'ignore' | 'drop' for an object with visible fraction vf."""
    if vf <= drop_thresh:
        return "drop"
    if in_subset and vf >= keep_thresh:
        return "label"
    return "ignore"


def tile_objects(objects: Iterable[dict], tile_xyxy: Box, subset_ids: dict[str, int],
                 keep_thresh: float = 0.6, drop_thresh: float = 0.2
                 ) -> tuple[list[tuple[int, Box]], list[Box]]:
    """Return (labels, ignores) for one tile.

    labels: [(class_id, (cx,cy,w,h) normalized to tile)]; ignores: [tile-local xyxy px].
    """
    tx1, ty1, tx2, ty2 = tile_xyxy
    tw, th = int(tx2 - tx1), int(ty2 - ty1)
    labels: list[tuple[int, Box]] = []
    ignores: list[Box] = []
    for o in objects:
        clipped, vf = clip_visibility(tuple(o["xyxy"]), tile_xyxy)
        if clipped is None:
            continue
        in_sub = o["category"] in subset_ids
        kind = classify(vf, in_sub, keep_thresh, drop_thresh)
        cx1, cy1, cx2, cy2 = clipped
        local = (cx1 - tx1, cy1 - ty1, cx2 - tx1, cy2 - ty1)
        if kind == "label":
            labels.append((subset_ids[o["category"]], xyxy_to_yolo(local, tw, th)))
        elif kind == "ignore":
            ignores.append(local)
    return labels, ignores


# --------------------------------------------------------------------------- #
# I/O layer
# --------------------------------------------------------------------------- #
def paint_out(arr, ignore_boxes: Iterable[Box]):
    """Mean-fill each ignore box in a HxWx3 numpy image (in place). Deterministic."""
    for (x1, y1, x2, y2) in ignore_boxes:
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(round(x2)), int(round(y2))
        region = arr[y1i:y2i, x1i:x2i]
        if region.size:
            arr[y1i:y2i, x1i:x2i] = region.mean(axis=(0, 1)).astype(arr.dtype)
    return arr


def tile_panorama(image_path: str | Path, record: dict, subset_ids: dict[str, int],
                  out_dir: str | Path, size: int = 640, overlap: int = 128,
                  keep_thresh: float = 0.6, drop_thresh: float = 0.2,
                  neg_keep_fn=None) -> list[dict]:
    """Crop one panorama into tiles; write images + YOLO labels + ignore sidecars.

    Returns tile_index entries. A tile with no labels/ignores (pure background) is
    kept only if ``neg_keep_fn(tile_name)`` is truthy — lets the caller sample a
    deterministic fraction of negatives.
    """
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    pid = record["id"]
    with Image.open(image_path) as im:
        pano = np.asarray(im.convert("RGB"))
    h, w = pano.shape[:2]

    index: list[dict] = []
    for (tx1, ty1, tx2, ty2) in tile_grid(w, h, size, overlap):
        labels, ignores = tile_objects(record["objects"], (tx1, ty1, tx2, ty2),
                                        subset_ids, keep_thresh, drop_thresh)
        name = f"{pid}_{int(tx1)}_{int(ty1)}"
        if not labels and not ignores:  # pure-background tile
            if neg_keep_fn is None or not neg_keep_fn(name):
                continue
        crop = pano[int(ty1):int(ty2), int(tx1):int(tx2)].copy()
        paint_out(crop, ignores)
        Image.fromarray(crop).save(out_dir / "images" / f"{name}.jpg", quality=95)
        lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                 for cid, (cx, cy, bw, bh) in labels]
        (out_dir / "labels" / f"{name}.txt").write_text("\n".join(lines))
        if ignores:
            (out_dir / "labels" / f"{name}.ignore.json").write_text(json.dumps(ignores))
        index.append({"tile": name, "panorama_id": pid,
                      "x_off": int(tx1), "y_off": int(ty1), "size": size})
    return index
