"""Tile 2048 panoramas into 640 crops WITHOUT resizing the signs (small-object safe).

Per tile we emit, for each object intersecting the tile (visible fraction vf):
  * vf <= drop_thresh                  -> DROP  (negligible sliver, nothing written)
  * subset class AND vf >= keep_thresh -> LABEL (clipped bbox, tile-local YOLO coords)
  * out-of-subset AND vf >= keep_thresh -> OTHER (open-set distractor class), IF other_id is
                                          given — labeled, NOT painted out, so the model learns
                                          "sign-but-not-a-target"; else falls through to IGNORE.
  * everything else visible            -> IGNORE (out-of-subset when other is off, or subset
                                          sign only partly visible): painted out + recorded in a
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


def classify(vf: float, in_subset: bool, keep_thresh: float, drop_thresh: float,
             other_enabled: bool = False) -> str:
    """One of 'label' | 'other' | 'ignore' | 'drop' for an object with visible fraction vf.

    other_enabled: when True, a fully-visible OUT-OF-SUBSET sign becomes an 'other' label
    (the open-set distractor class) instead of being painted out as 'ignore'. This teaches
    the detector "sign-but-not-a-target -> other" so it stops assigning a target class to
    non-target signs (the open-set false-positive confound). Default False = legacy behavior.
    """
    if vf <= drop_thresh:
        return "drop"
    if vf >= keep_thresh:
        return "label" if in_subset else ("other" if other_enabled else "ignore")
    return "ignore"


def tile_objects(objects: Iterable[dict], tile_xyxy: Box, subset_ids: dict[str, int],
                 keep_thresh: float = 0.6, drop_thresh: float = 0.2,
                 other_id: int | None = None, other_exclude: set[str] | None = None
                 ) -> tuple[list[tuple[int, Box]], list[Box]]:
    """Return (labels, ignores) for one tile.

    labels: [(class_id, (cx,cy,w,h) normalized to tile)]; ignores: [tile-local xyxy px].
    other_id: if given, fully-visible out-of-subset signs are LABELED with this class id
    (the open-set 'other' distractor) instead of painted out. subset_ids must hold only the
    target classes (not 'other') so the in-subset check stays correct.
    other_exclude: out-of-subset category NAMES that must NOT become 'other' (they fall back
    to ignore/paint-out). Use to keep fine-grained look-alikes of the target/tail classes out
    of the distractor pool — pooling them into 'other' steals target recall (the model labels
    a real target sign 'other'). Default None = collapse ALL out-of-subset into 'other'.
    """
    tx1, ty1, tx2, ty2 = tile_xyxy
    tw, th = int(tx2 - tx1), int(ty2 - ty1)
    other_exclude = other_exclude or frozenset()
    labels: list[tuple[int, Box]] = []
    ignores: list[Box] = []
    for o in objects:
        clipped, vf = clip_visibility(tuple(o["xyxy"]), tile_xyxy)
        if clipped is None:
            continue
        cat = o["category"]
        in_sub = cat in subset_ids
        other_ok = other_id is not None and not in_sub and cat not in other_exclude
        kind = classify(vf, in_sub, keep_thresh, drop_thresh, other_enabled=other_ok)
        cx1, cy1, cx2, cy2 = clipped
        local = (cx1 - tx1, cy1 - ty1, cx2 - tx1, cy2 - ty1)
        if kind == "label":
            labels.append((subset_ids[cat], xyxy_to_yolo(local, tw, th)))
        elif kind == "other":
            labels.append((other_id, xyxy_to_yolo(local, tw, th)))
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
                  mode: str = "train", neg_keep_fn=None,
                  other_id: int | None = None,
                  other_exclude: set[str] | None = None) -> list[dict]:
    """Crop one panorama into tiles; write images + YOLO labels + ignore sidecars.

    mode="train": keep a tile only if it has a subset LABEL, or it is sampled by
      ``neg_keep_fn(tile_name)`` (ignore-only / background tiles are negatives, not
      auto-kept — otherwise the ~180 non-subset classes would keep almost every tile).
    mode="eval": keep EVERY grid tile (full-coverage inference) so false positives in
      background tiles are counted; ignores are still painted out + recorded for
      reconstruct's exclusion.
    Returns tile_index entries.
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
    # Invariant: reconstruct/evaluate normalize by a fixed panorama_size (2048).
    # Fail loud if a panorama breaks it — otherwise bboxes would be mis-scaled.
    if (h, w) != (record["height"], record["width"]):
        raise ValueError(f"panorama {pid}: image is {w}x{h} but record says "
                         f"{record['width']}x{record['height']} — bbox scaling would break")

    index: list[dict] = []
    for (tx1, ty1, tx2, ty2) in tile_grid(w, h, size, overlap):
        labels, ignores = tile_objects(record["objects"], (tx1, ty1, tx2, ty2),
                                        subset_ids, keep_thresh, drop_thresh, other_id,
                                        other_exclude)
        name = f"{pid}_{int(tx1)}_{int(ty1)}"
        if mode == "train" and not labels:
            # no subset signal -> candidate negative (incl. ignore-only tiles)
            if neg_keep_fn is None or not neg_keep_fn(name):
                continue
        # mode == "eval": keep every tile (full-coverage inference)
        crop = pano[int(ty1):int(ty2), int(tx1):int(tx2)].copy()
        paint_out(crop, ignores)
        Image.fromarray(crop).save(out_dir / "images" / f"{name}.jpg", quality=95)
        lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                 for cid, (cx, cy, bw, bh) in labels]
        (out_dir / "labels" / f"{name}.txt").write_text("\n".join(lines))
        if ignores:
            (out_dir / "labels" / f"{name}.ignore.json").write_text(json.dumps(ignores))
        index.append({"tile": name, "panorama_id": pid,
                      "x_off": int(tx1), "y_off": int(ty1), "size": size,
                      "pano_w": int(w), "pano_h": int(h)})  # for per-image reconstruct
    return index
