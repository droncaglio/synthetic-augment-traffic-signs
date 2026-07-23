"""Reconstruct per-tile predictions onto the panorama + global NMS.

A sign near a tile seam appears in several overlapping tiles, so per-crop AP would
double-count it. This module maps each tile detection back to panorama-normalized
coordinates, drops detections falling inside ignore regions, and runs per-class
global NMS so each real sign survives once. The result feeds evaluate.py -> ap_by_size.

Detections are dicts: {"class_id": int, "conf": float, "box": (cx,cy,w,h)} with the
box normalized to the panorama (matching the AP evaluator convention).
"""
from __future__ import annotations

from typing import Iterable

Box = tuple[float, float, float, float]  # (cx, cy, w, h)


def _to_xyxy(b: Box) -> tuple[float, float, float, float]:
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_cxcywh(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pano_ref(tile_entry: dict, panorama_size: int) -> float:
    """Isotropic per-image reference for normalization: max(W,H) if the tile carries the
    panorama dims (variable-size datasets like DFG), else the fixed panorama_size (TT100K).

    Using ONE reference for both axes keeps IoU/area undistorted on non-square images
    (per-axis W/H would warp aspect ratio -> wrong IoU). For TT100K (W=H=2048) this is
    byte-identical to the old fixed-size path.
    """
    if "pano_w" in tile_entry and "pano_h" in tile_entry:
        return float(max(tile_entry["pano_w"], tile_entry["pano_h"]))
    return float(panorama_size)


def map_det_to_panorama(box_tile_norm: Box, tile_entry: dict, panorama_size: int) -> Box:
    """Tile-normalized (cx,cy,w,h) -> panorama-normalized (cx,cy,w,h)."""
    size = tile_entry["size"]
    ref = _pano_ref(tile_entry, panorama_size)
    cx, cy, w, h = box_tile_norm
    px_cx = cx * size + tile_entry["x_off"]
    px_cy = cy * size + tile_entry["y_off"]
    return (px_cx / ref, px_cy / ref, w * size / ref, h * size / ref)


def nms_per_class(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Greedy per-class NMS. Returns kept detections (stable, conf-desc within class)."""
    kept: list[dict] = []
    for cls in sorted({d["class_id"] for d in dets}):
        cand = sorted((d for d in dets if d["class_id"] == cls),
                      key=lambda d: -d["conf"])
        while cand:
            best = cand.pop(0)
            kept.append(best)
            cand = [d for d in cand if iou_cxcywh(best["box"], d["box"]) < iou_thresh]
    return kept


def _ignore_to_panorama(ignore_xyxy_tilepx, tile_entry: dict, panorama_size: int):
    x1, y1, x2, y2 = ignore_xyxy_tilepx
    ox, oy = tile_entry["x_off"], tile_entry["y_off"]
    ref = _pano_ref(tile_entry, panorama_size)
    return ((x1 + ox) / ref, (y1 + oy) / ref, (x2 + ox) / ref, (y2 + oy) / ref)


def reconstruct_panorama(tiles: Iterable[dict], panorama_size: int,
                         nms_iou: float = 0.5) -> list[dict]:
    """Reconstruct all tiles of ONE panorama into deduplicated detections.

    tiles: [{"entry": tile_index_entry,
             "dets": [{"class_id","conf","box"(tile-norm cxcywh)}],
             "ignores": [xyxy tile-pixel]}]
    """
    mapped: list[dict] = []
    ignore_boxes: list[tuple] = []
    for t in tiles:
        e = t["entry"]
        for d in t.get("dets", []):
            mapped.append({
                "class_id": d["class_id"], "conf": d["conf"],
                "box": map_det_to_panorama(d["box"], e, panorama_size),
            })
        for ig in t.get("ignores", []):
            ignore_boxes.append(_ignore_to_panorama(ig, e, panorama_size))

    def center_in_ignore(box: Box) -> bool:
        cx, cy = box[0], box[1]
        return any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2) in ignore_boxes)

    mapped = [d for d in mapped if not center_in_ignore(d["box"])]
    return nms_per_class(mapped, nms_iou)
