"""AP@IoU=0.5 stratified by relative bbox size (small/medium/large).

Metric CORE copied (verbatim, with light trimming) from the legacy detection
pipeline `synthetic-longtail-detection/src/eval/ap_by_size.py`. The legacy
per-image prediction/dataset-yaml/CLI flow is intentionally omitted — here we
evaluate at the PANORAMA level (predictions reconstructed + globally NMS'd by
detection.reconstruct, then fed into compute_ap_by_size by detection.evaluate).

Buckets are relative to image area (bbox_w * bbox_h, both normalized). For 2048
panoramas nearly every traffic sign falls in `small` — which is exactly AP@small.

  small : [0.00, 0.15)   medium : [0.15, 0.30)   large : [0.30, 1.00]

Methodology (unchanged from legacy):
  * greedy COCO-style matching, preds sorted by confidence desc, one GT per match;
  * 101-point COCO interpolation; AP = NaN when a (class,bucket) has no GT.
"""
from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

BUCKET_DEFS: dict[str, tuple[float, float]] = {
    "small": (0.00, 0.15),
    "medium": (0.15, 0.30),
    "large": (0.30, 1.00),
}
BUCKET_ORDER = list(BUCKET_DEFS.keys())


class Detection(NamedTuple):
    image_id: int
    class_id: int
    confidence: float
    cx: float
    cy: float
    w: float
    h: float


class GroundTruth(NamedTuple):
    image_id: int
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def _relative_area(w: float, h: float) -> float:
    return w * h


def _assign_bucket(rel_area: float) -> str:
    for name, (lo, hi) in BUCKET_DEFS.items():
        if lo <= rel_area < hi:
            return name
    return "large"  # rel_area == 1.0 edge


def _iou(b1: tuple[float, float, float, float],
         b2: tuple[float, float, float, float]) -> float:
    cx1, cy1, w1, h1 = b1
    cx2, cy2, w2, h2 = b2
    x1_min, x1_max = cx1 - w1 / 2, cx1 + w1 / 2
    y1_min, y1_max = cy1 - h1 / 2, cy1 + h1 / 2
    x2_min, x2_max = cx2 - w2 / 2, cx2 + w2 / 2
    y2_min, y2_max = cy2 - h2 / 2, cy2 + h2 / 2
    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _ap_from_pr(tp_fp_sorted: list[tuple[bool, str]], n_gt: int) -> float:
    """AP@IoU=0.5 via 101-point COCO interpolation. NaN if n_gt == 0."""
    if n_gt == 0:
        return float("nan")
    if not tp_fp_sorted:
        return 0.0
    tp_cumsum = fp_cumsum = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for is_tp, _ in tp_fp_sorted:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_gt)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(p_at_thr) if p_at_thr else 0.0
    return float(ap / 101)


def compute_ap_by_size(detections: list[Detection], ground_truths: list[GroundTruth],
                       class_names: list[str], iou_threshold: float = 0.5) -> dict:
    """AP50 per (class, bucket) and per bucket overall. See module docstring."""
    n_classes = len(class_names)
    gt_index: dict[tuple[int, int], list[GroundTruth]] = {}
    for gt in ground_truths:
        gt_index.setdefault((gt.class_id, gt.image_id), []).append(gt)

    matched_gt: dict[int, dict[int, set[int]]] = {c: {} for c in range(n_classes)}
    tp_fp_lists: dict[tuple[int, str], list[tuple[bool, str]]] = {}

    n_gt_cb: dict[tuple[int, str], int] = {}
    for gt in ground_truths:
        bkt = _assign_bucket(_relative_area(gt.w, gt.h))
        n_gt_cb[(gt.class_id, bkt)] = n_gt_cb.get((gt.class_id, bkt), 0) + 1

    for det in sorted(detections, key=lambda d: -d.confidence):
        cid, iid = det.class_id, det.image_id
        det_bkt = _assign_bucket(_relative_area(det.w, det.h))
        gts_here = gt_index.get((cid, iid), [])
        matched_set = matched_gt.setdefault(cid, {}).setdefault(iid, set())
        best_iou, best_idx = -1.0, -1
        for idx, gt in enumerate(gts_here):
            if idx in matched_set:
                continue
            iou_val = _iou((det.cx, det.cy, det.w, det.h), (gt.cx, gt.cy, gt.w, gt.h))
            if iou_val > best_iou:
                best_iou, best_idx = iou_val, idx
        if best_iou >= iou_threshold:
            matched_set.add(best_idx)
            gm = gts_here[best_idx]
            bkt = _assign_bucket(_relative_area(gm.w, gm.h))
            tp_fp_lists.setdefault((cid, bkt), []).append((True, bkt))
        else:
            tp_fp_lists.setdefault((cid, det_bkt), []).append((False, det_bkt))

    per_class: dict[str, dict[str, dict]] = {}
    bucket_tp_fp: dict[str, list[tuple[bool, str]]] = {b: [] for b in BUCKET_ORDER}
    bucket_n_gt: dict[str, int] = {b: 0 for b in BUCKET_ORDER}
    for cid, cls_name in enumerate(class_names):
        cls_result: dict[str, dict] = {}
        for bkt in BUCKET_ORDER:
            tp_fp = tp_fp_lists.get((cid, bkt), [])
            n_gt = n_gt_cb.get((cid, bkt), 0)
            cls_result[bkt] = {"ap50": _ap_from_pr(tp_fp, n_gt), "n_gt": n_gt}
            bucket_n_gt[bkt] += n_gt
            bucket_tp_fp[bkt].extend(tp_fp)
        per_class[cls_name] = cls_result

    overall = {bkt: {"ap50": _ap_from_pr(bucket_tp_fp[bkt], bucket_n_gt[bkt]),
                     "n_gt": bucket_n_gt[bkt]} for bkt in BUCKET_ORDER}
    return {"buckets": {k: list(v) for k, v in BUCKET_DEFS.items()},
            "overall": overall, "per_class": per_class}


def nan_safe_dumps(obj) -> str:
    """JSON with NaN -> null (JSON has no NaN)."""
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o
    return json.dumps(_clean(obj), indent=2)
