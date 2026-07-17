"""Tail error decomposition (pure, testable): classify every tail-class GT as
HIT / CLS / LOC / MISS from a run's reconstructed panorama detections.

Motivation: a global AP-tail number says the tail is hard but NOT *why*. Small-object
detection usually fails by RECALL/RESOLUTION (the sign is too few pixels to detect at
all) rather than by lack of data — and if that is the bottleneck, no amount of synthetic
augmentation can help. This module separates the failure modes so we target the right
lever (or conclude the ceiling is resolution).

Failure taxonomy, per tail ground-truth box (matched against ALL saved detections of a
run, i.e. conf >= the eval floor, panorama-normalized boxes):
  HIT  — matched by a SAME-class detection at IoU >= iou_hit (greedy COCO assignment).
  CLS  — not hit; some detection covers it (IoU >= iou_hit) but of a DIFFERENT class
         (localized, misclassified) -> a tail confusion. Records the confused-to class.
  LOC  — not hit/cls; a detection is NEAR it (iou_near <= IoU < iou_hit) -> poor box.
  MISS — nothing overlaps even at iou_near -> pure recall/resolution failure.

Reuses detection.ap_by_size._iou (same IoU used by the AP metric) so the decomposition
is consistent with the reported number.
"""
from __future__ import annotations

from detection.ap_by_size import _iou

Box = tuple  # (cx, cy, w, h), normalized to the panorama

CATEGORIES = ("HIT", "CLS", "LOC", "MISS")


def greedy_same_class_hits(gts: list[dict], dets: list[dict], iou_hit: float = 0.5) -> set[int]:
    """Indices (into `gts`) matched by a same-class detection at IoU>=iou_hit.

    COCO-style greedy: detections in confidence-desc order, one GT per detection, best
    IoU among unmatched same-class GTs. Mirrors compute_ap_by_size's TP assignment.
    """
    matched: set[int] = set()
    for d in sorted(dets, key=lambda x: -x.get("conf", 0.0)):
        best_iou, best_idx = iou_hit, -1  # only accept IoU >= iou_hit
        for i, gt in enumerate(gts):
            if i in matched or gt["class_id"] != d["class_id"]:
                continue
            iou = _iou(tuple(gt["box"]), tuple(d["box"]))
            if iou >= best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0:
            matched.add(best_idx)
    return matched


def classify_gt(gt: dict, dets: list[dict], is_hit: bool,
                iou_hit: float = 0.5, iou_near: float = 0.3) -> tuple[str, int | None]:
    """(category, confused_to_class_or_None) for one GT given the run's detections."""
    if is_hit:
        return ("HIT", None)
    gbox, gcls = tuple(gt["box"]), gt["class_id"]
    best_same = 0.0
    best_any, best_any_cls = 0.0, None
    for d in dets:
        iou = _iou(gbox, tuple(d["box"]))
        if d["class_id"] == gcls and iou > best_same:
            best_same = iou
        if iou > best_any:
            best_any, best_any_cls = iou, d["class_id"]
    if best_same >= iou_hit:      # same-class covers it but greedy gave the det to another GT
        return ("HIT", None)      # -> contended hit, not a real failure
    if best_any >= iou_hit:       # localized by a different class = misclassification
        return ("CLS", best_any_cls)
    if best_any >= iou_near:       # something near but poorly localized
        return ("LOC", best_any_cls)
    return ("MISS", None)         # nothing even near = recall/resolution failure


def size_band_px(box: Box, panorama_size: int = 2048) -> float:
    """Max side of the box in panorama pixels (the detector-scale sign size)."""
    _cx, _cy, w, h = box
    return max(w, h) * panorama_size


BANDS = [(0, 16), (16, 24), (24, 32), (32, 48), (48, 96), (96, 10 ** 9)]
BAND_LABELS = ["<16", "16-24", "24-32", "32-48", "48-96", ">=96"]


def band_of(px: float) -> str:
    for (lo, hi), lab in zip(BANDS, BAND_LABELS):
        if lo <= px < hi:
            return lab
    return BAND_LABELS[-1]


def tail_fp_precision(gts_by_pid: dict, dets_by_pid: dict, tail_ids: set[int],
                      conf: float, iou_hit: float = 0.5) -> dict:
    """Precision side: tail-class detections >= conf split into TP / FP.

    A tail detection is TP if it matches a same-class tail GT at IoU>=iou_hit (greedy),
    else FP (the model fired a tail class where there is no such sign -> hallucination /
    fine-grained confusion). Complements the recall-side decomposition: on a recall-
    saturated tail, AP is capped by these FPs, and adding foreground instances tends to
    INCREASE them. Returns {tp, fp, precision}.
    """
    tp = fp = 0
    for pid, gl in gts_by_pid.items():
        dl = [d for d in dets_by_pid.get(pid, []) if d.get("conf", 0.0) >= conf
              and d["class_id"] in tail_ids]
        used: set[int] = set()
        for d in sorted(dl, key=lambda x: -x.get("conf", 0.0)):
            best, bi = iou_hit, -1
            for i, g in enumerate(gl):
                if i in used or g["class_id"] != d["class_id"]:
                    continue
                iou = _iou(tuple(g["box"]), tuple(d["box"]))
                if iou >= best:
                    best, bi = iou, i
            if bi >= 0:
                used.add(bi)
                tp += 1
            else:
                fp += 1
    return {"tp": tp, "fp": fp, "precision": tp / (tp + fp) if (tp + fp) else 0.0}


def classify_tail_fps(gts_by_pid: dict, dets_by_pid: dict, tail_ids: set[int],
                      conf: float, iou_hit: float = 0.5, iou_near: float = 0.3) -> dict:
    """Where do tail-class false positives land? Decides the hard-negative design.

    For each tail detection >= conf that is NOT a same-class TP, check overlap with the
    NON-tail real GTs of that panorama:
      on_sign — overlaps a non-tail GT (IoU>=iou_hit): fine-grained confusion (tail fired
                on a lookalike sign) -> hard negative = that confuser sign (labeled true).
      on_bg   — nothing near (IoU<iou_near with any GT): background false fire -> hard
                negative = that background patch (unlabeled).
      ambiguous — near a GT but < iou_hit.
    Returns counts + a Counter of (tail_name is not resolved here; caller maps ids) of
    confuser classes it fires ON (by class_id).
    """
    on_sign = on_bg = ambiguous = 0
    fires_on: dict[int, int] = {}
    for pid, gl in gts_by_pid.items():
        dl = [d for d in dets_by_pid.get(pid, []) if d.get("conf", 0.0) >= conf
              and d["class_id"] in tail_ids]
        # same-class tail TP set (so we only look at the FPs)
        used: set[int] = set()
        for d in sorted(dl, key=lambda x: -x.get("conf", 0.0)):
            best, bi = iou_hit, -1
            for i, g in enumerate(gl):
                if i in used or g["class_id"] != d["class_id"]:
                    continue
                iou = _iou(tuple(g["box"]), tuple(d["box"]))
                if iou >= best:
                    best, bi = iou, i
            if bi >= 0:
                used.add(bi)
                continue  # TP, not a FP
            # FP: where does it land relative to NON-tail GTs?
            best_iou, best_cls = 0.0, None
            for g in gl:
                if g["class_id"] in tail_ids:
                    continue
                iou = _iou(tuple(g["box"]), tuple(d["box"]))
                if iou > best_iou:
                    best_iou, best_cls = iou, g["class_id"]
            if best_iou >= iou_hit:
                on_sign += 1
                fires_on[best_cls] = fires_on.get(best_cls, 0) + 1
            elif best_iou < iou_near:
                on_bg += 1
            else:
                ambiguous += 1
    return {"on_sign": on_sign, "on_bg": on_bg, "ambiguous": ambiguous, "fires_on": fires_on}


def decompose_run(gts_by_pid: dict, dets_by_pid: dict, tail_ids: set[int],
                  iou_hit: float = 0.5, iou_near: float = 0.3,
                  panorama_size: int = 2048) -> list[dict]:
    """One record per tail GT: {class_id, size_px, band, category, confused_to}.

    gts_by_pid: {pid: [{class_id, box}]} (all subset classes; needed so contended hits
                and cross-class confusions are found).
    dets_by_pid: {pid: [{class_id, conf, box}]}.
    """
    out: list[dict] = []
    for pid, gts in gts_by_pid.items():
        dets = dets_by_pid.get(pid, [])
        hits = greedy_same_class_hits(gts, dets, iou_hit)
        for i, gt in enumerate(gts):
            if gt["class_id"] not in tail_ids:
                continue
            cat, conf_to = classify_gt(gt, dets, i in hits, iou_hit, iou_near)
            px = size_band_px(tuple(gt["box"]), panorama_size)
            out.append({"class_id": gt["class_id"], "size_px": round(px, 1),
                        "band": band_of(px), "category": cat, "confused_to": conf_to})
    return out
