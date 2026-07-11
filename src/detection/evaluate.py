"""Panorama-level evaluation: build GTs/Detections and derive headline metrics.

Detections come from detection.reconstruct (per-tile preds mapped to the panorama +
global NMS), so a sign near a seam is counted once. Ground truth is the subset-class
annotations of each panorama. We feed both into ap_by_size.compute_ap_by_size and
report AP@small (near-all signs are 'small' on 2048 panoramas) and AP-tail
(macro-average of per-class small AP over the tail classes).
"""
from __future__ import annotations

from detection.ap_by_size import Detection, GroundTruth, compute_ap_by_size
from detection.prepare import xyxy_to_yolo
from detection.tiling import tile_grid, clip_visibility


def panorama_ground_truths(records_by_id: dict[str, dict], split_ids: list[str],
                           subset_ids: dict[str, int], panorama_size: int,
                           tile_size: int = 640, tile_overlap: int = 128,
                           keep_thresh: float = 0.6) -> tuple[list[GroundTruth], int]:
    """Subset-class GTs (normalized), EXCLUDING those never labelable in any tile.

    Symmetric with the paint-out ignore: a GT that never reaches keep_thresh in any
    tile is painted out everywhere (model can't detect it), so it must also be removed
    from the denominator — otherwise it is a guaranteed structural FN. With overlap
    >= max sign size this never fires for TT100K, but we enforce the symmetry anyway.
    Returns (gts, n_excluded).
    """
    grid = tile_grid(panorama_size, panorama_size, tile_size, tile_overlap)
    gts: list[GroundTruth] = []
    n_excluded = 0
    for img_id, pid in enumerate(split_ids):
        for o in records_by_id[pid]["objects"]:
            cid = subset_ids.get(o["category"])
            if cid is None:
                continue
            xyxy = tuple(o["xyxy"])
            if not any(clip_visibility(xyxy, t)[1] >= keep_thresh for t in grid):
                n_excluded += 1
                continue
            cx, cy, w, h = xyxy_to_yolo(xyxy, panorama_size, panorama_size)
            gts.append(GroundTruth(img_id, cid, cx, cy, w, h))
    return gts, n_excluded


def panorama_detections(dets_by_panorama: dict[str, list[dict]],
                        split_ids: list[str]) -> list[Detection]:
    """Reconstructed detections ({class_id, conf, box=(cx,cy,w,h)}) -> Detection list."""
    id_of = {pid: i for i, pid in enumerate(split_ids)}
    out: list[Detection] = []
    for pid, dets in dets_by_panorama.items():
        img_id = id_of.get(pid)
        if img_id is None:
            continue
        for d in dets:
            cx, cy, w, h = d["box"]
            out.append(Detection(img_id, d["class_id"], d["conf"], cx, cy, w, h))
    return out


def derive_headline_metrics(ap_result: dict, subset: dict) -> dict:
    """AP@small (overall + macro) and AP-tail (macro over tail classes)."""
    per_class = ap_result["per_class"]

    def small_ap(name: str):
        return per_class.get(name, {}).get("small", {}).get("ap50")

    def macro(names: list[str]):
        vals = [small_ap(n) for n in names]
        vals = [v for v in vals if v is not None and v == v]  # drop None/NaN
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "ap_small_overall": ap_result["overall"]["small"]["ap50"],
        "ap_small_macro": macro(subset["names"]),
        "ap_tail": macro(subset["by_tier"]["tail"]),
        "per_class_small": {n: small_ap(n) for n in subset["names"]},
    }


def evaluate_split(records_by_id: dict[str, dict], split_ids: list[str], subset: dict,
                   dets_by_panorama: dict[str, list[dict]], panorama_size: int = 2048
                   ) -> dict:
    """Full panorama-level evaluation for one split. Returns ap_result + headline."""
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    class_names = subset["names"]  # indexed by class_id (0..K-1)
    gts, n_excluded = panorama_ground_truths(records_by_id, split_ids, subset_ids, panorama_size)
    dets = panorama_detections(dets_by_panorama, split_ids)
    ap_result = compute_ap_by_size(dets, gts, class_names, panorama_size=panorama_size)
    return {"ap_by_size": ap_result, "headline": derive_headline_metrics(ap_result, subset),
            "n_gt_excluded_unlabelable": n_excluded}
