"""Unit tests for detection.evaluate (panorama-level GT/Detection + headline metrics)."""
import pytest

from detection.evaluate import (
    panorama_ground_truths, panorama_detections, derive_headline_metrics, evaluate_split,
)

SUBSET = {
    "names": ["A", "B"],
    "classes": [{"name": "A", "id": 0, "tier": "head"},
                {"name": "B", "id": 1, "tier": "tail"}],
    "by_tier": {"head": ["A"], "mid": [], "tail": ["B"]},
}
SUBSET_IDS = {"A": 0, "B": 1}

RECORDS = {
    "p0": {"objects": [
        {"category": "A", "xyxy": [100, 100, 120, 120]},   # 20px -> COCO small
        {"category": "B", "xyxy": [500, 500, 520, 520]},
        {"category": "Z", "xyxy": [0, 0, 10, 10]},          # non-subset -> ignored
    ]},
}


def test_panorama_ground_truths_normalizes_and_filters():
    gts, n_excluded = panorama_ground_truths(RECORDS, ["p0"], SUBSET_IDS, panorama_size=1000)
    assert len(gts) == 2 and n_excluded == 0  # Z dropped; both labelable
    a = next(g for g in gts if g.class_id == 0)
    assert (a.cx, a.cy, a.w) == pytest.approx((0.11, 0.11, 0.02))
    assert a.image_id == 0


def test_panorama_detections_maps_ids():
    dets = panorama_detections(
        {"p0": [{"class_id": 0, "conf": 0.9, "box": (0.12, 0.12, 0.04, 0.04)}]}, ["p0"])
    assert len(dets) == 1 and dets[0].image_id == 0 and dets[0].class_id == 0


def test_evaluate_split_perfect_detection():
    dets = {"p0": [
        {"class_id": 0, "conf": 0.9, "box": (0.11, 0.11, 0.02, 0.02)},
        {"class_id": 1, "conf": 0.8, "box": (0.51, 0.51, 0.02, 0.02)},
    ]}
    out = evaluate_split(RECORDS, ["p0"], SUBSET, dets, panorama_size=1000)
    hl = out["headline"]
    assert hl["ap_small_macro"] == pytest.approx(1.0)
    assert hl["ap_tail"] == pytest.approx(1.0)          # B (tail) perfectly detected
    assert hl["per_class_small"]["A"] == pytest.approx(1.0)


def test_evaluate_split_tail_miss_drops_ap_tail():
    dets = {"p0": [  # only A detected; B (tail) missed
        {"class_id": 0, "conf": 0.9, "box": (0.11, 0.11, 0.02, 0.02)},
    ]}
    out = evaluate_split(RECORDS, ["p0"], SUBSET, dets, panorama_size=1000)
    hl = out["headline"]
    assert hl["per_class_small"]["A"] == pytest.approx(1.0)
    assert hl["per_class_small"]["B"] == pytest.approx(0.0)  # missed -> AP 0
    assert hl["ap_tail"] == pytest.approx(0.0)


def test_absolute_coco_buckets_actually_stratify():
    # small (<32px), medium (32-96px), large (>96px) on a 2048 panorama
    from detection.ap_by_size import compute_ap_by_size, Detection, GroundTruth
    names = ["c"]

    def area_norm(px):  # side px -> normalized w (on 2048)
        return px / 2048

    for px, bucket in [(20, "small"), (60, "medium"), (150, "large")]:
        w = area_norm(px)
        gt = [GroundTruth(0, 0, 0.5, 0.5, w, w)]
        det = [Detection(0, 0, 0.9, 0.5, 0.5, w, w)]
        res = compute_ap_by_size(det, gt, names, panorama_size=2048)
        assert res["overall"][bucket]["n_gt"] == 1, f"{px}px should be {bucket}"
        assert res["overall"][bucket]["ap50"] == pytest.approx(1.0)


def test_macro_metric_matches_headline():
    from detection.stats import make_macro_metric
    ap_result = {"overall": {"small": {"ap50": 0.5}},
                 "per_class": {"A": {"small": {"ap50": 0.8}},
                               "B": {"small": {"ap50": 0.2}}}}
    tail_metric = make_macro_metric(SUBSET, tier="tail")   # tail = ["B"]
    assert tail_metric(ap_result) == pytest.approx(0.2)    # matches derive_headline ap_tail
    all_metric = make_macro_metric(SUBSET)                 # both -> mean(0.8,0.2)
    assert all_metric(ap_result) == pytest.approx(0.5)


def test_derive_headline_handles_nan_absent_class():
    ap_result = {"overall": {"small": {"ap50": 0.5}},
                 "per_class": {"A": {"small": {"ap50": 0.8}},
                               "B": {"small": {"ap50": float("nan")}}}}
    hl = derive_headline_metrics(ap_result, SUBSET)
    assert hl["ap_small_macro"] == pytest.approx(0.8)  # NaN B dropped from macro
