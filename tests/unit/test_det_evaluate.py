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
        {"category": "A", "xyxy": [100, 100, 140, 140]},   # -> small, norm center 0.12
        {"category": "B", "xyxy": [500, 500, 540, 540]},
        {"category": "Z", "xyxy": [0, 0, 10, 10]},          # non-subset -> ignored
    ]},
}


def test_panorama_ground_truths_normalizes_and_filters():
    gts = panorama_ground_truths(RECORDS, ["p0"], SUBSET_IDS, panorama_size=1000)
    assert len(gts) == 2  # Z dropped
    a = next(g for g in gts if g.class_id == 0)
    assert (a.cx, a.cy, a.w) == pytest.approx((0.12, 0.12, 0.04))
    assert a.image_id == 0


def test_panorama_detections_maps_ids():
    dets = panorama_detections(
        {"p0": [{"class_id": 0, "conf": 0.9, "box": (0.12, 0.12, 0.04, 0.04)}]}, ["p0"])
    assert len(dets) == 1 and dets[0].image_id == 0 and dets[0].class_id == 0


def test_evaluate_split_perfect_detection():
    dets = {"p0": [
        {"class_id": 0, "conf": 0.9, "box": (0.12, 0.12, 0.04, 0.04)},
        {"class_id": 1, "conf": 0.8, "box": (0.52, 0.52, 0.04, 0.04)},
    ]}
    out = evaluate_split(RECORDS, ["p0"], SUBSET, dets, panorama_size=1000)
    hl = out["headline"]
    assert hl["ap_small_macro"] == pytest.approx(1.0)
    assert hl["ap_tail"] == pytest.approx(1.0)          # B (tail) perfectly detected
    assert hl["per_class_small"]["A"] == pytest.approx(1.0)


def test_evaluate_split_tail_miss_drops_ap_tail():
    dets = {"p0": [  # only A detected; B (tail) missed
        {"class_id": 0, "conf": 0.9, "box": (0.12, 0.12, 0.04, 0.04)},
    ]}
    out = evaluate_split(RECORDS, ["p0"], SUBSET, dets, panorama_size=1000)
    hl = out["headline"]
    assert hl["per_class_small"]["A"] == pytest.approx(1.0)
    assert hl["per_class_small"]["B"] == pytest.approx(0.0)  # missed -> AP 0
    assert hl["ap_tail"] == pytest.approx(0.0)


def test_derive_headline_handles_nan_absent_class():
    ap_result = {"overall": {"small": {"ap50": 0.5}},
                 "per_class": {"A": {"small": {"ap50": 0.8}},
                               "B": {"small": {"ap50": float("nan")}}}}
    hl = derive_headline_metrics(ap_result, SUBSET)
    assert hl["ap_small_macro"] == pytest.approx(0.8)  # NaN B dropped from macro
