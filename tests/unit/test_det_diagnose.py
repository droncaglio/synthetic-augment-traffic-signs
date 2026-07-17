"""Unit tests for the tail error decomposition taxonomy (pure, no IO)."""
import pytest

from detection.diagnose import (
    greedy_same_class_hits, classify_gt, decompose_run, tail_fp_precision,
    classify_tail_fps, size_band_px, band_of)


def _gt(cid, cx, cy, w=0.02, h=0.02):
    return {"class_id": cid, "box": (cx, cy, w, h)}


def _det(cid, cx, cy, w=0.02, h=0.02, conf=0.9):
    return {"class_id": cid, "conf": conf, "box": (cx, cy, w, h)}


def test_hit_when_same_class_overlaps():
    gts = [_gt(14, 0.5, 0.5)]
    dets = [_det(14, 0.5, 0.5)]                     # exact overlap, same class
    hits = greedy_same_class_hits(gts, dets)
    assert hits == {0}
    assert classify_gt(gts[0], dets, is_hit=True) == ("HIT", None)


def test_cls_when_other_class_covers():
    gt = _gt(14, 0.5, 0.5)
    dets = [_det(3, 0.5, 0.5)]                      # same box, WRONG class -> misclassified
    assert greedy_same_class_hits([gt], dets) == set()
    assert classify_gt(gt, dets, is_hit=False) == ("CLS", 3)


def test_loc_when_near_but_low_iou():
    gt = _gt(14, 0.5, 0.5, 0.04, 0.04)
    dets = [_det(14, 0.517, 0.5, 0.04, 0.04)]       # shifted -> IoU in [0.3,0.5)
    from detection.ap_by_size import _iou
    iou = _iou(gt["box"], dets[0]["box"])
    assert 0.3 <= iou < 0.5, iou
    assert classify_gt(gt, dets, is_hit=False)[0] == "LOC"


def test_miss_when_nothing_near():
    gt = _gt(14, 0.5, 0.5)
    dets = [_det(14, 0.1, 0.1)]                     # far away
    assert classify_gt(gt, dets, is_hit=False) == ("MISS", None)
    assert classify_gt(gt, [], is_hit=False) == ("MISS", None)


def test_contended_hit_not_counted_as_cls():
    # two GTs, one detection covering both regions' overlap; greedy assigns to one,
    # the other must fall back to HIT (contended), not CLS, since a same-class det covers it.
    gts = [_gt(14, 0.5, 0.5), _gt(14, 0.505, 0.5)]
    dets = [_det(14, 0.5, 0.5), _det(14, 0.505, 0.5)]
    hits = greedy_same_class_hits(gts, dets)
    assert hits == {0, 1}


def test_size_band():
    assert size_band_px((0.5, 0.5, 0.01, 0.008), 2048) == pytest.approx(20.48)
    assert band_of(20.48) == "16-24"
    assert band_of(10) == "<16"
    assert band_of(200) == ">=96"


def test_tail_fp_precision_counts_tp_and_fp():
    gts_by_pid = {"p0": [_gt(14, 0.5, 0.5)]}          # one tail GT (class 14)
    dets_by_pid = {"p0": [
        _det(14, 0.5, 0.5, conf=0.9),                 # TP: same-class, overlaps
        _det(15, 0.2, 0.2, conf=0.8),                 # FP: tail class 15, no GT there
        _det(14, 0.8, 0.8, conf=0.05),                # below conf floor -> ignored at 0.1
    ]}
    r = tail_fp_precision(gts_by_pid, dets_by_pid, {14, 15, 16}, conf=0.1)
    assert r["tp"] == 1 and r["fp"] == 1
    assert r["precision"] == pytest.approx(0.5)
    # raising conf drops the FP's... no: both above 0.1; drop the low-conf extra only
    r2 = tail_fp_precision(gts_by_pid, dets_by_pid, {14, 15, 16}, conf=0.85)
    assert r2["tp"] == 1 and r2["fp"] == 0            # class-15 FP (0.8) now filtered out


def test_classify_tail_fps_bg_vs_sign():
    # tail GT (14) at center; a non-tail sign (3=head) at a different spot.
    gts_by_pid = {"p0": [_gt(14, 0.5, 0.5), _gt(3, 0.2, 0.2)]}
    dets_by_pid = {"p0": [
        _det(14, 0.5, 0.5, conf=0.9),      # TP tail (matches GT 14) -> excluded from FP
        _det(15, 0.2, 0.2, conf=0.9),      # tail FP ON the head sign (confuser) -> on_sign
        _det(16, 0.8, 0.8, conf=0.9),      # tail FP on empty background -> on_bg
    ]}
    r = classify_tail_fps(gts_by_pid, dets_by_pid, {14, 15, 16}, conf=0.5)
    assert r["on_sign"] == 1 and r["on_bg"] == 1 and r["ambiguous"] == 0
    assert r["fires_on"] == {3: 1}         # the confuser FP fired where class 3 is


def test_decompose_run_only_tail_and_sums():
    gts_by_pid = {"p0": [_gt(14, 0.5, 0.5), _gt(0, 0.2, 0.2)]}   # one tail, one head
    dets_by_pid = {"p0": [_det(14, 0.5, 0.5)]}
    recs = decompose_run(gts_by_pid, dets_by_pid, tail_ids={14, 15, 16})
    assert len(recs) == 1                          # head GT (id 0) excluded
    assert recs[0]["category"] == "HIT" and recs[0]["class_id"] == 14
