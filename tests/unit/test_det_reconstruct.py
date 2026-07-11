"""Unit tests for detection.reconstruct (tile->panorama mapping, global NMS, ignore)."""
import pytest

from detection.reconstruct import (
    iou_cxcywh, map_det_to_panorama, nms_per_class, reconstruct_panorama,
)


def test_iou_identical_and_disjoint():
    a = (0.5, 0.5, 0.2, 0.2)
    assert iou_cxcywh(a, a) == pytest.approx(1.0)
    assert iou_cxcywh(a, (0.9, 0.9, 0.05, 0.05)) == 0.0


def test_map_det_to_panorama():
    entry = {"x_off": 1024, "y_off": 512, "size": 640}
    # tile-local center (0.11875, 0.1375) -> panorama px (1100, 600) -> /2048
    box = map_det_to_panorama((0.11875, 0.1375, 0.0625, 0.0625), entry, 2048)
    assert box[0] == pytest.approx(1100 / 2048)
    assert box[1] == pytest.approx(600 / 2048)
    assert box[2] == pytest.approx(40 / 2048)


def test_nms_collapses_duplicates():
    b = (0.5, 0.5, 0.1, 0.1)
    dets = [{"class_id": 0, "conf": 0.9, "box": b},
            {"class_id": 0, "conf": 0.6, "box": (0.505, 0.5, 0.1, 0.1)}]  # ~same
    kept = nms_per_class(dets, iou_thresh=0.5)
    assert len(kept) == 1 and kept[0]["conf"] == 0.9


def test_nms_keeps_different_classes_and_far_boxes():
    dets = [{"class_id": 0, "conf": 0.9, "box": (0.5, 0.5, 0.1, 0.1)},
            {"class_id": 1, "conf": 0.8, "box": (0.5, 0.5, 0.1, 0.1)},   # same box, other class
            {"class_id": 0, "conf": 0.7, "box": (0.9, 0.9, 0.1, 0.1)}]   # far
    assert len(nms_per_class(dets, 0.5)) == 3


def test_reconstruct_dedupes_sign_in_two_overlapping_tiles():
    # same panorama sign (center 1100,600, 40x40) seen in tile A and tile B
    tileA = {"entry": {"x_off": 1024, "y_off": 512, "size": 640},
             "dets": [{"class_id": 17, "conf": 0.9,
                       "box": (0.11875, 0.1375, 0.0625, 0.0625)}]}
    tileB = {"entry": {"x_off": 512, "y_off": 512, "size": 640},
             "dets": [{"class_id": 17, "conf": 0.7,
                       "box": (0.91875, 0.1375, 0.0625, 0.0625)}]}
    kept = reconstruct_panorama([tileA, tileB], panorama_size=2048, nms_iou=0.5)
    assert len(kept) == 1                       # <- double-count removed
    assert kept[0]["conf"] == 0.9
    assert kept[0]["box"][0] == pytest.approx(1100 / 2048)


def test_reconstruct_drops_detection_in_ignore_region():
    # ignore box covers the sign's panorama location
    tile = {"entry": {"x_off": 0, "y_off": 0, "size": 640},
            "dets": [{"class_id": 3, "conf": 0.8, "box": (0.5, 0.5, 0.05, 0.05)}],
            "ignores": [(300, 300, 340, 340)]}  # tile px around panorama px 320 = 0.5*640
    kept = reconstruct_panorama([tile], panorama_size=640, nms_iou=0.5)
    assert kept == []
