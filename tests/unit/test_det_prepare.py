"""Unit tests for detection.prepare (annotations -> normalized records + catalog)."""
import json

import pytest

from detection.prepare import (
    split_from_path, parse_bbox, xyxy_to_yolo, iter_records, build_catalog, prepare,
)


def test_xyxy_to_yolo_known_box():
    cx, cy, bw, bh = xyxy_to_yolo((100, 200, 300, 400), w=2000, h=1000)
    assert (cx, cy, bw, bh) == pytest.approx((0.1, 0.3, 0.1, 0.2))


def test_xyxy_to_yolo_rejects_bad_size():
    with pytest.raises(ValueError):
        xyxy_to_yolo((0, 0, 1, 1), 0, 100)


def test_parse_bbox_both_shapes():
    assert parse_bbox({"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}) == (1.0, 2.0, 3.0, 4.0)
    assert parse_bbox({"x": 1, "y": 2, "w": 3, "h": 4}) == (1.0, 2.0, 4.0, 6.0)
    with pytest.raises(ValueError):
        parse_bbox({"foo": 1})


def test_split_from_path():
    assert split_from_path("train/62627.jpg") == "train"
    assert split_from_path("test/1.jpg") == "test"
    assert split_from_path("other/9.jpg") == "other"
    assert split_from_path("weird/9.jpg") == "unknown"
    assert split_from_path("nopath.jpg") == "unknown"


def _fake_annotations():
    return {
        "types": ["pn", "i5", "w57"],
        "imgs": {
            "A": {"id": "A", "path": "train/A.jpg", "objects": [
                {"category": "pn", "bbox": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}},
                {"category": "pn", "bbox": {"xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30}},
                {"category": "i5", "bbox": {"xmin": 5, "ymin": 5, "xmax": 25, "ymax": 25}},
                {"category": "?", "bbox": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}},  # dropped
            ]},
            "B": {"id": "B", "path": "test/B.jpg", "objects": [
                {"category": "pn", "bbox": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2}},
            ]},
        },
    }


def test_iter_records_parses_and_drops_malformed():
    recs = list(iter_records(_fake_annotations(), panorama_size=2048))
    by_id = {r["id"]: r for r in recs}
    assert by_id["A"]["split_orig"] == "train"
    assert by_id["A"]["width"] == 2048 and by_id["A"]["height"] == 2048
    # "?" category dropped -> A has 3 objects not 4
    assert len(by_id["A"]["objects"]) == 3
    assert by_id["A"]["objects"][0] == {"category": "pn", "xyxy": [0.0, 0.0, 100.0, 100.0]}
    assert len(by_id["B"]["objects"]) == 1


def test_build_catalog_counts():
    recs = list(iter_records(_fake_annotations(), panorama_size=2048))
    cat = build_catalog(recs)
    assert cat["n_panoramas"] == 2
    assert cat["categories"]["pn"] == {"instances": 3, "images": 2}
    assert cat["categories"]["i5"] == {"instances": 1, "images": 1}
    assert "w57" not in cat["categories"]  # declared in types but never instantiated
    # ordered by (-instances, name): pn first
    assert list(cat["categories"])[0] == "pn"
    assert cat["split_orig_counts"] == {"train": 1, "test": 1}


def test_prepare_writes_and_is_idempotent(tmp_path):
    ann = tmp_path / "annotations_all.json"
    ann.write_text(json.dumps(_fake_annotations()))
    out = tmp_path / "prepared"

    cat = prepare(ann, out, panorama_size=2048)
    assert (out / "panoramas.jsonl").exists()
    assert (out / "catalog.json").exists()
    assert cat["n_panoramas"] == 2
    lines = (out / "panoramas.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    # idempotent: second call returns cached catalog without needing the annotations
    ann.unlink()
    cat2 = prepare("/nonexistent/annotations.json", out, panorama_size=2048)
    assert cat2 == cat
