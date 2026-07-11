"""Unit tests for detection.tiling (grid, visibility, label/ignore/drop, paint-out)."""
import numpy as np
import pytest

from detection.tiling import (
    tile_grid, clip_visibility, classify, tile_objects, paint_out,
)


def test_tile_grid_covers_2048():
    tiles = tile_grid(2048, 2048, size=640, overlap=128)
    assert len(tiles) == 16                       # 4x4 offsets {0,512,1024,1408}
    assert tiles[0] == (0, 0, 640, 640)
    assert tiles[-1] == (1408, 1408, 2048, 2048)  # edge-clamped, full coverage
    assert max(t[2] for t in tiles) == 2048


def test_tile_grid_small_image_single_tile():
    assert tile_grid(640, 640, 640, 128) == [(0, 0, 640, 640)]


def test_tile_grid_rejects_bad_overlap():
    with pytest.raises(ValueError):
        tile_grid(2048, 2048, 640, 640)


def test_clip_visibility_inside_partial_outside():
    tile = (0, 0, 640, 640)
    assert clip_visibility((100, 100, 140, 140), tile) == ((100, 100, 140, 140), 1.0)
    clipped, vf = clip_visibility((600, 600, 680, 680), tile)  # 40x40 of 80x80
    assert clipped == (600, 600, 640, 640) and vf == pytest.approx(0.25)
    assert clip_visibility((700, 700, 800, 800), tile) == (None, 0.0)


def test_classify_bands():
    assert classify(0.05, True, 0.6, 0.2) == "drop"
    assert classify(0.9, True, 0.6, 0.2) == "label"
    assert classify(0.9, False, 0.6, 0.2) == "ignore"   # non-subset never labels
    assert classify(0.4, True, 0.6, 0.2) == "ignore"    # subset mid-band -> ignore


def test_tile_objects_label_local_coords():
    tile = (512, 512, 1152, 1152)
    objs = [{"category": "A", "xyxy": [700, 700, 740, 740]}]  # fully inside, subset
    labels, ignores = tile_objects(objs, tile, {"A": 0})
    assert ignores == []
    assert len(labels) == 1
    cid, (cx, cy, bw, bh) = labels[0]
    assert cid == 0
    assert (cx, cy, bw, bh) == pytest.approx((0.325, 0.325, 0.0625, 0.0625))


def test_tile_objects_non_subset_becomes_ignore():
    tile = (512, 512, 1152, 1152)
    objs = [{"category": "Z", "xyxy": [800, 800, 900, 900]}]  # not in subset
    labels, ignores = tile_objects(objs, tile, {"A": 0})
    assert labels == []
    assert ignores == [(288, 288, 388, 388)]


def test_tile_objects_midband_and_sliver():
    tile = (0, 0, 640, 640)
    objs = [
        {"category": "A", "xyxy": [600, 600, 680, 680]},  # vf=0.25 -> ignore
        {"category": "A", "xyxy": [630, 630, 680, 680]},  # vf=0.04 -> drop
    ]
    labels, ignores = tile_objects(objs, tile, {"A": 0})
    assert labels == []
    assert ignores == [(600, 600, 640, 640)]  # only the mid-band one


def test_tile_panorama_train_vs_eval_keep(tmp_path):
    from PIL import Image
    from detection.tiling import tile_panorama

    # 1280x1280 fake panorama -> 3x3=9 grid tiles
    Image.fromarray(np.full((1280, 1280, 3), 128, np.uint8)).save(tmp_path / "p.jpg")
    record = {"id": "p", "objects": [
        {"category": "A", "xyxy": [100, 100, 160, 160]},      # subset -> label, tile (0,0)
        {"category": "Z", "xyxy": [1100, 1100, 1160, 1160]},  # non-subset -> ignore only
    ]}
    subset_ids = {"A": 0}

    train = tile_panorama(tmp_path / "p.jpg", record, subset_ids, tmp_path / "tr",
                          mode="train", neg_keep_fn=None)
    ev = tile_panorama(tmp_path / "p.jpg", record, subset_ids, tmp_path / "ev", mode="eval")
    # train (no negatives) keeps only the label tile; eval keeps every grid tile
    assert len(train) == 1
    assert len(ev) == 9
    assert len(ev) > len(train)


def test_paint_out_makes_region_uniform():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[2:6, 2:6] = np.arange(4 * 4 * 3).reshape(4, 4, 3).astype(np.uint8)
    paint_out(arr, [(2, 2, 6, 6)])
    region = arr[2:6, 2:6]
    # every pixel equals the (single) fill value -> zero spatial variance
    assert (region == region[0, 0]).all()
