"""Unit tests for SAM-mask filtering, cache load, and copy_paste_mask SAM/fallback wiring.
The SAM model inference itself is GPU/heavy and NOT unit-tested (mirrors diffusion_bg)."""
import json
import random

import numpy as np
from PIL import Image

from detection.generators.sam_masks import filter_mask, load_cached_mask, mask_path
from detection.generators.copy_paste_mask import CopyPasteMask


# ---- filter_mask (sign-tuned thresholds) ------------------------------------
def test_filter_mask_accepts_full_convex():
    m = np.zeros((40, 40), np.uint8)
    m[5:35, 5:35] = 1                      # convex block, area_ratio=0.5625, solidity~1
    _, meta = filter_mask(m)
    assert meta["status"] == "ok"


def test_filter_mask_rejects_small_fragment():
    m = np.zeros((40, 40), np.uint8)
    m[18:23, 18:23] = 1                    # area_ratio=25/1600=0.016 < 0.20
    _, meta = filter_mask(m)
    assert meta["status"] == "rejected" and "area_small" in meta["reason"]


def test_filter_mask_rejects_low_solidity():
    m = np.zeros((40, 40), np.uint8)
    m[15:25, :] = 1; m[:, 15:25] = 1       # cruz: area_ratio~0.44 (passa área), solidez ~0.6 < 0.80
    _, meta = filter_mask(m)
    assert meta["status"] == "rejected" and "solidity" in meta["reason"]


def test_filter_mask_ok_when_touching_all_borders():
    m = np.ones((40, 40), np.uint8)        # touches every border -> ENIAC would reject; we don't
    _, meta = filter_mask(m)
    assert meta["status"] == "ok"


# ---- cache load -------------------------------------------------------------
def test_load_cached_mask_missing_returns_none(tmp_path):
    assert load_cached_mask(tmp_path, "nope") is None
    assert load_cached_mask(tmp_path, "") is None


def test_load_cached_mask_reads_png(tmp_path):
    m = np.zeros((20, 20), np.uint8); m[5:15, 5:15] = 255
    Image.fromarray(m).save(mask_path(tmp_path, "t0"))
    got = load_cached_mask(tmp_path, "t0")
    assert got is not None and got.dtype == np.uint8 and got[10, 10] == 1 and got[0, 0] == 0


# ---- copy_paste_mask: SAM cache drives the paste; fallback when absent ------
def _scene(tmp_path, sam_mask=None):
    tiles = tmp_path / "tiles"
    (tiles / "train" / "images").mkdir(parents=True)
    (tiles / "train" / "labels").mkdir(parents=True)
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "subset.json").write_text(json.dumps(
        {"classes": [{"id": 7, "name": "w57"}]}))
    src = np.full((640, 640, 3), 50, np.uint8); src[280:360, 280:360] = 222
    Image.fromarray(src).save(tiles / "train" / "images" / "src.jpg")
    (tiles / "train" / "labels" / "src.txt").write_text("7 0.5 0.5 0.125 0.125")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(tiles / "train" / "images" / "bg.jpg")
    (tiles / "train" / "labels" / "bg.txt").write_text("")
    if sam_mask is not None:
        (tmp_path / "masks" / "train").mkdir(parents=True)
        Image.fromarray(sam_mask).save(mask_path(tmp_path / "masks" / "train", "src"))
    return tiles


def test_copy_paste_mask_uses_cached_sam_mask(tmp_path):
    band = np.zeros((80, 80), np.uint8); band[30:50, :] = 255      # só faixa horizontal do meio
    tiles = _scene(tmp_path, sam_mask=band)
    gen = CopyPasteMask(tiles, seed=5, mask_source="sam")
    entry = {"class_id": 7, "source_tile": "src", "bbox": [0.5, 0.5, 0.125, 0.125],
             "recipient_tile": "bg", "place": [0.5, 0.5, 0.125, 0.125]}
    img, _ = gen.make_tile(entry, random.Random(5))
    tw = th = 80; px1 = py1 = int(round(0.5 * 640 - 40))
    assert img[py1 + 40, px1 + 40].max() > 150     # dentro da faixa SAM -> placa
    assert img[py1 + 5, px1 + 40].max() < 60       # fora da faixa -> fundo (a máscara SAM mandou)


def test_copy_paste_mask_falls_back_to_geometric_without_cache(tmp_path):
    tiles = _scene(tmp_path, sam_mask=None)         # sem cache -> geométrico (triângulo p/ w57)
    gen = CopyPasteMask(tiles, seed=5, mask_source="sam")
    entry = {"class_id": 7, "source_tile": "src", "bbox": [0.5, 0.5, 0.125, 0.125],
             "recipient_tile": "bg", "place": [0.5, 0.5, 0.125, 0.125]}
    img, _ = gen.make_tile(entry, random.Random(5))
    tw = th = 80; px1 = py1 = int(round(0.5 * 640 - 40))
    # triângulo ápice-pra-cima: canto superior = fundo; base = placa
    assert img[py1 + 3, px1 + 3].max() < 60
    assert img[py1 + th - 4, px1 + tw // 2].max() > 150
