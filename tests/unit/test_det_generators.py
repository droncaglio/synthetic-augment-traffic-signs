"""Unit tests for the Stage-2 arm generators."""
import random

import numpy as np
from PIL import Image

from detection.generators.real_duplicate import RealDuplicate
from detection.generators.bg_photometric import BgPhotometric
from detection.generators.copy_paste import CopyPaste


def _fake_train(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    Image.fromarray(np.full((640, 640, 3), 100, np.uint8)).save(timg / "t0.jpg")
    (tlbl / "t0.txt").write_text("0 0.5 0.5 0.1 0.1")
    return tmp_path


def test_real_duplicate_generates_and_preserves_labels(tmp_path):
    _fake_train(tmp_path)
    sources = [{"class_id": 0, "source_tile": "t0", "bbox": [0.5, 0.5, 0.1, 0.1]}] * 2
    gen = RealDuplicate(tmp_path, seed=1)
    out = tmp_path / "arms" / "real_duplicate"
    manifest = gen.generate(sources, out)

    imgs = sorted((out / "images").glob("*.jpg"))
    assert len(imgs) == 2                                   # 2 sources -> 2 tiles
    # duplicated tile keeps the source label verbatim
    lbl = sorted((out / "labels").glob("*.txt"))[0].read_text().strip()
    assert lbl == "0 0.5 0.5 0.1 0.1"
    # duplicated image is pixel-identical to the source (pure oversampling)
    assert np.array_equal(np.asarray(Image.open(imgs[0])),
                          np.full((640, 640, 3), 100, np.uint8))
    assert manifest["allocated_per_class"] == {0: 2}
    assert manifest["realized_per_class"] == {0: 2}
    assert manifest["n_tiles_written"] == 2


def test_bg_photometric_preserves_sign_perturbs_background(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    # gradient background (so any gamma/contrast perturbation is visible) + flat sign
    grad = np.tile(np.arange(640, dtype=np.uint8), (640, 1))
    arr = np.stack([grad, grad, grad], axis=-1).copy()
    arr[300:340, 300:340] = 200                         # the sign region (distinct, flat)
    Image.fromarray(arr).save(timg / "t0.jpg")
    # sign bbox ~ (320/640, 320/640, 40/640, 40/640)
    (tlbl / "t0.txt").write_text("0 0.5 0.5 0.0625 0.0625")

    loaded = np.asarray(Image.open(timg / "t0.jpg").convert("RGB"))  # JPEG as the gen sees it
    gen = BgPhotometric(tmp_path, seed=3)
    img, labels = gen.make_tile({"class_id": 0, "source_tile": "t0",
                                 "bbox": [0.5, 0.5, 0.0625, 0.0625]}, random.Random(3))
    # sign pixels pixel-exact (vs the loaded tile); background perturbed
    assert np.array_equal(img[300:340, 300:340], loaded[300:340, 300:340])
    assert not np.array_equal(img[:100, :100], loaded[:100, :100])
    assert labels == ["0 0.5 0.5 0.0625 0.0625"]


def test_copy_paste_relocates_sign_into_background(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    src = np.full((640, 640, 3), 50, np.uint8)
    src[300:340, 300:340] = 222                         # sign crop (distinct color)
    Image.fromarray(src).save(timg / "src.jpg")
    (tlbl / "src.txt").write_text("0 0.5 0.5 0.0625 0.0625")
    Image.fromarray(np.full((640, 640, 3), 10, np.uint8)).save(timg / "bg.jpg")
    (tlbl / "bg.txt").write_text("")                    # background tile (no labels)

    gen = CopyPaste(tmp_path, seed=5)
    entry = {"class_id": 0, "source_tile": "src", "bbox": [0.5, 0.5, 0.0625, 0.0625],
             "recipient_tile": "bg", "place": [0.25, 0.25, 0.0625, 0.0625]}
    img, labels = gen.make_tile(entry, random.Random(5))
    # sign now around (0.25,0.25) on the background; a new label added
    assert len(labels) == 1 and labels[0].startswith("0 ")
    assert (img == 222).any()                           # pasted sign present
    assert (img == 10).any()                            # background base present

