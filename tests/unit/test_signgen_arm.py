"""Unit tests for the signgen_controlnet arm — generate->verify->paste logic, GPU mocked."""
import json
import random

import numpy as np
from PIL import Image


def _arm(tmp_path, monkeypatch):
    prep = tmp_path / "tiles" / ".." / "prepared"
    (tmp_path / "prepared").mkdir(parents=True, exist_ok=True)
    (tmp_path / "prepared" / "subset.json").write_text(
        json.dumps({"classes": [{"id": 7, "name": "pl70"}, {"id": 15, "name": "il100"}]}))
    (tmp_path / "tiles").mkdir(exist_ok=True)
    w = tmp_path / "w.pt"
    w.write_text("x")                                        # dummy; SignClassifier is lazy
    from detection.generators import signgen_arm as M
    monkeypatch.setattr(M, "load_template", lambda name, md: np.zeros((280, 280, 4), np.uint8))
    arm = M.SignGenArm(tmp_path / "tiles", seed=0, verifier_weights=str(w), marks_dir="x")

    def fake_generate(tpl, n, rng):                          # a warped sign filling the center
        warped = np.zeros((512, 512, 4), np.uint8)
        warped[100:400, 100:400, 3] = 255
        return [{"warped": warped, "image": np.full((512, 512, 3), 200, np.uint8)}]
    monkeypatch.setattr(arm.gen, "generate", fake_generate)
    return arm


def test_sign_crop_accepts_when_verifier_agrees(tmp_path, monkeypatch):
    arm = _arm(tmp_path, monkeypatch)
    monkeypatch.setattr(arm.clf, "predict", lambda c: (7, 0.9, np.array([0.9])))
    out = arm._sign_crop({"class_id": 7}, 40, 40, random.Random(0))
    assert out is not None and out.shape == (40, 40, 3)
    assert arm._scan_stats["rejected"] == 0


def test_sign_crop_rejects_and_regenerates_then_none(tmp_path, monkeypatch):
    arm = _arm(tmp_path, monkeypatch)
    monkeypatch.setattr(arm.clf, "predict", lambda c: (3, 0.9, np.array([0.9])))  # wrong class
    out = arm._sign_crop({"class_id": 7}, 40, 40, random.Random(0))
    assert out is None
    assert arm._scan_stats["rejected"] == 1
    assert arm._scan_stats["attempts"] == arm.max_regen       # tried max_regen times


def test_sign_crop_skip_verify_trusts_construction(tmp_path, monkeypatch):
    arm = _arm(tmp_path, monkeypatch)
    arm.skip_verify = {"pl70"}

    def boom(c):
        raise AssertionError("verifier must NOT be called for skip_verify classes")
    monkeypatch.setattr(arm.clf, "predict", boom)
    out = arm._sign_crop({"class_id": 7}, 40, 40, random.Random(0))
    assert out is not None and out.shape == (40, 40, 3)


def test_make_tile_pastes_generated_sign_with_valid_label(tmp_path, monkeypatch):
    arm = _arm(tmp_path, monkeypatch)
    timg, tlbl = arm.tiles_dir / "train" / "images", arm.tiles_dir / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    Image.fromarray(np.full((640, 640, 3), 100, np.uint8)).save(timg / "bg.jpg")
    (tlbl / "bg.txt").write_text("")                          # empty background recipient
    monkeypatch.setattr(arm.clf, "predict", lambda c: (7, 0.9, np.array([0.9])))
    res = arm.make_tile({"class_id": 7, "recipient_tile": "bg", "place": [0.5, 0.5, 0.1, 0.1]},
                        random.Random(0))
    assert res is not None
    img, labels = res
    assert img.shape == (640, 640, 3)
    assert labels[-1].startswith("7 ")                        # pasted sign -> valid class-7 label
