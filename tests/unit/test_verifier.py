"""Unit tests for the sign class-verifier logic (no torch/torchvision, no training)."""
import numpy as np
from PIL import Image

from detection.verifier import SignClassifier, build_crop_index, load_crop


def _fake_train(tmp_path):
    timg = tmp_path / "train" / "images"
    tlbl = tmp_path / "train" / "labels"
    timg.mkdir(parents=True)
    tlbl.mkdir(parents=True)
    arr = np.zeros((640, 640, 3), np.uint8)
    arr[300:340, 300:340] = 255                       # a sign patch
    Image.fromarray(arr).save(timg / "t0.jpg")
    (tlbl / "t0.txt").write_text("7 0.5 0.5 0.0625 0.0625")   # single-sign, class 7
    (tlbl / "bg.txt").write_text("")                          # empty (ignored)
    Image.fromarray(arr).save(timg / "bg.jpg")
    return tmp_path


def test_build_crop_index_single_sign_only(tmp_path):
    _fake_train(tmp_path)
    idx = build_crop_index(tmp_path / "train" / "labels")
    assert idx == [("t0", 7, [0.5, 0.5, 0.0625, 0.0625])]      # bg tile excluded


def test_load_crop_cuts_the_sign(tmp_path):
    _fake_train(tmp_path)
    crop = load_crop(tmp_path / "train" / "images", "t0", [0.5, 0.5, 0.0625, 0.0625])
    assert crop.shape == (40, 40, 3) and crop.mean() > 240      # ~white (JPEG-lossy)


def test_valid_rate_top1_accept_and_conf(monkeypatch):
    clf = SignClassifier(class_ids=[3, 7, 9])                  # index 0->3, 1->7, 2->9
    # fake predict: crop value encodes (pred_class_id, conf, prob_of_intended)
    def fake_predict(crop):
        cid, conf, p_int = crop["cid"], crop["conf"], crop["p_int"]
        probs = np.zeros(3)
        probs[clf.class_ids.index(cid)] = conf
        probs[clf.class_ids.index(7)] = p_int                 # intended=7 in this test
        return cid, conf, probs
    monkeypatch.setattr(clf, "predict", fake_predict)

    crops = [
        {"cid": 7, "conf": 0.9, "p_int": 0.9},   # correct, high conf -> accept
        {"cid": 7, "conf": 0.4, "p_int": 0.4},   # correct class but below thr -> not accepted
        {"cid": 3, "conf": 0.8, "p_int": 0.1},   # wrong class -> not top1, not accepted
    ]
    r = clf.valid_rate(crops, intended_class_id=7, conf_thr=0.5)
    assert r["n"] == 3
    assert abs(r["top1_acc"] - 2 / 3) < 1e-9                   # 2 of 3 argmax==7
    assert abs(r["accept_rate"] - 1 / 3) < 1e-9               # only the high-conf correct one
    assert abs(r["mean_conf"] - (0.9 + 0.4 + 0.1) / 3) < 1e-9  # prob assigned to intended=7


def test_valid_rate_empty():
    clf = SignClassifier(class_ids=[1, 2])
    assert clf.valid_rate([], 1)["n"] == 0
