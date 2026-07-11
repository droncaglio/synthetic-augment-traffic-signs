"""Unit tests for the Stage-2 arm generators (base orchestration + real_duplicate)."""
import numpy as np
from PIL import Image

from detection.generators.real_duplicate import RealDuplicate


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
