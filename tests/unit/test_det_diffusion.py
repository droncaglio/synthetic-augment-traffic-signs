"""Unit tests for the pure (CPU) logic of diffusion_bg — inverted mask + hallucination
scan. The FluxFill inference itself is GPU/model and validated on the workstation."""
import random

import numpy as np
from PIL import Image

from detection.generators.diffusion_bg import DiffusionBg


class _FakePipe:
    """Returns a solid-color PIL image (like FluxFill) — np.asarray of it is read-only,
    which is exactly the composite path we must survive."""
    def __init__(self, fill): self.fill = fill

    def __call__(self, **k):
        h, w = k["height"], k["width"]
        arr = np.full((h, w, 3), self.fill, np.uint8)
        class _R: images = [Image.fromarray(arr)]
        return _R()


class _FakeBoxes:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def __len__(self):
        return len(self._arr)

    @property
    def xywhn(self):
        class _T:
            def __init__(s, a): s.a = a
            def cpu(s): return s
            def numpy(s): return s.a
        return _T(self._arr)


class _FakeResult:
    def __init__(self, arr): self.boxes = _FakeBoxes(arr)


class _FakeScanner:
    def __init__(self, dets): self.dets = dets

    def predict(self, *a, **k):
        return [_FakeResult(self.dets)]


def test_inverted_mask_keeps_sign_regenerates_background():
    gen = DiffusionBg("unused", seed=0)
    mask = np.asarray(gen._inverted_mask(["0 0.5 0.5 0.1 0.1"], 100, 100))
    # sign bbox px = (45,45,55,55) -> 0 (keep); background -> 255 (regenerate)
    assert (mask[45:55, 45:55] == 0).all()
    assert (mask[:40, :40] == 255).all()


def test_hallucinated_true_when_detection_off_sign():
    gen = DiffusionBg("unused", seed=0)
    gen._scanner = _FakeScanner([[0.1, 0.1, 0.05, 0.05]])   # detection far from the sign
    out = np.zeros((100, 100, 3), np.uint8)
    assert gen._hallucinated(out, sign_boxes=[(45, 45, 55, 55)]) is True


def test_hallucinated_false_when_detection_on_sign():
    gen = DiffusionBg("unused", seed=0)
    gen._scanner = _FakeScanner([[0.5, 0.5, 0.1, 0.1]])     # detection on the real sign
    out = np.zeros((100, 100, 3), np.uint8)
    assert gen._hallucinated(out, sign_boxes=[(45, 45, 55, 55)]) is False


def test_hallucinated_false_without_scanner():
    gen = DiffusionBg("unused", seed=0)                     # no scan_weights -> None
    out = np.zeros((100, 100, 3), np.uint8)
    assert gen._hallucinated(out, sign_boxes=[(45, 45, 55, 55)]) is False


def test_make_tile_composites_real_sign_over_regenerated_bg():
    """Background = (read-only) FluxFill output; the sign is the ORIGINAL crop pasted
    back with a FEATHERED edge — core stays byte-identical, border ramps (no hard seam),
    label stays valid. Box 40px wide -> ~4px ramp, so the inner core is untouched."""
    gen = DiffusionBg("unused", seed=0, imgsz=100)
    gen._pipe = _FakePipe(fill=7)                           # "regenerated" bg = 7
    sign = np.full((100, 100, 3), 200, np.uint8)            # original tile = 200
    gen.load_tile = lambda name: (sign, ["0 0.5 0.5 0.2 0.2"], [])   # bbox px (40,40,60,60)
    out, labels = gen.make_tile({"source_tile": "t"}, random.Random(0))
    assert labels == ["0 0.5 0.5 0.2 0.2"]
    assert (out[46:54, 46:54] == 200).all()                # core = original sign pixels
    assert (out[:38, :38] == 7).all()                      # background = regenerated
    assert 7 < out[40, 50, 0] < 200                        # bbox edge = feathered blend
