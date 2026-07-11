"""Unit tests for the pure (CPU) logic of diffusion_bg — inverted mask + hallucination
scan. The FluxFill inference itself is GPU/model and validated on the workstation."""
import numpy as np

from detection.generators.diffusion_bg import DiffusionBg


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
