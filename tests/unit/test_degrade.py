"""Unit tests for the sim-to-real degradation helpers (pure cv2/numpy)."""
import random

import numpy as np

from detection.generators.degrade import lapvar, sample_real_bbox, degrade_to_real


def _sharp_sign(size=200):
    """A high-contrast synthetic sign (sharp edges -> high lapvar)."""
    img = np.full((size, size, 3), 240, np.uint8)
    img[40:160, 40:160] = 20                       # a hard black square
    return img


def test_sample_real_bbox_from_pool():
    index = {7: [("t0", [0.5, 0.5, 0.1, 0.12]), ("t1", [0.4, 0.4, 0.2, 0.2])]}
    box = sample_real_bbox(7, index, random.Random(0))
    assert box in ([0.5, 0.5, 0.1, 0.12], [0.4, 0.4, 0.2, 0.2])


def test_sample_real_bbox_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        sample_real_bbox(9, {7: [("t", [0, 0, 1, 1])]}, random.Random(0))


def test_degrade_shape_and_softens():
    sign = _sharp_sign()
    out = degrade_to_real(sign, 40, random.Random(1))
    assert out.shape == (40, 40, 3) and out.dtype == np.uint8
    # degraded (blur+noise+jpeg) is softer than a plain INTER_AREA downscale to the same size
    import cv2
    down = cv2.resize(sign, (40, 40), interpolation=cv2.INTER_AREA)
    assert lapvar(out) < lapvar(down)              # blur/jpeg killed high-freq detail


def test_degrade_zero_ops_is_downscale():
    sign = _sharp_sign()
    out = degrade_to_real(sign, 32, random.Random(0), blur=(0, 0), noise=(0, 0), jpeg_q=(100, 100))
    assert out.shape == (32, 32, 3)                 # no crash with disabled ops
