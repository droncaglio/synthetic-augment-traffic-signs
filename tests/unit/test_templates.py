"""Unit tests for the template helpers (pure, no GPU/model) — method #5-bis."""
import random

import numpy as np
from PIL import Image

from detection.generators.templates import (
    render_parametric, pose_warp, template_canny, _sign_geometry,
)


def _fake_ring(tmp_path, size=280):
    """A red ring on white interior over transparent corners (a pl*-like base icon)."""
    import cv2
    rgba = np.zeros((size, size, 4), np.uint8)
    c, R = size // 2, size // 2 - 10
    cv2.circle(rgba, (c, c), R, (255, 255, 255, 255), -1)          # white disc + opaque
    cv2.circle(rgba, (c, c), R, (200, 30, 30, 255), 16)            # red ring
    cv2.putText(rgba, "40", (c - 55, c + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 0, 0, 255), 6)                                  # old number
    p = tmp_path / "pl40.png"
    Image.fromarray(rgba).save(p)
    return p


def test_render_parametric_draws_number_and_keeps_ring(tmp_path):
    base = _fake_ring(tmp_path)
    out = render_parametric(base, "70", size=280)
    assert out.shape == (280, 280, 4)
    cx, cy, R = _sign_geometry(out[..., 3])
    # center has dark pixels (the rendered "70")
    center = out[int(cy) - 30:int(cy) + 30, int(cx) - 45:int(cx) + 45, :3]
    assert (center.mean(axis=2) < 90).any()                        # black glyph present
    # the red ring survived (red-ish pixels near the rim)
    rim = out[int(cy) - int(R):int(cy) + int(R), int(cx) - int(R):int(cx) + int(R), :3]
    red = (rim[..., 0] > 120) & (rim[..., 1] < 90) & (rim[..., 2] < 90)
    assert red.any()


def test_pose_warp_preserves_shape_and_gives_matrix(tmp_path):
    base = _fake_ring(tmp_path)
    tpl = np.asarray(Image.open(base).convert("RGBA"))
    warped, H = pose_warp(tpl, random.Random(0))
    assert warped.shape == tpl.shape
    assert H.shape == (3, 3) and abs(np.linalg.det(H)) > 1e-6      # invertible
    assert (warped[..., 3] > 10).any()                            # sign still visible


def test_template_canny_returns_edges(tmp_path):
    base = _fake_ring(tmp_path)
    tpl = np.asarray(Image.open(base).convert("RGBA"))
    edges = template_canny(tpl)
    assert edges.dtype == np.uint8 and edges.ndim == 2
    assert set(np.unique(edges)).issubset({0, 255})
    assert 0.001 < (edges > 0).mean() < 0.5                        # non-trivial edges
