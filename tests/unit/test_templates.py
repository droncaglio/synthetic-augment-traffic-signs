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


def _fake_blue(tmp_path, size=280):
    """A solid-blue sign with a white number (il*-like min-speed base)."""
    import cv2
    rgba = np.zeros((size, size, 4), np.uint8)
    c, R = size // 2, size // 2 - 10
    cv2.circle(rgba, (c, c), R, (40, 60, 170, 255), -1)            # solid blue, opaque
    cv2.putText(rgba, "50", (c - 55, c + 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255, 255), 6)                           # white number
    p = tmp_path / "il50.png"
    Image.fromarray(rgba).save(p)
    return p


def test_render_parametric_il_keeps_blue_bg_white_text(tmp_path):
    # il* are solid-blue with white number — must NOT become white-disc/black (a pl-lookalike)
    base = _fake_blue(tmp_path)
    out = render_parametric(base, "60", size=280, fill_from_bg=True, text_color=(255, 255, 255, 255))
    cx, cy, R = _sign_geometry(out[..., 3])
    side = out[int(cy) - 10:int(cy) + 10, int(cx) + 45:int(cx) + 70, :3]   # fill beside the number
    assert side[..., 2].mean() > side[..., 0].mean() + 30          # blue channel dominates (not white)
    center = out[int(cy) - 25:int(cy) + 25, int(cx) - 45:int(cx) + 45, :3]
    assert (center.mean(axis=2) > 200).any()                       # white glyph present


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
