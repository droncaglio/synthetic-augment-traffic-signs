"""Copy-Paste arm: real sign RELOCATED into another train (background) tile at a new
position — the orthogonal 'real new context' reference (FlexibleCP/Ghiasi-style). Uses
the placement manifest (recipient tile + target box). Self-contained paste (extract ->
resize with scale jitter -> paste -> recompute label).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image

from detection.generators.base import ArmGenerator
from detection.generators.bg_photometric import _yolo_to_px


def _feather_alpha(th: int, tw: int) -> np.ndarray:
    """Alpha (th,tw) = 1 in the center, ramping to ~0 over a small border (soft paste)."""
    f = max(1, min(tw, th) // 10)
    ay = np.ones(th, np.float32)
    ax = np.ones(tw, np.float32)
    for i in range(f):
        v = (i + 1) / (f + 1)
        ay[i] = ay[-1 - i] = min(ay[i], v)
        ax[i] = ax[-1 - i] = min(ax[i], v)
    return np.minimum(ay[:, None], ax[None, :])


class CopyPaste(ArmGenerator):
    name = "copy_paste"

    def make_tile(self, source: dict, rng: random.Random):
        src_img, _labels, _ig = self.load_tile(source["source_tile"])
        h, w = src_img.shape[:2]
        x1, y1, x2, y2 = _yolo_to_px(source["bbox"], w, h)
        crop = src_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        bg, bg_labels, _ = self.load_tile(source["recipient_tile"])
        H, W = bg.shape[:2]
        cx, cy, bw, bh = source["place"]
        tw, th = max(1, int(round(bw * W))), max(1, int(round(bh * H)))
        crop_r = np.asarray(Image.fromarray(crop).resize((tw, th)))

        px1 = max(0, min(W - tw, int(round(cx * W - tw / 2))))
        py1 = max(0, min(H - th, int(round(cy * H - th / 2))))
        out = bg.copy()
        # feathered alpha border to avoid hard 'glue-line' paste artifacts
        alpha = _feather_alpha(th, tw)[..., None]
        region = out[py1:py1 + th, px1:px1 + tw].astype(np.float32)
        blended = alpha * crop_r.astype(np.float32) + (1 - alpha) * region
        out[py1:py1 + th, px1:px1 + tw] = blended.astype(np.uint8)

        ncx, ncy = (px1 + tw / 2) / W, (py1 + th / 2) / H
        nw, nh = tw / W, th / H
        labels = list(bg_labels) + [
            f"{source['class_id']} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"]
        return out, labels
