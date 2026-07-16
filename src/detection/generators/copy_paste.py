"""Copy-Paste arm: real sign RELOCATED into another train (background) tile at a new
position — the orthogonal 'real new context' reference (FlexibleCP/Ghiasi-style). Uses
the placement manifest (recipient tile + target box). Self-contained paste (extract ->
resize with scale jitter -> paste -> recompute label).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image

from detection.generators.base import ArmGenerator, feather_alpha
from detection.generators.bg_photometric import _yolo_to_px


def _iou(a, b) -> float:
    """IoU of two normalized [cx,cy,w,h] boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


class CopyPaste(ArmGenerator):
    name = "copy_paste"

    def _blend_alpha(self, th: int, tw: int, source: dict) -> np.ndarray:
        """Alpha (th,tw,1) for compositing the crop. Base = rectangular feather (soft edge,
        shared with diffusion_bg). copy_paste_mask overrides with a tight silhouette."""
        return feather_alpha(th, tw)[..., None]

    def _sign_crop(self, source: dict, tw: int, th: int, rng: random.Random):
        """The (th,tw,3) sign to paste. Base = the REAL source crop resized. signgen_controlnet
        overrides to GENERATE a synthetic sign of the same class (same size -> paired paste)."""
        src_img, _labels, _ig = self.load_tile(source["source_tile"])
        h, w = src_img.shape[:2]
        x1, y1, x2, y2 = _yolo_to_px(source["bbox"], w, h)
        crop = src_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return np.asarray(Image.fromarray(crop).resize((tw, th)))

    def make_tile(self, source: dict, rng: random.Random):
        bg, bg_labels, _ = self.load_tile(source["recipient_tile"])
        H, W = bg.shape[:2]
        cx, cy, bw, bh = source["place"]
        tw, th = max(1, int(round(bw * W))), max(1, int(round(bh * H)))
        crop_r = self._sign_crop(source, tw, th, rng)
        if crop_r is None:
            return None

        px1 = max(0, min(W - tw, int(round(cx * W - tw / 2))))
        py1 = max(0, min(H - th, int(round(cy * H - th / 2))))
        out = bg.copy()
        # blend alpha (overridable): rectangular feather here; a TIGHT silhouette in
        # copy_paste_mask (to drop the rectangular halo of alien background at the corners)
        alpha = self._blend_alpha(th, tw, source)
        region = out[py1:py1 + th, px1:px1 + tw].astype(np.float32)
        blended = alpha * crop_r.astype(np.float32) + (1 - alpha) * region
        out[py1:py1 + th, px1:px1 + tw] = blended.astype(np.uint8)

        ncx, ncy = (px1 + tw / 2) / W, (py1 + th / 2) / H
        nw, nh = tw / W, th / H
        # realistic placement: the recipient's OWN sign sits under our paste -> drop the labels
        # it covers (IoU>0.3 = significant overlap -> would duplicate/conflict), keep the rest.
        # (empty recipient -> bg_labels=[] -> no-op.) pbox is the FINAL (clamped) paste box.
        pbox = [ncx, ncy, nw, nh]
        kept = [ln for ln in bg_labels if len(ln.split()) >= 5
                and _iou([float(v) for v in ln.split()[1:5]], pbox) <= 0.3]
        labels = kept + [f"{source['class_id']} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"]
        return out, labels
