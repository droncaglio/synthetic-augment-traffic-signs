"""Bg-Photometric arm: real sign fixed, cheap photometric/weather perturbation of the
BACKGROUND only (brightness/contrast/gamma/noise/fog). The cheap rung of the context
ladder — novelty of context without generating anything. All subset signs in the tile
are preserved pixel-exact (labels unchanged), only the non-sign pixels are perturbed.
"""
from __future__ import annotations

import random

import numpy as np

from detection.generators.base import ArmGenerator


def _yolo_to_px(box, w, h):
    cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2) * w)); y1 = int(round((cy - bh / 2) * h))
    x2 = int(round((cx + bw / 2) * w)); y2 = int(round((cy + bh / 2) * h))
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def perturb_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Deterministic photometric+weather perturbation (numpy, seeded by rng)."""
    x = img.astype(np.float32) / 255.0
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)      # brightness/contrast
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)              # gamma
    if rng.random() < 0.5:                                     # gaussian noise
        nrng = np.random.default_rng(rng.randrange(2 ** 31))
        x = x + nrng.normal(0, rng.uniform(0.01, 0.05), x.shape)
    if rng.random() < 0.3:                                     # fog-ish haze
        fog = rng.uniform(0.1, 0.4)
        x = x * (1 - fog) + 0.6 * fog
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class BgPhotometric(ArmGenerator):
    name = "bg_photometric"

    def _preserve_mask(self, labels, h: int, w: int, source: dict) -> np.ndarray:
        """Boolean (h,w): pixels a manter pixel-exato (as placas). Base = união dos RETÂNGULOS
        de bbox. bg_photometric_mask sobrescreve com a SILHUETA justa (SAM/geométrica)."""
        sign = np.zeros((h, w), dtype=bool)
        for ln in labels:
            p = ln.split()
            x1, y1, x2, y2 = _yolo_to_px([float(v) for v in p[1:5]], w, h)
            sign[y1:y2, x1:x2] = True
        return sign

    def make_tile(self, source: dict, rng: random.Random):
        img, labels, _ignores = self.load_tile(source["source_tile"])
        h, w = img.shape[:2]
        sign = self._preserve_mask(labels, h, w, source)       # placas a preservar
        out = perturb_background(img, rng)
        out[sign] = img[sign]                                  # signs pixel-exact
        return out, labels
