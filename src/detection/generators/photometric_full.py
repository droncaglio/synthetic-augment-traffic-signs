"""Photometric-Full arm: like bg_photometric, but perturbs the WHOLE tile — background
AND sign — with the same cheap photometric/weather transform (brightness/contrast/gamma/
noise/fog). Physically faithful: real fog/haze/lighting covers the sign too, not only the
background. The orthogonal contrast to bg_photometric isolates whether perturbing the
SIGN's photometrics (the gamma/noise/fog beyond the runtime HSV that EVERY arm already
gets on the whole image) helps the tail.

Label stays valid — a photometric perturbation moves nothing (bbox/class unchanged).
"""
from __future__ import annotations

import numpy as np

from detection.generators.bg_photometric import BgPhotometric


class PhotometricFull(BgPhotometric):
    name = "photometric_full"

    def _preserve_mask(self, labels, h: int, w: int, source: dict) -> np.ndarray:
        # preserve NOTHING -> perturb_background hits the whole tile (sign included).
        return np.zeros((h, w), dtype=bool)
