"""Bg-Photometric-Mask: como o bg_photometric (perturba SÓ o fundo, placa pixel-exato),
mas preserva a placa pela **silhueta justa** em vez do RETÂNGULO da bbox — então os cantos
de fundo dentro da bbox (entre a silhueta e o retângulo) também são perturbados.

Silhueta = máscara SAM cacheada (do #1v2), com fallback geométrico por-tile. Preservação
HARD (binária) -> placa pixel-exato, como o bg_photometric. Reusa a infra de máscara.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from detection.generators.bg_photometric import BgPhotometric, _yolo_to_px
from detection.generators.masks import shape_alpha, sign_shape
from detection.generators.sam_masks import load_cached_mask


class BgPhotometricMask(BgPhotometric):
    name = "bg_photometric_mask"

    def __init__(self, tiles_dir, seed: int = 0, *, mask_source: str = "sam"):
        super().__init__(tiles_dir, seed)
        if mask_source not in ("sam", "geometric"):
            raise ValueError(f"mask_source deve ser 'sam' ou 'geometric', recebi {mask_source!r}")
        self.mask_source = mask_source
        self.masks_dir = self.tiles_dir.parent / "masks" / "train"
        cache_empty = not (self.masks_dir.exists() and any(self.masks_dir.glob("*.png")))
        if mask_source == "sam" and cache_empty:
            print(f"[WARN] bg_photometric_mask(mask_source=sam): cache SAM vazio/ausente em "
                  f"{self.masks_dir} — rode `precompute_sam_masks.py` (ou reproduce.py "
                  f"--step build_masks). Sem cache, TODAS as placas caem no fallback geométrico.")
        sub = self.tiles_dir.parent / "prepared" / "subset.json"
        self._id2name = ({c["id"]: c["name"] for c in json.loads(sub.read_text())["classes"]}
                         if sub.exists() else {})

    def _preserve_mask(self, labels, h: int, w: int, source: dict) -> np.ndarray:
        sign = np.zeros((h, w), dtype=bool)
        # cache é 1 máscara/tile (single-sign) -> só usar SAM quando há exatamente 1 placa
        sam = (load_cached_mask(self.masks_dir, source.get("source_tile", ""))
               if self.mask_source == "sam" and len(labels) == 1 else None)
        for ln in labels:
            p = ln.split()
            cid = int(p[0])
            x1, y1, x2, y2 = _yolo_to_px([float(v) for v in p[1:5]], w, h)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if sam is not None and sam.max() > 0:
                # cv2.resize((bw,bh)) -> shape (bh,bw) = (h,w), alinha com sign[y1:y2, x1:x2]
                sil = cv2.resize(sam, (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
            else:
                # feather_px=0: silhueta HARD-binária (placa pixel-exato, como o bg_photometric);
                # shape_alpha(th=bh, tw=bw) -> shape (bh,bw), mesma orientação do resize acima.
                name = self._id2name.get(cid, "")
                sil = shape_alpha(bh, bw, sign_shape(name) if name else "circle", feather_px=0) > 0.5
            sign[y1:y2, x1:x2] |= sil
        return sign
