"""Copy-Paste-Mask: como o copy_paste, mas cola a placa por uma SILHUETA JUSTA em vez da
bbox retangular — removendo o 'halo retangular' de fundo alheio nos cantos (que prejudicou
o copy_paste simples na cauda).

Fonte da silhueta (`mask_source`):
  - "sam" (default): máscara do SAM precomputada/cacheada (segue a borda real + perspectiva
    oblíqua). Cai pra silhueta GEOMÉTRICA quando o SAM está ausente/rejeitado.
  - "geometric": só a forma geométrica (círculo/triângulo/retângulo) — p/ ablação.
Placement, rótulo e resize são herdados de CopyPaste; só o alpha de blend difere.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from detection.generators.copy_paste import CopyPaste
from detection.generators.masks import feather_mask, shape_alpha, sign_shape
from detection.generators.sam_masks import load_cached_mask


class CopyPasteMask(CopyPaste):
    name = "copy_paste_mask"

    def __init__(self, tiles_dir, seed: int = 0, *, feather_px: int = 2, mask_source: str = "sam"):
        super().__init__(tiles_dir, seed)
        if mask_source not in ("sam", "geometric"):
            raise ValueError(f"mask_source deve ser 'sam' ou 'geometric', recebi {mask_source!r}")
        self.feather_px = feather_px
        self.mask_source = mask_source                       # "sam" (+fallback) | "geometric"
        self.masks_dir = self.tiles_dir.parent / "masks" / "train"
        # Loud guard: sam-mode sem cache = build_masks não rodou -> degradaria em silêncio p/
        # geométrico em TODOS os tiles. Avisa (não aborta: o fallback por-tile é intencional).
        cache_empty = not (self.masks_dir.exists() and any(self.masks_dir.glob("*.png")))
        if mask_source == "sam" and cache_empty:
            print(f"[WARN] copy_paste_mask(mask_source=sam): cache SAM vazio/ausente em "
                  f"{self.masks_dir} — rode `precompute_sam_masks.py` (ou reproduce.py "
                  f"--step build_masks). Sem cache, TODOS os tiles caem no fallback geométrico.")
        sub = self.tiles_dir.parent / "prepared" / "subset.json"
        self._id2name = ({c["id"]: c["name"] for c in json.loads(sub.read_text())["classes"]}
                         if sub.exists() else {})

    def _geometric_alpha(self, th: int, tw: int, source: dict) -> np.ndarray:
        name = self._id2name.get(source.get("class_id"), "")
        return shape_alpha(th, tw, sign_shape(name) if name else "circle", feather_px=self.feather_px)

    def _blend_alpha(self, th: int, tw: int, source: dict) -> np.ndarray:
        if self.mask_source == "sam":
            m = load_cached_mask(self.masks_dir, source.get("source_tile", ""))
            if m is not None and m.max() > 0:                        # não-vazio (evita paste nulo)
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
                return feather_mask(m, self.feather_px)[..., None]   # silhueta SAM
        return self._geometric_alpha(th, tw, source)[..., None]      # fallback geométrico
