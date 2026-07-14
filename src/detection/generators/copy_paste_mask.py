"""Copy-Paste-Mask: mesma realocação do copy_paste, mas a placa é composta por uma
SILHUETA JUSTA (círculo/triângulo/retângulo inscrito na bbox) em vez da bbox retangular.
Remove o 'halo retangular' de fundo alheio nos cantos do crop — testando se esse halo é
por que o copy_paste simples piorou a cauda no test.

Sem máscara no TT100K, a silhueta é sintetizada pela família de forma da classe
(detection.generators.masks). Placement, rótulo e resize são herdados de CopyPaste sem
mudança — só o alpha de blend difere.
"""
from __future__ import annotations

import json

import numpy as np

from detection.generators.copy_paste import CopyPaste
from detection.generators.masks import shape_alpha, sign_shape


class CopyPasteMask(CopyPaste):
    name = "copy_paste_mask"

    def __init__(self, tiles_dir, seed: int = 0, *, feather_px: int = 2):
        super().__init__(tiles_dir, seed)
        self.feather_px = feather_px
        # id -> nome da classe, para escolher a silhueta por instância de origem
        sub = self.tiles_dir.parent / "prepared" / "subset.json"
        self._id2name = ({c["id"]: c["name"] for c in json.loads(sub.read_text())["classes"]}
                         if sub.exists() else {})

    def _blend_alpha(self, th: int, tw: int, source: dict) -> np.ndarray:
        name = self._id2name.get(source.get("class_id"), "")
        shape = sign_shape(name) if name else "circle"
        return shape_alpha(th, tw, shape, feather_px=self.feather_px)[..., None]
