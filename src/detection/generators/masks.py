"""Síntese de uma silhueta justa de placa (alpha) a partir da bbox + forma da classe.

O TT100K 2021 não traz máscara (só bbox + category), então aproximamos a silhueta pela
FAMÍLIA GEOMÉTRICA da classe (código): advertência `w*` = triângulo; indicação `i*`
(exceto `il*`) = retângulo; o resto (`p*`, `pl*`, `pn`, `il*`, `ph*`, `pm*`, ...) = círculo.
Usado por copy_paste_mask (e futuros braços-máscara) para colar SÓ a placa, sem o halo
retangular de fundo alheio nos cantos — testando se esse halo é o que prejudicou o
copy_paste simples na cauda.
"""
from __future__ import annotations

import cv2
import numpy as np


def sign_shape(name: str) -> str:
    """Família geométrica de uma classe TT100K pelo prefixo do código."""
    if name.startswith("w"):
        return "triangle"    # advertência (triangular)
    if name.startswith("il"):
        return "circle"      # velocidade mínima (azul, circular)
    if name.startswith("i"):
        return "rectangle"   # indicação (azul, retangular/quadrada)
    return "circle"          # p*, pl*, pn, pne, ph*, pm*, pa*, pr*, po, pcl -> circular


def feather_mask(binmask: np.ndarray, feather_px: int = 2) -> np.ndarray:
    """(H,W) float32 em [0,1] a partir de uma máscara binária, com feather PARA DENTRO
    (distance transform) — borda anti-serrilhada ao longo da silhueta. Compartilhado pela
    silhueta geométrica e pela máscara SAM (mesmo blend, comparação justa)."""
    m = (binmask > 0).astype(np.uint8)
    if m.max() == 0:
        return m.astype(np.float32)
    if feather_px and feather_px > 0:
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        return np.clip(d / float(feather_px), 0.0, 1.0).astype(np.float32)
    return m.astype(np.float32)


def shape_alpha(th: int, tw: int, shape: str, feather_px: int = 2) -> np.ndarray:
    """(th,tw) float32 em [0,1]: 1 dentro da silhueta inscrita, borda suave.

    rectangle -> bbox cheia (placa retangular não tem halo). circle -> elipse inscrita
    (lida com bbox não-quadrada). triangle -> isósceles ápice-pra-cima inscrito na bbox.
    """
    th, tw = int(max(1, th)), int(max(1, tw))
    m = np.zeros((th, tw), np.uint8)
    if shape == "rectangle":
        m[:] = 1
    elif shape == "triangle":
        pts = np.array([[tw // 2, 0], [0, th - 1], [tw - 1, th - 1]], np.int32)
        cv2.fillConvexPoly(m, pts, 1)
    else:  # circle / default -> elipse inscrita
        cv2.ellipse(m, (tw // 2, th // 2),
                    (max(1, tw // 2 - 1), max(1, th // 2 - 1)), 0, 0, 360, 1, -1)
    if m.max() == 0:      # degenerado (bbox minúscula) -> cai pra bbox cheia
        m[:] = 1
    return feather_mask(m, feather_px)
