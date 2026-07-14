"""Máscaras SAM para silhueta de placa (precomputadas + cacheadas, com fallback geométrico).

O SAM vanilla (facebook/sam-vit-base), com prompt = bbox da placa, dá uma silhueta que
supera a forma geométrica: segue a borda real e a projeção oblíqua, com blend limpo. Aqui
precomputamos uma máscara por tile de treino (single-sign), filtramos por validade
(re-tunada p/ placa-na-bbox), cacheamos como PNG, e o copy_paste_mask (e futuros braços)
leem — caindo pra silhueta geométrica quando o SAM está ausente/rejeitado.

Portado/adaptado do ferramental MedSAM do ENIAC
(synthetic-allocation-derma/scripts/classification/generate_masks_pad.py). Deps pesadas
(torch/transformers) são importadas LAZY -> o módulo importa numa máquina CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAM_MODEL_ID = "facebook/sam-vit-base"

# Thresholds para placa-DENTRO-da-bbox (DIFERENTES do filtro lesão-na-imagem do ENIAC):
AREA_MIN_RATIO = 0.20   # a máscara deve preencher >=20% do crop justo; menor = SAM pegou fragmento
SOLIDITY_MIN = 0.80     # placa é convexa (círculo/triângulo/retângulo); rasgada = pegou poste/fundo
# SEM filtro de border-touch: bbox justa -> a silhueta ENCOSTA nas bordas por construção
# (o inverso do caso lesão, onde encostar na borda sinalizava recorte ruim).


def mask_path(masks_dir, tile: str) -> Path:
    return Path(masks_dir) / f"{tile}.png"


def load_cached_mask(masks_dir, tile: str):
    """Máscara binária (HxW uint8 0/1) para um tile-fonte, ou None se não cacheada.

    Só máscaras que PASSARAM no filtro são escritas em disco -> existência == válida.
    """
    if not tile:
        return None
    p = mask_path(masks_dir, tile)
    if not p.exists():
        return None
    from PIL import Image
    return (np.asarray(Image.open(p).convert("L")) > 127).astype(np.uint8)


def filter_mask(mask: np.ndarray):
    """Maior componente conexa + fill holes + validade re-tunada. Retorna (clean uint8, métricas)."""
    from scipy import ndimage as ndi
    from skimage import measure
    h, w = mask.shape
    total = h * w
    if mask.sum() == 0:
        return mask, {"area_ratio": 0.0, "solidity": 0.0, "status": "rejected", "reason": "empty"}
    lab = measure.label(mask, connectivity=2)
    regs = measure.regionprops(lab)
    if not regs:
        return mask, {"area_ratio": 0.0, "solidity": 0.0, "status": "rejected", "reason": "no_region"}
    largest = max(regs, key=lambda r: r.area)
    # fill_holes DE PROPÓSITO antes de medir solidez: uma placa "60"/"30" tem furos nos
    # dígitos; queremos a convexidade da SILHUETA EXTERNA (contorno da placa), não da textura
    # interna. Solidez pós-fill ~1 p/ disco/triângulo; baixa = SAM pegou poste/fundo rasgado.
    clean = ndi.binary_fill_holes(lab == largest.label).astype(np.uint8)
    r = max(measure.regionprops(measure.label(clean, connectivity=2)), key=lambda x: x.area)
    area_ratio = float(r.area) / total
    solidity = float(r.solidity)
    status, reason = "ok", ""
    if area_ratio < AREA_MIN_RATIO:
        status, reason = "rejected", f"area_small({area_ratio:.3f}<{AREA_MIN_RATIO})"
    elif solidity < SOLIDITY_MIN:
        status, reason = "rejected", f"low_solidity({solidity:.3f}<{SOLIDITY_MIN})"
    return clean, {"area_ratio": round(area_ratio, 4), "solidity": round(solidity, 3),
                   "status": status, "reason": reason}


class SamMasker:
    """Segmentador SAM (facebook/sam-vit-base) por prompt de bbox. GPU-pesado; imports lazy."""

    def __init__(self, device: str | None = None, model_id: str = SAM_MODEL_ID):
        self.device, self.model_id = device, model_id
        self._model = self._proc = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import SamModel, SamProcessor
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SamModel.from_pretrained(self.model_id).to(self.device).eval()
        self._proc = SamProcessor.from_pretrained(self.model_id)

    def infer_mask(self, img_rgb: np.ndarray, bbox_xyxy) -> np.ndarray:
        """Máscara binária (HxW uint8) da placa indicada por bbox_xyxy (px) sobre img_rgb."""
        import torch
        from PIL import Image
        self._load()
        inp = self._proc(Image.fromarray(img_rgb),
                         input_boxes=[[list(map(float, bbox_xyxy))]],
                         return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inp, multimask_output=False)
        probs = self._proc.image_processor.post_process_masks(
            out.pred_masks.sigmoid().cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu(), binarize=False)
        return (probs[0][0, 0].numpy() > 0.5).astype(np.uint8)
