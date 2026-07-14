#!/usr/bin/env python
"""Precomputa máscaras SAM de placa para os tiles de treino (uma vez), cacheadas p/ os braços-máscara.

Itera as instâncias-fonte single-sign do treino (index_instances_by_class), roda SAM com a
bbox da placa como prompt, filtra por validade (re-tunada p/ placa-na-bbox) e grava
data/tt100k/masks/train/{tile}.png (só as que passam) + manifest.json. copy_paste_mask lê
essas máscaras e cai na silhueta geométrica onde faltar/for rejeitada.

Uso (workstation, GPU):
  python scripts/detection/precompute_sam_masks.py                     # todas
  python scripts/detection/precompute_sam_masks.py --limit 40 --overlays   # smoke + QA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detection.generators.manifests import index_instances_by_class  # noqa: E402
from detection.generators.bg_photometric import _yolo_to_px          # noqa: E402
from detection.generators.sam_masks import SamMasker, filter_mask, mask_path  # noqa: E402


def _save_overlay(img, box, mask_crop, out):
    x1, y1, x2, y2 = box
    ov = img.copy()
    reg = ov[y1:y2, x1:x2]
    m = mask_crop.astype(bool)
    reg[m] = (reg[m] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    ov[y1:y2, x1:x2] = reg
    Image.fromarray(np.concatenate([img, ov], axis=1)).save(out, quality=85)


def _stratified(inst, limit):
    by = {}
    for t, b, c in inst:
        by.setdefault(c, []).append((t, b, c))
    out = []
    while len(out) < limit and any(by.values()):
        for c in sorted(by):
            if by[c]:
                out.append(by[c].pop(0))
                if len(out) >= limit:
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--out", default="data/tt100k/masks")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = todas; N = smoke estratificado")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overlays", action="store_true", help="QA lado-a-lado (img|overlay)")
    args = ap.parse_args()

    tiles, out = Path(args.tiles), Path(args.out)
    tr = out / "train"
    tr.mkdir(parents=True, exist_ok=True)
    if args.overlays:
        (out / "overlays").mkdir(exist_ok=True)

    index = index_instances_by_class(tiles / "train" / "labels")
    inst = sorted((t, b, c) for c, lst in index.items() for (t, b) in lst)
    if args.limit:
        inst = _stratified(inst, args.limit)
    print(f"[info] {len(inst)} instâncias a segmentar")

    masker = SamMasker(device=args.device)
    records, n_ok, n_rej = [], 0, 0
    for i, (tile, bbox, cid) in enumerate(inst):
        p = mask_path(tr, tile)
        if args.resume and p.exists():
            n_ok += 1
            continue
        img = np.asarray(Image.open(tiles / "train" / "images" / f"{tile}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _yolo_to_px(bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_mask = masker.infer_mask(img, [x1, y1, x2, y2])[y1:y2, x1:x2]
        clean, m = filter_mask(crop_mask)
        records.append({"tile": tile, "class_id": cid, "bbox": bbox, **m})
        if m["status"] == "ok":
            Image.fromarray((clean * 255).astype(np.uint8)).save(p)
            n_ok += 1
        else:
            n_rej += 1
        if args.overlays:
            _save_overlay(img, (x1, y1, x2, y2), clean, out / "overlays" / f"{m['status']}_{tile}.jpg")
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(inst)}] ok={n_ok} rej={n_rej}", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "n_total": len(records), "n_ok": n_ok, "n_rejected": n_rej,
        "filters": {"area_min_ratio": 0.20, "solidity_min": 0.80, "border_touch": "disabled"},
        "model": "facebook/sam-vit-base", "records": records}, indent=2))
    rate = 100 * n_rej / max(1, len(records))
    print(f"[done] ok={n_ok} rejected={n_rej} ({rate:.1f}%) total={len(records)} -> {out}")


if __name__ == "__main__":
    main()
