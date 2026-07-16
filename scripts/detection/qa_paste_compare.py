#!/usr/bin/env python
"""QA lado-a-lado dos braços de colagem: ORIGINAL (tile-fonte, placa real na posição original)
| copy_paste (placa real relocada) | signgen (placa sintética). copy_paste e signgen usam o
MESMO placement (fundo/posição/tamanho) — só muda real vs sintética.

Gera 2 folhas: tile inteiro (com bbox) + zoom da placa (p/ julgar aparência).

Uso:
  python scripts/detection/qa_paste_compare.py --per-class 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _draw(im, box, S=300):
    im = cv2.resize(im.copy(), (S, S))
    h, w = im.shape[:2]
    cx, cy, bw, bh = box
    cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                  (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
    return im


def _zoom(im, box, pad=0.3, S=150):
    h, w = im.shape[:2]
    cx, cy, bw, bh = box
    x1, y1 = max(0, int((cx - bw / 2 * (1 + pad)) * w)), max(0, int((cy - bh / 2 * (1 + pad)) * h))
    x2, y2 = min(w, int((cx + bw / 2 * (1 + pad)) * w)), min(h, int((cy + bh / 2 * (1 + pad)) * h))
    c = im[y1:y2, x1:x2]
    return cv2.resize(c, (S, S), interpolation=cv2.INTER_NEAREST) if c.size else np.zeros((S, S, 3), np.uint8)


def _load(p):
    return np.asarray(Image.open(p).convert("RGB")) if Path(p).exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/qa")
    args = ap.parse_args()

    root, prep = Path(args.tiles), Path(args.prepared)
    id2n = {c["id"]: c["name"] for c in json.loads((prep / "subset.json").read_text())["classes"]}
    pl = json.loads((prep / "placements_signgen_controlnet_seed42.json").read_text())
    T, CP, SG = root / "train/images", root / "arms/copy_paste/images", root / "arms/signgen_controlnet/images"
    by_cls: dict = {}
    for i, e in enumerate(pl):
        by_cls.setdefault(e["class_id"], []).append(i)

    rng = random.Random(args.seed)
    full_rows, zoom_rows = [], []
    for cid in sorted(by_cls):
        idxs = by_cls[cid][:]
        rng.shuffle(idxs)
        n = 0
        for i in idxs:
            if n >= args.per_class:
                break
            e = pl[i]
            src = _load(T / f"{e['source_tile']}.jpg")
            cpt = _load(CP / f"syn_copy_paste_{i:06d}.jpg")
            sgt = _load(SG / f"syn_signgen_controlnet_{i:06d}.jpg")
            if src is None or cpt is None or sgt is None:
                continue
            n += 1
            a, b, c = _draw(src, e["bbox"]), _draw(cpt, e["place"]), _draw(sgt, e["place"])
            for im, txt in [(a, id2n[cid] + " ORIG"), (b, "copy_paste"), (c, "signgen")]:
                cv2.putText(im, txt, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            sep = np.full((300, 3, 3), 255, np.uint8)
            full_rows.append(np.concatenate([a, sep, b, sep, c], axis=1))
            za, zb, zc = _zoom(src, e["bbox"]), _zoom(cpt, e["place"]), _zoom(sgt, e["place"])
            lab = np.full((150, 64, 3), 30, np.uint8)
            cv2.putText(lab, id2n[cid], (3, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            zs = np.full((150, 3, 3), 255, np.uint8)
            zoom_rows.append(np.concatenate([lab, za, zs, zb, zs, zc], axis=1))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not full_rows:
        sys.exit("nada gerado — os tiles dos braços existem? (rode os braços primeiro)")
    Image.fromarray(np.concatenate(full_rows, axis=0)).save(out / "paste_compare_full.png")
    Image.fromarray(np.concatenate(zoom_rows, axis=0)).save(out / "paste_compare_zoom.png")
    print(f"-> {out}/paste_compare_full.png  (ORIGINAL | copy_paste | signgen — tile inteiro)")
    print(f"-> {out}/paste_compare_zoom.png  (zoom da placa: ORIGINAL | copy_paste | signgen)")


if __name__ == "__main__":
    main()
