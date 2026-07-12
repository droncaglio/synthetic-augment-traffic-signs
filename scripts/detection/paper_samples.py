#!/usr/bin/env python
"""Qualitative figure for the paper: the context-novelty ladder, one sign across arms.

Because every content arm consumes the SAME shared source manifest (seed 42), the same
real sign instance can be shown transformed by each arm side by side — a truly paired
ladder of increasing background novelty:

  original(=zero_aug=real_duplicate) | da_only | bg_photometric | copy_paste | diffusion_bg

Each cell is a magnified crop around the sign so the (background) change is visible.
- original / real_duplicate: identity — the source tile crop.
- da_only: rendered offline here (Ultralytics augment_hsv + fliplr from the arm config),
  labelled REPRESENTATIVE — the real arm applies it online at train time.
- bg_photometric / copy_paste / diffusion_bg: the actual generated tiles (copy_paste
  relocates the sign, so its own label gives the bbox). Missing arms render as "n/a".

Usage:
  python scripts/detection/paper_samples.py --tiles data/tt100k/tiles \
      --prepared data/tt100k/prepared --seed 42 --n 8 --zoom 1.6 \
      --out reports/qa/paper_ladder.png [--class-id 5] [--boxes]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ladder order; "original" and "da_only" are rendered, the rest are read from disk.
COLUMNS = ["original", "da_only", "bg_photometric", "copy_paste", "diffusion_bg"]
COL_LABELS = {"original": "Original (Real-dup.)", "da_only": "DA-only (repr.)",
              "bg_photometric": "Bg-Photometric", "copy_paste": "Copy-Paste",
              "diffusion_bg": "Diffusion-Bg"}
DA_AUG = {"fliplr": 0.5, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}


def _augment_hsv(img: np.ndarray, h: float, s: float, v: float, rng) -> np.ndarray:
    """Ultralytics augment_hsv (RGB in/out)."""
    r = rng.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)


def _da_only(img: np.ndarray, box, rng):
    """Representative da_only render: augment_hsv + optional horizontal flip."""
    out = _augment_hsv(img, DA_AUG["hsv_h"], DA_AUG["hsv_s"], DA_AUG["hsv_v"], rng)
    cx, cy, bw, bh = box
    if rng.random() < DA_AUG["fliplr"]:
        out = out[:, ::-1].copy()
        cx = 1.0 - cx
    return out, (cx, cy, bw, bh)


def _first_box(label_path: Path):
    if not label_path.exists():
        return None
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            return tuple(float(v) for v in p[1:5])
    return None


def _zoom(img: np.ndarray, box, pad: float):
    """Crop a square window around the normalized box (cx,cy,bw,bh) with pad margin."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    half = max(bw * w, bh * h) * (0.5 + pad)
    px, py = cx * w, cy * h
    x0, y0 = max(0, int(px - half)), max(0, int(py - half))
    x1, y1 = min(w, int(px + half)), min(h, int(py + half))
    crop = img[y0:y1, x0:x1]
    bx = ((px - bw * w / 2) - x0, (py - bh * h / 2) - y0, bw * w, bh * h)
    return crop, bx


def _cell(ax, img, box, pad, boxes):
    crop, bx = _zoom(img, box, pad)
    ax.imshow(crop, interpolation="nearest")
    if boxes:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2], bx[3], fill=False,
                               edgecolor="lime", linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=8, help="number of sample signs (rows)")
    ap.add_argument("--class-id", type=int, default=None, help="only sample this class")
    ap.add_argument("--zoom", type=float, default=1.6)
    ap.add_argument("--boxes", action="store_true", help="draw the sign bbox (default off)")
    ap.add_argument("--render-seed", type=int, default=0, help="RNG for the da_only render")
    ap.add_argument("--out", default="reports/qa/paper_ladder.png")
    args = ap.parse_args()

    tiles, prepared = Path(args.tiles), Path(args.prepared)
    sources = json.loads((prepared / f"sources_seed{args.seed}.json").read_text())
    train_img = tiles / "train" / "images"
    rng = np.random.default_rng(args.render_seed)

    # pick sample source indices (optionally filtered by class), spread across the list
    idxs = [i for i, s in enumerate(sources)
            if args.class_id is None or s["class_id"] == args.class_id]
    if not idxs:
        raise SystemExit(f"no sources for class_id={args.class_id}")
    step = max(1, len(idxs) // args.n)
    picks = idxs[::step][:args.n]

    nrows, ncols = len(picks), len(COLUMNS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0), squeeze=False)
    for c, col in enumerate(COLUMNS):
        axes[0][c].set_title(COL_LABELS[col], fontsize=9)

    for r, i in enumerate(picks):
        src = sources[i]
        src_box = tuple(src["bbox"])
        orig_path = train_img / f"{src['source_tile']}.jpg"
        orig = np.asarray(Image.open(orig_path).convert("RGB")) if orig_path.exists() else None
        for c, col in enumerate(COLUMNS):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if col == "original":
                if orig is not None:
                    _cell(ax, orig, src_box, args.zoom, args.boxes)
            elif col == "da_only":
                if orig is not None:
                    img, box = _da_only(orig, src_box, rng)
                    _cell(ax, img, box, args.zoom, args.boxes)
            else:
                name = f"syn_{col}_{i:06d}"
                p = tiles / "arms" / col / "images" / f"{name}.jpg"
                box = _first_box(tiles / "arms" / col / "labels" / f"{name}.txt") or src_box
                if p.exists():
                    _cell(ax, np.asarray(Image.open(p).convert("RGB")), box, args.zoom, args.boxes)
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.6")
        axes[r][0].set_ylabel(f"{src['source_tile']}", fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=20)

    fig.suptitle("Context-novelty ladder — same real sign across arms (seed "
                 f"{args.seed})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"-> {out}  ({nrows} signs x {ncols} arms)")


if __name__ == "__main__":
    main()
