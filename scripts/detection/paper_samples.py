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
    """Ultralytics-style HSV jitter (RGB in/out), via PIL — no cv2 dependency.

    PIL 'HSV' hue is 0-255 (cv2 uses 0-179), so the hue wrap is mod 256; the visual
    effect matches — this render is a REPRESENTATIVE illustration of the da_only arm.
    """
    r = rng.uniform(-1, 1, 3) * [h, s, v] + 1.0
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


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


def _arm_image(col: str, i: int, src: dict, orig, tiles: Path, rng):
    """(img, box) for one arm+sample, or None if the tile is absent (n/a)."""
    src_box = tuple(src["bbox"])
    if col == "original":
        return (orig, src_box) if orig is not None else None
    if col == "da_only":
        return _da_only(orig, src_box, rng) if orig is not None else None
    name = f"syn_{col}_{i:06d}"
    p = tiles / "arms" / col / "images" / f"{name}.jpg"
    if not p.exists():
        return None
    box = _first_box(tiles / "arms" / col / "labels" / f"{name}.txt") or src_box
    return np.asarray(Image.open(p).convert("RGB")), box


def _cell(ax, img, box, pad, boxes):
    crop, bx = _zoom(img, box, pad)
    ax.imshow(crop, interpolation="nearest")
    if boxes:
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2], bx[3], fill=False,
                               edgecolor="lime", linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])


def _save_crop(img, box, pad, path: Path) -> None:
    """Write just the magnified crop (no axes/title) — clean asset for LaTeX composition."""
    crop, _ = _zoom(img, box, pad)
    Image.fromarray(crop).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=8, help="number of sample signs (rows)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit source indices to show (cherry-pick; overrides --n/--class-id)")
    ap.add_argument("--class-id", type=int, default=None, help="only sample this class")
    ap.add_argument("--zoom", type=float, default=1.6)
    ap.add_argument("--boxes", action="store_true", help="draw the sign bbox (default off)")
    ap.add_argument("--render-seed", type=int, default=0, help="RNG for the da_only render")
    ap.add_argument("--out", default="reports/qa/ladder/paper_ladder.png")
    ap.add_argument("--per-sample", action="store_true",
                    help="also write one PNG per sign (a 1xN arm row) -> <out>__<tile>.png")
    ap.add_argument("--per-cell", action="store_true",
                    help="also write one clean crop per arm+sign (no chrome) for LaTeX "
                         "composition -> <out>__<tile>__<arm>.png")
    args = ap.parse_args()

    tiles, prepared = Path(args.tiles), Path(args.prepared)
    sources = json.loads((prepared / f"sources_seed{args.seed}.json").read_text())
    train_img = tiles / "train" / "images"
    rng = np.random.default_rng(args.render_seed)

    # pick sample source indices: explicit cherry-pick, else spread across the class
    if args.indices:
        picks = [i for i in args.indices if 0 <= i < len(sources)]
        if not picks:
            raise SystemExit(f"--indices out of range (n_sources={len(sources)})")
    else:
        idxs = [i for i, s in enumerate(sources)
                if args.class_id is None or s["class_id"] == args.class_id]
        if not idxs:
            raise SystemExit(f"no sources for class_id={args.class_id}")
        step = max(1, len(idxs) // args.n)
        picks = idxs[::step][:args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _row(axrow, i, titles: bool):
        """Fill one axis row for source index i; returns the loaded (img, box) per arm."""
        src = sources[i]
        op = train_img / f"{src['source_tile']}.jpg"
        orig = np.asarray(Image.open(op).convert("RGB")) if op.exists() else None
        loaded = {}
        for c, col in enumerate(COLUMNS):
            ax = axrow[c]
            ax.set_xticks([]); ax.set_yticks([])
            if titles:
                ax.set_title(COL_LABELS[col], fontsize=9)
            res = _arm_image(col, i, src, orig, tiles, rng)
            if res is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.6")
            else:
                _cell(ax, res[0], res[1], args.zoom, args.boxes)
                loaded[col] = res
        return src, loaded

    # 1) combined grid (overview)
    nrows, ncols = len(picks), len(COLUMNS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0), squeeze=False)
    per_sample_data = []
    for r, i in enumerate(picks):
        src, loaded = _row(axes[r], i, titles=(r == 0))
        axes[r][0].set_ylabel(src["source_tile"], fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=20)
        per_sample_data.append((i, src, loaded))
    fig.suptitle(f"Context-novelty ladder — same real sign across arms (seed {args.seed})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}  ({nrows} signs x {ncols} arms)")

    # 2) one PNG per sign (a 1xN arm row)
    if args.per_sample:
        for i in picks:
            f1, ax1 = plt.subplots(1, ncols, figsize=(ncols * 2.0, 2.2), squeeze=False)
            src, _ = _row(ax1[0], i, titles=True)
            f1.tight_layout()
            p = out.with_name(f"{out.stem}__{i:06d}_{src['source_tile']}{out.suffix}")
            f1.savefig(p, dpi=200)
            plt.close(f1)
            print(f"   per-sample -> {p.name}")

    # 3) one clean crop per arm+sign (no chrome) for free LaTeX composition
    if args.per_cell:
        for i, src, loaded in per_sample_data:
            for col, (img, box) in loaded.items():
                p = out.with_name(f"{out.stem}__{i:06d}_{src['source_tile']}__{col}{out.suffix}")
                _save_crop(img, box, args.zoom, p)
        print(f"   per-cell -> {out.stem}__<tile>__<arm>{out.suffix} "
              f"({sum(len(l) for _, _, l in per_sample_data)} crops)")


if __name__ == "__main__":
    main()
