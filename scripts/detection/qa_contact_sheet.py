#!/usr/bin/env python
"""Visual-QA contact sheet for a generated arm (diffusion_bg especially).

Pairs each synthetic tile with its ORIGINAL source tile, side by side, with the
sign bbox drawn in green on both. One glance confirms the two things that keep the
arm reviewer-proof:
  (1) the sign bbox is byte-identical (real sign composited back, label still valid);
  (2) the background is regenerated but stays in-domain (coherent street scene).

The synthetic tiles are written in source order (syn_<arm>_000000..), so tile i pairs
with the i-th entry of prepared/sources_seed<seed>.json.

Usage:
  python scripts/detection/qa_contact_sheet.py --arm diffusion_bg \
      --tiles data/tt100k/tiles --prepared data/tt100k/prepared --seed 42 \
      --out reports/qa/diffusion_bg_seed42.png [--pairs-per-row 2]
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


def _boxes_px(label_path: Path, w: int, h: int) -> list[tuple]:
    """YOLO-normalized labels -> [(x, y, bw, bh)] in pixels (matplotlib Rectangle args)."""
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((( cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
    return out


def _draw(ax, img: np.ndarray, label_path: Path, title: str) -> None:
    h, w = img.shape[:2]
    ax.imshow(img)
    for (x, y, bw, bh) in _boxes_px(label_path, w, h):
        ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor="lime", linewidth=1.5))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="diffusion_bg")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs-per-row", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="cap tiles shown (0 = all present)")
    args = ap.parse_args()

    tiles, prepared = Path(args.tiles), Path(args.prepared)
    gen_dir = tiles / "arms" / args.arm
    gen_imgs = sorted((gen_dir / "images").glob(f"syn_{args.arm}_*.jpg"))
    if args.limit:
        gen_imgs = gen_imgs[:args.limit]
    if not gen_imgs:
        raise SystemExit(f"no generated tiles in {gen_dir/'images'} — run generate_arm first.")

    sources = json.loads((prepared / f"sources_seed{args.seed}.json").read_text())
    train_img = tiles / "train" / "images"

    ppr = args.pairs_per_row
    n = len(gen_imgs)
    rows = (n + ppr - 1) // ppr
    fig, axes = plt.subplots(rows, ppr * 2, figsize=(ppr * 2 * 2.4, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for i, gpath in enumerate(gen_imgs):
        idx = int(gpath.stem.rsplit("_", 1)[1])          # syn_<arm>_000007 -> 7
        src_stem = sources[idx]["source_tile"] if idx < len(sources) else None
        r, c = divmod(i, ppr)
        gen_img = np.asarray(Image.open(gpath).convert("RGB"))
        gen_lbl = gen_dir / "labels" / f"{gpath.stem}.txt"
        if src_stem is not None and (train_img / f"{src_stem}.jpg").exists():
            orig = np.asarray(Image.open(train_img / f"{src_stem}.jpg").convert("RGB"))
            _draw(axes[r][c * 2], orig, tiles / "train" / "labels" / f"{src_stem}.txt",
                  f"orig {src_stem}")
        _draw(axes[r][c * 2 + 1], gen_img, gen_lbl, f"gen {gpath.stem.split('_')[-1]}")

    fig.suptitle(f"QA {args.arm} — orig | gen (bbox=lime), seed {args.seed}, n={n}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out or f"reports/qa/{args.arm}_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}  ({n} tiles)")


if __name__ == "__main__":
    main()
