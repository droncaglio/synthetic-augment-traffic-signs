#!/usr/bin/env python
"""POC: sign-APPEARANCE generation via SD1.5 img2img (method #4 of the extra arms).

For ONE class, build a side-by-side contact sheet — original real crop vs img2img variants
across a strength sweep — with SSIM(variant, original) per cell. Answers the POC question:
does low/medium-strength img2img preserve the glyph while adding useful variation?

Usage:
  conda activate longtail-synth
  python scripts/detection/poc_signgen.py --class-name pl70 --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.generators.manifests import index_instances_by_class  # noqa: E402
from detection.generators.signgen_i2i import (  # noqa: E402
    SignGenI2I, crop_bbox, contact_sheet, DEFAULT_PROMPT, DEFAULT_NEG,
)


def _ssim_note(orig: np.ndarray, var: np.ndarray) -> tuple[str, float | None]:
    """SSIM(var, orig) on grayscale (both are the same size). Falls back to normalized MAE
    for crops too small for the SSIM window (<7 px)."""
    import cv2
    go = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    gv = cv2.cvtColor(var, cv2.COLOR_RGB2GRAY)
    if min(go.shape) < 7:
        mae = float(np.abs(go.astype(np.float32) - gv.astype(np.float32)).mean()) / 255.0
        return f"MAE {mae:.2f}", None
    from skimage.metrics import structural_similarity
    val = float(structural_similarity(go, gv, data_range=255))
    return f"SSIM {val:.2f}", val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class-name", default="pl70")
    ap.add_argument("--n-sources", type=int, default=6, help="how many real crops (largest first)")
    ap.add_argument("--strengths", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--neg-prompt", default=DEFAULT_NEG)
    ap.add_argument("--device", default=None, help="GPU index -> CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # before torch import

    tiles, prepared = Path(args.tiles), Path(args.prepared)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]

    subset = json.loads((prepared / "subset.json").read_text())
    name2id = {c["name"]: c["id"] for c in subset["classes"]}
    if args.class_name not in name2id:
        sys.exit(f"class {args.class_name!r} not in subset ({sorted(name2id)})")
    cid = name2id[args.class_name]

    index = index_instances_by_class(tiles / "train" / "labels", single_sign_only=True)
    pool = index.get(cid, [])
    if not pool:
        sys.exit(f"no single-sign source tiles for class {args.class_name} (id {cid})")
    # largest bboxes first (most legible -> best to judge glyph fidelity)
    pool = sorted(pool, key=lambda tb: tb[1][2] * tb[1][3], reverse=True)[:args.n_sources]
    print(f"[poc] class={args.class_name} id={cid} sources={len(pool)} strengths={strengths}")

    gen = SignGenI2I(args.model_id, steps=args.steps, guidance=args.guidance,
                     prompt=args.prompt, neg_prompt=args.neg_prompt)
    rng = random.Random(args.seed)
    img_dir = tiles / "train" / "images"

    rows, ssim_acc = [], {s: [] for s in strengths}
    for stem, bbox in pool:
        tile = np.asarray(Image.open(img_dir / f"{stem}.jpg").convert("RGB"))
        orig = crop_bbox(tile, bbox)
        ch, cw = orig.shape[:2]
        variants = gen.generate_variants(orig, strengths, rng)
        row, notes = [(orig, f"{cw}x{ch}px")], {}
        for s in strengths:
            note, val = _ssim_note(orig, variants[s])
            if val is not None:
                ssim_acc[s].append(val)
            notes[s] = note
            row.append((variants[s], note))
        rows.append(row)
        print(f"  {stem}: {cw}x{ch}px -> " + " ".join(f"s{s}:{notes[s]}" for s in strengths))

    sheet = contact_sheet(rows, ["original"] + [f"strength {s}" for s in strengths])
    out = Path(args.out) if args.out else Path("reports/qa") / f"poc_signgen_{args.class_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"\n[poc] mean SSIM by strength: "
          + ", ".join(f"{s}={np.mean(v):.3f}" for s, v in ssim_acc.items() if v))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
