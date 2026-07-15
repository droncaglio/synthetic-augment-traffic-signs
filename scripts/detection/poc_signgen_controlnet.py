#!/usr/bin/env python
"""POC: sign generation via TEMPLATE + ControlNet-Canny (method #5-bis).

For ONE class, render the official template (marks/), warp it to varied poses, and let
ControlNet-Canny generate photorealistic signs whose CLASS is pinned by the template. Build a
contact sheet [warped template | Canny control | generated] to answer: does this give REAL
pose/appearance diversity (what img2img could not) with the glyph/class preserved?

Usage:
  conda activate longtail-synth
  python scripts/detection/poc_signgen_controlnet.py --class-name pl70 --device 0
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.generators.templates import load_template  # noqa: E402
from detection.generators.signgen_i2i import contact_sheet  # noqa: E402
from detection.generators.signgen_controlnet import (  # noqa: E402
    SignGenControlNet, DEFAULT_PROMPT, DEFAULT_NEG,
)


def _rgba_on(bg_val: int, rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA template onto a flat background for display."""
    m = (rgba[..., 3:4] > 10)
    return np.where(m, rgba[..., :3], bg_val).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="pl70",
                    help="comma-separated subset classes (rows), e.g. pl70,w58,p19,il100")
    ap.add_argument("--per-class", type=int, default=3, help="variants per class (columns)")
    ap.add_argument("--marks", default="data/tt100k/tt100k_2021/marks")
    ap.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet-id", default="lllyasviel/sd-controlnet-canny")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cn-scale", type=float, default=1.0)
    ap.add_argument("--color-anchor", action="store_true",
                    help="img2img from the colored template -> preserve the class palette")
    ap.add_argument("--strength", type=float, default=0.6, help="color-anchor img2img strength")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--neg-prompt", default=DEFAULT_NEG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="GPU index -> CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # before torch import

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        sys.exit("--classes vazio: passe ao menos uma classe (ex. --classes pl70,w58)")
    if args.per_class <= 0:
        sys.exit(f"--per-class deve ser > 0 (recebi {args.per_class})")
    mode = f"color-anchor(str={args.strength})" if args.color_anchor else "canny-only"
    print(f"[poc-cn] classes={classes} per_class={args.per_class} mode={mode} cn_scale={args.cn_scale}")

    gen = SignGenControlNet(args.model_id, args.controlnet_id, steps=args.steps,
                            guidance=args.guidance, cn_scale=args.cn_scale,
                            color_anchor=args.color_anchor, strength=args.strength,
                            prompt=args.prompt, neg_prompt=args.neg_prompt)
    rng = random.Random(args.seed)

    rows = []
    for cname in classes:
        tpl = load_template(cname, args.marks)
        variants = gen.generate(tpl, args.per_class, rng)
        row = [(_rgba_on(128, tpl), cname)]
        row += [(v["image"], "gerada") for v in variants]
        rows.append(row)
        print(f"  {cname}: {len(variants)} amostras")
    sheet = contact_sheet(rows, ["template"] + [f"var {i}" for i in range(args.per_class)])
    tag = "color" if args.color_anchor else "canny"
    default_out = f"poc_signgen_controlnet_{tag}_{'_'.join(classes)[:40]}.png"
    out = Path(args.out) if args.out else Path("reports/qa") / default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
