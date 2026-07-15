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
    ap.add_argument("--class-name", default="pl70")
    ap.add_argument("--n", type=int, default=8, help="how many pose/seed variants")
    ap.add_argument("--marks", default="data/tt100k/tt100k_2021/marks")
    ap.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet-id", default="lllyasviel/sd-controlnet-canny")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cn-scale", type=float, default=1.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--neg-prompt", default=DEFAULT_NEG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="GPU index -> CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # before torch import

    tpl = load_template(args.class_name, args.marks)
    print(f"[poc-cn] class={args.class_name} template={tpl.shape} n={args.n} "
          f"steps={args.steps} cn_scale={args.cn_scale}")

    gen = SignGenControlNet(args.model_id, args.controlnet_id, steps=args.steps,
                            guidance=args.guidance, cn_scale=args.cn_scale,
                            prompt=args.prompt, neg_prompt=args.neg_prompt)
    rng = random.Random(args.seed)
    variants = gen.generate(tpl, args.n, rng)

    rows = []
    for i, v in enumerate(variants):
        rows.append([(_rgba_on(128, v["warped"]), f"pose {i}"),
                     (v["control"], "canny"),
                     (v["image"], "gerada")])
    sheet = contact_sheet(rows, ["template (pose)", "controle (canny)", "placa gerada"])
    out = Path(args.out) if args.out else Path("reports/qa") / f"poc_signgen_controlnet_{args.class_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
