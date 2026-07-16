#!/usr/bin/env python
"""REQ2 deliverable: match generated signs to the real small-object distribution + validate.

For each class: generate signs, crop the sign, sample a real target size, degrade to real
statistics. Reports, per class: sharpness (lapvar) of REAL vs downscale-only vs DEGRADED (the
degraded should land in the real range), and the verifier valid-rate CLEAN vs DEGRADED (class
must survive the degradation). Writes reports/qa/sim2real.{json,png}.

Usage:
  python scripts/detection/qa_sim2real.py --classes pl70,w58,p19,ph5 --per-class 20 --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.generators.manifests import index_instances_by_class  # noqa: E402
from detection.generators.templates import load_template  # noqa: E402
from detection.generators.signgen_i2i import contact_sheet  # noqa: E402
from detection.generators.signgen_controlnet import SignGenControlNet  # noqa: E402
from detection.generators.degrade import lapvar, sample_real_bbox, degrade_to_real  # noqa: E402
from detection.verifier import SignClassifier, load_crop  # noqa: E402


def _alpha_bbox(rgba):
    ys, xs = np.where(rgba[..., 3] > 10)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None


def _up(img, s=96):
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="pl70,w58,p19,ph5")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--marks", default="data/tt100k/tt100k_2021/marks")
    ap.add_argument("--weights", default="data/tt100k/verifier/convnext_signcls.pt")
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--blur", type=float, nargs=2, default=[0.35, 0.85])   # calibrated default
    ap.add_argument("--noise", type=float, nargs=2, default=[1.5, 4.0])
    ap.add_argument("--jpeg-q", type=int, nargs=2, default=[45, 80])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/qa/sim2real.json")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    tiles = Path(args.tiles)
    sub = json.loads(Path("data/tt100k/prepared/subset.json").read_text())
    name2id = {c["name"]: c["id"] for c in sub["classes"]}
    index = index_instances_by_class(tiles / "train" / "labels", single_sign_only=True)

    gen = SignGenControlNet(color_anchor=True, strength=0.6, steps=30)
    clf = SignClassifier(weights_path=args.weights) if Path(args.weights).exists() else None
    rng = random.Random(args.seed)

    per_class, rows = {}, []
    for cname in classes:
        cid = name2id[cname]
        tpl = load_template(cname, args.marks)
        variants = gen.generate(tpl, args.per_class, rng)
        clean, downonly, degraded, tgts = [], [], [], []
        for v in variants:
            bb = _alpha_bbox(v["warped"])
            crop = v["image"][bb[1]:bb[3], bb[0]:bb[2]] if bb else v["image"]
            box = sample_real_bbox(cid, index, rng)
            t = max(6, int(round(max(box[2], box[3]) * 640)))       # target px (real max-dim)
            tgts.append(t)
            clean.append(crop)
            downonly.append(cv2.resize(crop, (t, t), interpolation=cv2.INTER_AREA))
            degraded.append(degrade_to_real(crop, t, rng, blur=tuple(args.blur),
                                             noise=tuple(args.noise), jpeg_q=tuple(args.jpeg_q)))
        # real crops (native size) for the sharpness target
        real = [load_crop(tiles / "train" / "images", s, b) for s, b in index[cid][:40]]
        real = [r for r in real if min(r.shape[:2]) >= 6]

        def med_lap(cs):
            return round(float(np.median([lapvar(c) for c in cs])), 1) if cs else None
        rec = {"n": len(variants), "target_px_med": int(np.median(tgts)),
               "lapvar_real": med_lap(real), "lapvar_downscale_only": med_lap(downonly),
               "lapvar_degraded": med_lap(degraded)}
        if clf:
            rec["valid_clean"] = round(clf.valid_rate(clean, cid, args.conf_thr)["top1_acc"], 3)
            rec["valid_degraded"] = round(clf.valid_rate(degraded, cid, args.conf_thr)["top1_acc"], 3)
        per_class[cname] = rec
        print(f"  {cname:6s} tgt~{rec['target_px_med']}px  lapvar real={rec['lapvar_real']} "
              f"down={rec['lapvar_downscale_only']} degr={rec['lapvar_degraded']}"
              + (f"  valid clean={rec['valid_clean']} degr={rec['valid_degraded']}" if clf else ""))
        # sheet: real | clean(down-only) | 3x degraded  (all upscaled NEAREST for view)
        rr = real[rng.randrange(len(real))] if real else np.zeros((8, 8, 3), np.uint8)
        row = [(_up(rr), f"{cname} real"), (_up(downonly[0]), "down-only")]
        row += [(_up(degraded[i]), "degr") for i in range(min(3, len(degraded)))]
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"blur": args.blur, "noise": args.noise,
                                          "jpeg_q": args.jpeg_q, "per_class": per_class}, indent=2))
    sheet = contact_sheet(rows, ["real", "só-downscale", "degr 0", "degr 1", "degr 2"])
    Image.fromarray(sheet).save("reports/qa/sim2real.png")
    print(f"\n-> {args.out}\n-> reports/qa/sim2real.png")


if __name__ == "__main__":
    main()
