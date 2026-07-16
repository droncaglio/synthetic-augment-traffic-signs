#!/usr/bin/env python
"""REQ1 deliverable: measure the VALID-LABEL RATE of the template+ControlNet signgen.

Generates N samples per class (color-anchor), crops each generated sign to the warped
template's bbox (matches the real-crop framing), runs the trained SignClassifier, and reports
the per-class top-1 valid rate + mean confidence, split by DIRECT vs PARAMETRIC templates.
Writes reports/qa/signgen_verify.json + a contact sheet (green=valid, red=off-class).

Usage:
  python scripts/detection/verify_signgen.py --classes pl70,w58,p19,il100,ph5 --per-class 20 --device 0
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
from detection.generators.templates import load_template  # noqa: E402
from detection.generators.signgen_i2i import contact_sheet  # noqa: E402
from detection.generators.signgen_controlnet import SignGenControlNet  # noqa: E402
from detection.verifier import SignClassifier  # noqa: E402


def _alpha_bbox(warped_rgba: np.ndarray):
    ys, xs = np.where(warped_rgba[..., 3] > 10)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _border(img: np.ndarray, color, t: int = 6) -> np.ndarray:
    o = img.copy()
    o[:t] = o[-t:] = color
    o[:, :t] = o[:, -t:] = color
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="pl70,w58,p19,il100,ph5")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--marks", default="data/tt100k/tt100k_2021/marks")
    ap.add_argument("--weights", default="data/tt100k/verifier/convnext_signcls.pt")
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=6, help="crops shown per class in the sheet")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/qa/signgen_verify.json")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        sys.exit("--classes vazio")
    if not Path(args.weights).exists():
        sys.exit(f"pesos do verificador ausentes: {args.weights} — rode train_verifier.py antes")

    sub = json.loads(Path("data/tt100k/prepared/subset.json").read_text())
    name2id = {c["name"]: c["id"] for c in sub["classes"]}
    direct = {c["name"] for c in sub["classes"]
              if (Path(args.marks) / f"{c['name']}.png").exists()}

    gen = SignGenControlNet(color_anchor=True, strength=args.strength, steps=args.steps)
    clf = SignClassifier(weights_path=args.weights)
    rng = random.Random(args.seed)

    per_class, rows, GREEN, RED = {}, [], [0, 200, 0], [220, 0, 0]
    for cname in classes:
        if cname not in name2id:
            print(f"[warn] classe {cname} fora do subset — pulando"); continue
        cid = name2id[cname]
        tpl = load_template(cname, args.marks)
        variants = gen.generate(tpl, args.per_class, rng)
        crops = []
        for v in variants:
            bb = _alpha_bbox(v["warped"])
            if not bb:
                print(f"[warn] {cname}: template deformado todo-transparente — usando frame inteiro")
            crops.append(v["image"][bb[1]:bb[3], bb[0]:bb[2]] if bb else v["image"])
        stats = clf.valid_rate(crops, cid, conf_thr=args.conf_thr)
        stats["kind"] = "direct" if cname in direct else "parametric"
        per_class[cname] = stats
        kind = stats["kind"]
        print(f"  {cname:6s} [{kind:10s}] top1={stats['top1_acc']:.2f} "
              f"accept={stats['accept_rate']:.2f} conf={stats['mean_conf']:.2f} (n={stats['n']})")
        # sheet row: template + first `show` crops with verdict border
        row = [(np.where((tpl[..., 3:4] > 10), tpl[..., :3], 128).astype(np.uint8), cname)]
        for c in crops[:args.show]:
            pcid, pconf, _ = clf.predict(c)
            ok = pcid == cid
            note = f"{'OK' if ok else 'X'} {pconf:.2f}"
            row.append((_border(c, GREEN if ok else RED), note))
        rows.append(row)

    # aggregate by kind
    def agg(kind):
        s = [v for v in per_class.values() if v["kind"] == kind]
        n = sum(v["n"] for v in s)
        if not n:
            return {"classes": len(s), "n": 0, "top1_acc": None, "mean_conf": None, "conf_std": None}
        confs = [v["mean_conf"] for v in s]
        return {"classes": len(s), "n": n,
                "top1_acc": round(sum(v["top1_acc"] * v["n"] for v in s) / n, 4),
                "mean_conf": round(float(np.mean(confs)), 4),
                "conf_std": round(float(np.std(confs)), 4)}  # alto std -> instabilidade de domínio

    report = {"conf_thr": args.conf_thr, "strength": args.strength, "per_class": per_class,
              "by_kind": {"direct": agg("direct"), "parametric": agg("parametric")},
              "caveat": ("verificador treinado em crops REAIS; parte da rejeição pode ser "
                         "domain-gap (gerado limpo vs real pequeno/borrado), não classe errada. "
                         "conf_std alto sugere instabilidade de domínio. Ver REQ2.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    sheet = contact_sheet(rows, ["template"] + [f"amostra {i}" for i in range(args.show)])
    sheet_path = Path("reports/qa") / "signgen_verify.png"
    Image.fromarray(sheet).save(sheet_path)
    print(f"\n[verify] direct top1={report['by_kind']['direct']['top1_acc']} | "
          f"parametric top1={report['by_kind']['parametric']['top1_acc']}")
    print(f"-> {args.out}\n-> {sheet_path}")


if __name__ == "__main__":
    main()
