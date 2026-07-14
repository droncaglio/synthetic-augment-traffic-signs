#!/usr/bin/env python
"""AP por CLASSE × braço (foco na cauda) — instantâneo, lê os ap_report_<split>.json.

Para cada classe do subset mostra o AP@small médio (sobre seeds) de cada braço, ordenado
head→tail, com o tier e a contagem de instâncias no treino. Responde "onde o detector mais
sofre (cauda), alguma técnica ganhou?". Também marca o melhor braço por classe rara.

Uso (workstation): python scripts/detection/class_report.py --eval-split test
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from detection.budget import budget_tag           # noqa: E402
from detection.report import load_runs             # noqa: E402

ARMS = ["zero_aug", "da_only", "real_duplicate", "bg_photometric", "bg_photometric_mask", "copy_paste", "copy_paste_mask", "diffusion_bg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--out", default="reports/det")
    args = ap.parse_args()

    subset = json.loads((Path(args.prepared) / "subset.json").read_text())
    meta = {c["name"]: c for c in subset["classes"]}
    order = [c["name"] for c in subset["classes"]]  # já vem head->tail por instances
    bm = budget_tag(args.K)

    # arm -> {class -> mean AP over seeds}
    per_arm = {}
    for arm in ARMS:
        runs = load_runs(args.project, arm, args.seeds, bm, eval_split=args.eval_split)
        acc = {n: [] for n in order}
        for _s, (hl, _dets) in runs.items():
            pcs = hl.get("per_class_small", {})
            for n in order:
                v = pcs.get(n)
                if v is not None and v == v:  # not None / not NaN
                    acc[n].append(v)
        per_arm[arm] = {n: (mean(acc[n]) if acc[n] else float("nan")) for n in order}

    content = ["real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg"]
    # tabela md
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lines = [f"# AP@small por classe × braço — eval={args.eval_split}\n",
             "Média sobre seeds. tier H/M/T; inst = instâncias no treino. "
             "Δ = melhor braço de conteúdo − zero_aug.\n",
             "| classe | tier | inst | " + " | ".join(ARMS) + " | melhor | Δ vs zero |",
             "|---|:--:|--:|" + "|".join(["--:"] * len(ARMS)) + "|:--:|--:|"]
    tail_best = {}
    for n in order:
        tier = meta[n].get("tier", "?")[:1].upper()
        inst = meta[n].get("instances", "?")
        vals = {a: per_arm[a][n] for a in ARMS}
        best_arm = max(content, key=lambda a: (vals[a] if vals[a] == vals[a] else -1))
        z = vals["zero_aug"]
        delta = (vals[best_arm] - z) if (vals[best_arm] == vals[best_arm] and z == z) else float("nan")
        if tier == "T":
            tail_best[best_arm] = tail_best.get(best_arm, 0) + 1
        cells = " | ".join(f"{vals[a]:.3f}" if vals[a] == vals[a] else "—" for a in ARMS)
        lines.append(f"| `{n}` | {tier} | {inst} | {cells} | {best_arm} "
                     f"| {delta:+.3f} |" if delta == delta else
                     f"| `{n}` | {tier} | {inst} | {cells} | {best_arm} | — |")
    # resumo cauda
    lines += ["\n## Resumo cauda (tier T)\n",
              "Em quantas classes de cauda cada braço de conteúdo foi o melhor:\n"]
    for a in content:
        lines.append(f"- **{a}**: {tail_best.get(a, 0)}")
    (out / "class_report.md").write_text("\n".join(lines))
    print(f"[ok] {out}/class_report.md")
    print("\n[resumo cauda] melhor braço por classe de cauda:",
          {a: tail_best.get(a, 0) for a in content})


if __name__ == "__main__":
    main()
