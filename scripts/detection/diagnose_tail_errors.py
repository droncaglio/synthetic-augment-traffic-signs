#!/usr/bin/env python
"""STAGE 0 diagnostic (rigorous): decompose tail failures + PRECISION side with
per-seed PAIRED confidence intervals, reported at an OPERATING confidence.

Pure post-processing of the ALREADY-SAVED per-panorama detections (dets_<split>.json) —
no inference, no GPU, no retraining. Answers the linchpin questions with statistics, not
just means:
  1. Is the tail RECALL-saturated at the OPERATING point (not just conf-floor)?
  2. Is AP-tail limited by PRECISION (tail false positives)?
  3. WHERE do the tail FPs land — on background vs on a look-alike (confuser) sign?
  4. Which arm differences are REAL (paired-seed t-CI excludes 0) vs noise?

Reuses detection.report.load_runs + gts_by_pid (same loading as det_report.py),
detection.diagnose (pure taxonomy), and detection.stats.paired_ci (the study's paired CI).

Usage:
  python scripts/detection/diagnose_tail_errors.py --eval-split test --op-conf 0.25 \
      --arms zero_aug da_only real_duplicate copy_paste signgen_controlnet diffusion_bg
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.budget import budget_tag  # noqa: E402
from detection.report import gts_by_pid, load_runs  # noqa: E402
from detection.stats import paired_ci, ci_excludes_zero  # noqa: E402
from detection.diagnose import (  # noqa: E402
    decompose_run, tail_fp_precision, classify_tail_fps)

DEFAULT_ARMS = ["zero_aug", "da_only", "real_duplicate", "copy_paste",
                "signgen_controlnet", "diffusion_bg"]
# paired contrasts (treatment, baseline) — the questions the study cares about
CONTRASTS = [
    ("da_only", "zero_aug"),                 # does augmentation help at all
    ("real_duplicate", "da_only"),           # oversample real tail vs runtime aug
    ("copy_paste", "real_duplicate"),        # relocate real sign vs in-place oversample
    ("signgen_controlnet", "copy_paste"),    # synthetic appearance vs real relocated
    ("signgen_controlnet", "real_duplicate"),
    ("diffusion_bg", "real_duplicate"),      # background diversity vs in-place oversample
    ("diffusion_bg", "copy_paste"),          # background diversity vs relocation
]


def _filter_conf(dets_by_pid: dict, conf: float) -> dict:
    if conf <= 0:
        return dets_by_pid
    return {pid: [d for d in dl if d.get("conf", 0.0) >= conf]
            for pid, dl in dets_by_pid.items()}


def _seed_metrics(gts: dict, dets_by_pid: dict, tail_ids: set, n_gt: int,
                  op_conf: float, iou_hit: float, iou_near: float) -> dict:
    """All per-seed scalars for one run (used for mean±std and paired deltas)."""
    def cat_counts(dbp):
        c = Counter()
        for r in decompose_run(gts, dbp, tail_ids, iou_hit=iou_hit, iou_near=iou_near):
            c[r["category"]] += 1
        return c
    floor = cat_counts(dets_by_pid)
    op = cat_counts(_filter_conf(dets_by_pid, op_conf))
    prec = tail_fp_precision(gts, dets_by_pid, tail_ids, conf=op_conf, iou_hit=iou_hit)
    fpo = classify_tail_fps(gts, dets_by_pid, tail_ids, conf=op_conf,
                            iou_hit=iou_hit, iou_near=iou_near)
    tot = max(1, sum(floor.values()))
    return {
        "hit_floor_pct": 100.0 * floor["HIT"] / tot,
        "miss_floor_pct": 100.0 * floor["MISS"] / tot,
        "recall_op_pct": 100.0 * op["HIT"] / tot,     # HIT at the operating threshold
        "miss_op_pct": 100.0 * op["MISS"] / tot,
        "cls_op_pct": 100.0 * op["CLS"] / tot,
        "tp": float(prec["tp"]), "fp": float(prec["fp"]), "precision": prec["precision"],
        "fp_on_sign": float(fpo["on_sign"]), "fp_on_bg": float(fpo["on_bg"]),
        "fp_ambiguous": float(fpo["ambiguous"]), "fires_on": fpo["fires_on"],
    }


def _ci(vals: list[float]) -> str:
    if len(vals) < 2:
        return f"{vals[0]:.1f}" if vals else "–"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--op-conf", type=float, default=0.25,
                    help="operating confidence for the precision/recall side (default 0.25)")
    ap.add_argument("--iou-hit", type=float, default=0.5)
    ap.add_argument("--iou-near", type=float, default=0.3)
    ap.add_argument("--out", default="reports/det")
    args = ap.parse_args()

    prepared = Path(args.prepared)
    subset = json.loads((prepared / "subset.json").read_text())
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    id2name = {c["id"]: c["name"] for c in subset["classes"]}
    tail_names = subset["by_tier"]["tail"]
    tail_ids = {subset_ids[n] for n in tail_names}
    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines() if l.strip())}
    splits = json.loads((prepared / "splits.json").read_text())
    pids = splits[args.eval_split]
    gts = gts_by_pid(records, pids, subset_ids, 2048)
    n_gt = sum(1 for lst in gts.values() for g in lst if g["class_id"] in tail_ids)
    bm = budget_tag(args.K)
    print(f"tail={tail_names}  n_tail_gt={n_gt}  op_conf={args.op_conf}\n")

    # per arm: {seed: seed_metrics}
    per_seed: dict[str, dict[int, dict]] = {}
    fires_on_tot: dict[str, Counter] = {}
    for arm in args.arms:
        runs = load_runs(args.project, arm, args.seeds, bm, eval_split=args.eval_split)
        if not runs:
            print(f"[skip] {arm}: no runs")
            continue
        any_dets = next(iter(runs.values()))[1]
        if len(set(any_dets) & set(pids)) == 0:
            print(f"[WARN] {arm}: dets do not overlap '{args.eval_split}'; skipping.")
            continue
        sm, fo = {}, Counter()
        for s, (_hl, dbp) in runs.items():
            m = _seed_metrics(gts, dbp, tail_ids, n_gt, args.op_conf, args.iou_hit, args.iou_near)
            for cls, k in m.pop("fires_on").items():
                fo[id2name.get(cls, str(cls))] += k
            sm[s] = m
        per_seed[arm] = sm
        fires_on_tot[arm] = fo
        # console: mean±std for the headline metrics
        def col(key):
            return _ci([sm[s][key] for s in sm])
        print(f"[{arm:20}] n={len(sm)}  recall@op={col('recall_op_pct')}%  "
              f"FP={col('fp')}  prec={col('precision')}  "
              f"FP_bg={col('fp_on_bg')} FP_sign={col('fp_on_sign')}")

    _write(args, tail_names, n_gt, per_seed, fires_on_tot)


def _paired_rows(per_seed: dict, metric: str) -> list[dict]:
    rows = []
    for treat, base in CONTRASTS:
        if treat not in per_seed or base not in per_seed:
            continue
        common = sorted(set(per_seed[treat]) & set(per_seed[base]))
        if len(common) < 2:
            continue
        deltas = [per_seed[treat][s][metric] - per_seed[base][s][metric] for s in common]
        ci = paired_ci(deltas)
        rows.append({"treat": treat, "base": base, "n": len(common),
                     "mean": ci["mean"], "lo": ci["ci_low"], "hi": ci["ci_high"],
                     "sig": ci_excludes_zero(ci), "n_pos": ci["n_pos"]})
    return rows


def _write(args, tail_names, n_gt, per_seed, fires_on_tot) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = list(per_seed)

    def stat(arm, key):
        vals = [per_seed[arm][s][key] for s in per_seed[arm]]
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

    # JSON: full per-seed + paired contrasts (machine-readable, auditable)
    metrics_for_ci = ["hit_floor_pct", "recall_op_pct", "miss_op_pct", "cls_op_pct",
                      "tp", "fp", "precision", "fp_on_bg", "fp_on_sign"]
    j = {"eval_split": args.eval_split, "op_conf": args.op_conf, "n_tail_gt": n_gt,
         "tail_classes": tail_names,
         "per_arm": {a: {k: {"mean": round(stat(a, k)[0], 3), "std": round(stat(a, k)[1], 3)}
                         for k in metrics_for_ci} for a in arms},
         "per_seed": {a: {str(s): per_seed[a][s] for s in per_seed[a]} for a in arms},
         "contrasts": {m: _paired_rows(per_seed, m) for m in metrics_for_ci},
         "fires_on": {a: dict(fires_on_tot[a].most_common(8)) for a in arms}}
    (out / "tail_error_decomposition.json").write_text(json.dumps(j, indent=2))

    L = [f"# Tail error decomposition (rigoroso) — eval={args.eval_split}, op_conf={args.op_conf}\n",
         f"Tail: {', '.join(tail_names)} · n_tail_gt={n_gt} · 7 seeds · CI = t-pareado 95% "
         "(mesmo seed = mesma alocação). ✓ = IC exclui 0.\n",
         "## Por braço (média ± dp entre seeds)\n",
         "| arm | n | recall@floor% | recall@op% | MISS@op% | FP | precisão | FP_fundo | FP_placa |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        n = len(per_seed[a])
        def ms(k, d=1):
            m, s = stat(a, k)
            return f"{m:.{d}f}±{s:.{d}f}"
        L.append(f"| {a} | {n} | {ms('hit_floor_pct')} | {ms('recall_op_pct')} | "
                 f"{ms('miss_op_pct')} | {ms('fp')} | {ms('precision', 3)} | "
                 f"{ms('fp_on_bg')} | {ms('fp_on_sign')} |")

    # paired contrasts with CI for the decisive metrics
    def ctable(metric, title, unit=""):
        rows = _paired_rows(per_seed, metric)
        block = [f"\n### {title}\n", "| treatment − baseline | n | Δ | IC95 | sig | seeds+ |",
                 "|---|---|---|---|---|---|"]
        for r in rows:
            block.append(f"| {r['treat']} − {r['base']} | {r['n']} | {r['mean']:+.2f}{unit} "
                         f"| [{r['lo']:+.2f}, {r['hi']:+.2f}] | {'✓' if r['sig'] else '–'} "
                         f"| {r['n_pos']}/{r['n']} |")
        return block
    L += ["\n## Contrastes pareados por-seed (o que é sinal vs ruído)\n"]
    L += ctable("recall_op_pct", "Recall no ponto de operação (Δ pontos %)", "pp")
    L += ctable("fp", "Falsos-positivos da cauda @op (Δ contagem)")
    L += ctable("precision", "Precisão da cauda @op (Δ)")
    L += ctable("fp_on_bg", "FP em FUNDO @op (Δ contagem)")
    L += ctable("fp_on_sign", "FP em PLACA-confusora @op (Δ contagem)")

    # FP origin split (per arm) — decides the remedy
    L += ["\n## Origem dos FP da cauda @op — fundo vs placa-confusora\n",
          "Se FP predomina em PLACA-confusora → discriminação fina / confusion-aware. Se em "
          "FUNDO → hard-negative de fundo. (média entre seeds)\n",
          "| arm | FP_total | FP_fundo | FP_placa | FP_ambíguo | %placa |",
          "|---|---|---|---|---|---|"]
    for a in arms:
        bg = stat(a, "fp_on_bg")[0]; sg = stat(a, "fp_on_sign")[0]
        amb = stat(a, "fp_ambiguous")[0] if "fp_ambiguous" in per_seed[a][next(iter(per_seed[a]))] else 0.0
        tot = bg + sg + amb
        L.append(f"| {a} | {tot:.1f} | {bg:.1f} | {sg:.1f} | {amb:.1f} | "
                 f"{(100*sg/tot if tot else 0):.0f}% |")
    L += ["\n## Em quais classes a cauda dispara FP (fires_on, soma seeds)\n"]
    for a in arms:
        fo = fires_on_tot.get(a)
        if fo:
            L.append(f"- **{a}**: " + ", ".join(f"{k}({v})" for k, v in fo.most_common(6)))

    L += ["\n## Leitura (com a estatística)\n",
          "- **Recall@op**: se os contrastes entre braços aumentados NÃO excluem 0 → recall "
          "saturado (aug liga o ganho, sofisticação não move). Reportar SEMPRE no op_conf.\n",
          "- **FP / precisão**: contrastes com ✓ são reais; ranking dentro do cluster de "
          "foreground só vale se o IC excluir 0.\n",
          "- **diffusion_bg vs real_duplicate**: isola diversidade de FUNDO (ambos in-place) — "
          "se ΔFP ✓ negativo, o lever é fundo/contexto, NÃO adicionar/remover foreground.\n",
          "- **Origem dos FP** aponta o remédio: placa-confusora → confusion-aware; fundo → "
          "hard-negative de fundo.\n"]
    (out / "tail_error_decomposition.md").write_text("\n".join(L))
    print(f"\n-> {out}/tail_error_decomposition.md  (+ .json com per-seed e contrastes)")


if __name__ == "__main__":
    main()
