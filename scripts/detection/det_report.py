#!/usr/bin/env python
"""Aggregate the grid runs into the paper's report: per-arm AP, paired bootstrap CIs
for the primary contrasts, and a ΔAP × cost frontier.

For each arm it reads every seed's ap_report.json (headline) + dets.json (per-panorama
detections). The bootstrap resamples the eval panoramas and recomputes the SAME macro
metric the headline reports (make_macro_metric) — so the CI validates the reported number.

Usage:
  python scripts/detection/det_report.py --project experiments/tt100k \
      --prepared data/tt100k/prepared --eval-split val [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.budget import budget_tag  # noqa: E402
from detection.report import gts_by_pid, load_runs, aggregate_arm  # noqa: E402
from detection.stats import bootstrap_delta_ap, make_macro_metric, ci_excludes_zero  # noqa: E402

ARMS = ["zero_aug", "da_only", "real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg"]
# primary contrasts (treatment vs reference) — headline of the paper
CONTRASTS = [("diffusion_bg", "copy_paste"), ("diffusion_bg", "bg_photometric"),
             ("copy_paste", "real_duplicate"), ("bg_photometric", "real_duplicate")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/det")
    args = ap.parse_args()

    prepared = Path(args.prepared)
    subset = json.loads((prepared / "subset.json").read_text())
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    class_names = subset["names"]
    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines() if l.strip())}
    splits = json.loads((prepared / "splits.json").read_text())
    pids = splits[args.eval_split]
    gts = gts_by_pid(records, pids, subset_ids, 2048)
    bm = budget_tag(args.K)

    # per-arm aggregate + collect runs
    runs_by_arm, agg = {}, {}
    for arm in ARMS:
        runs = load_runs(args.project, arm, range(7), bm)  # up to 7 seeds
        runs_by_arm[arm] = runs
        a = aggregate_arm(runs)
        if a:
            agg[arm] = a

    # paired bootstrap CIs for the primary contrasts (metric = reported macro)
    metric_tail = make_macro_metric(subset, tier="tail")
    metric_small = make_macro_metric(subset)  # all subset classes, small bucket
    contrasts = []
    for treat, base in CONTRASTS:
        tr, ba = runs_by_arm.get(treat, []), runs_by_arm.get(base, [])
        if not tr or not ba:
            continue
        n = min(len(tr), len(ba))
        base_runs = [d for _, d in ba[:n]]
        treat_runs = [d for _, d in tr[:n]]
        row = {"treatment": treat, "baseline": base, "n_seeds": n}
        for label, metric in [("ap_tail", metric_tail), ("ap_small", metric_small)]:
            b = bootstrap_delta_ap(base_runs, treat_runs, gts, class_names, pids,
                                   metric=metric, n_boot=args.n_boot, seed=0)
            row[label] = {"delta_mean": round(b["mean"], 4),
                          "ci": [round(b["ci_low"], 4), round(b["ci_high"], 4)],
                          "excludes_zero": ci_excludes_zero(b)}
        contrasts.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"eval_split": args.eval_split, "K": args.K, "per_arm": agg, "contrasts": contrasts}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _write_md(out / "report.md", report)
    print(f"-> {out}/report.md  (arms with runs: {list(agg)})")


def _write_md(path, report):
    L = [f"# Detection grid report — eval={report['eval_split']}, K={report['K']}\n",
         "## Per-arm AP (mean ± std over seeds)\n",
         "| arm | n | AP-tail | AP@small(macro) |", "|---|---|---|---|"]
    for arm, a in report["per_arm"].items():
        L.append(f"| {arm} | {a['n_seeds']} | {a['ap_tail_mean']}±{a['ap_tail_std']} "
                 f"| {a['ap_small_mean']}±{a['ap_small_std']} |")
    L += ["\n## Primary contrasts — paired bootstrap ΔAP (CI 95%)\n",
          "| treatment vs baseline | n | ΔAP-tail [CI] | sig | ΔAP@small [CI] | sig |",
          "|---|---|---|---|---|---|"]
    for c in report["contrasts"]:
        t, s = c["ap_tail"], c["ap_small"]
        L.append(f"| {c['treatment']} vs {c['baseline']} | {c['n_seeds']} "
                 f"| {t['delta_mean']} {t['ci']} | {'✓' if t['excludes_zero'] else '–'} "
                 f"| {s['delta_mean']} {s['ci']} | {'✓' if s['excludes_zero'] else '–'} |")
    L += ["\n> ΔAP × custo (fronteira de Pareto): custo de geração por braço a preencher "
          "(GPU-h por amostra aceita) — auditoria separada.\n"]
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
