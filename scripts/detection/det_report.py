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
from detection.stats import (  # noqa: E402
    bootstrap_delta_ap_multi, make_macro_metric, ci_excludes_zero, paired_ci)

ARMS = ["zero_aug", "da_only", "real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg"]
# contrasts (treatment vs reference). Order = the cost ladder + the two baseline steps.
# The headline is da_only vs zero_aug (does augmenting help at all) and the ladder rungs
# vs real_duplicate (does context novelty/sophistication add anything over novelty-zero).
CONTRASTS = [
    ("da_only", "zero_aug"),               # augmentation vs none (expected big)
    ("real_duplicate", "da_only"),         # allocation+dup vs runtime aug only
    ("bg_photometric", "real_duplicate"),  # photometric context novelty vs novelty-zero
    ("copy_paste", "real_duplicate"),      # real relocated context vs novelty-zero
    ("diffusion_bg", "real_duplicate"),    # synthetic context vs novelty-zero (key)
    ("diffusion_bg", "copy_paste"),        # synthetic vs cheap real context
    ("diffusion_bg", "bg_photometric"),    # synthetic vs cheap photometric context
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=0,
                    help="0 (default) = INSTANTÂNEO: só contrastes pareados-por-seed "
                         "(t CI, do AP já salvo). >0 liga o bootstrap por panorama "
                         "(lento, com progresso) — use p/ o número final do paper.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)),
                    help="seeds to aggregate (default 0..6); matches the grid config")
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

    # per-arm aggregate + collect runs (keyed by seed)
    runs_by_arm, agg = {}, {}
    for arm in ARMS:
        runs = load_runs(args.project, arm, args.seeds, bm, eval_split=args.eval_split)
        runs_by_arm[arm] = runs
        # loud guard: dets must live on the SAME split we are evaluating. A disjoint
        # pid set (e.g. a val-evaluated dets.json vs a test eval) silently zeroes every
        # paired bootstrap — fail visibly instead.
        if runs:
            any_dets = next(iter(runs.values()))[1]
            overlap = len(set(any_dets) & set(pids))
            if overlap == 0:
                print(f"[WARN] arm '{arm}': dets panoramas do not overlap the "
                      f"'{args.eval_split}' split ({len(any_dets)} det pids, 0 overlap). "
                      f"Run the eval-only pass on --eval-split {args.eval_split} first "
                      f"(scripts/detection/eval_runs.py); bootstrap CIs will be 0 otherwise.")
        a = aggregate_arm(runs.values())
        if a:
            agg[arm] = a

    # --- INSTANTÂNEO: contrastes pareados-por-seed a partir do AP já salvo (sem resample).
    # Pair on the INTERSECTION of seeds present in BOTH arms (partial/resumed grids).
    metric_keys = [("ap_tail", "ap_tail"), ("ap_small", "ap_small_macro")]
    contrasts = []
    print(f"\n=== Contrastes pareados-por-seed (t CI 95%) — eval={args.eval_split} ===")
    for treat, base in CONTRASTS:
        tr, ba = runs_by_arm.get(treat, {}), runs_by_arm.get(base, {})
        common = sorted(set(tr) & set(ba))
        if not common:
            continue
        row = {"treatment": treat, "baseline": base, "n_seeds": len(common), "seeds": common}
        line = f"  {treat+' - '+base:32}"
        for label, hlkey in metric_keys:
            deltas = [tr[s][0].get(hlkey, 0.0) - ba[s][0].get(hlkey, 0.0) for s in common]
            ci = paired_ci(deltas)
            sig = ci_excludes_zero(ci)
            row[label] = {"delta_mean": round(ci["mean"], 4),
                          "ci": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
                          "excludes_zero": sig, "n_pos": ci["n_pos"]}
            line += f"  {label.split('_')[1]}:{ci['mean']:+.4f}[{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}]{'✓' if sig else '–'}"
        print(line)
        contrasts.append(row)

    # --- OPCIONAL: bootstrap por panorama (lento, com progresso) só se --n-boot > 0.
    if args.n_boot > 0:
        metrics = {"ap_tail": make_macro_metric(subset, tier="tail"),
                   "ap_small": make_macro_metric(subset)}
        for row in contrasts:
            tr, ba = runs_by_arm[row["treatment"]], runs_by_arm[row["baseline"]]
            base_runs = [ba[s][1] for s in row["seeds"]]
            treat_runs = [tr[s][1] for s in row["seeds"]]
            tag = f"{row['treatment']} vs {row['baseline']}"
            print(f"[bootstrap {args.n_boot}] {tag} ...", flush=True)
            b = bootstrap_delta_ap_multi(
                base_runs, treat_runs, gts, class_names, pids, metrics,
                n_boot=args.n_boot, seed=0,
                on_progress=lambda d, t, _t=tag: print(f"    {_t}: {d}/{t}", flush=True))
            for label in ("ap_tail", "ap_small"):
                row[label]["boot_ci"] = [round(b[label]["ci_low"], 4), round(b[label]["ci_high"], 4)]
                row[label]["boot_excludes_zero"] = ci_excludes_zero(b[label])

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
    L += ["\n## Primary contrasts — paired-seed ΔAP (t CI 95%)\n",
          "sig = IC exclui 0. seeds+ = em quantas seeds o tratamento > baseline.\n",
          "| treatment vs baseline | n | ΔAP-tail [CI] | sig | ΔAP@small [CI] | sig | tail seeds+ |",
          "|---|---|---|---|---|---|---|"]
    for c in report["contrasts"]:
        t, s = c["ap_tail"], c["ap_small"]
        L.append(f"| {c['treatment']} vs {c['baseline']} | {c['n_seeds']} "
                 f"| {t['delta_mean']} {t['ci']} | {'✓' if t['excludes_zero'] else '–'} "
                 f"| {s['delta_mean']} {s['ci']} | {'✓' if s['excludes_zero'] else '–'} "
                 f"| {t.get('n_pos', '?')}/{c['n_seeds']} |")
    if report["contrasts"] and "boot_ci" in report["contrasts"][0].get("ap_tail", {}):
        L += ["\n## Bootstrap CI (reamostragem de panoramas)\n",
              "| treatment vs baseline | ΔAP-tail boot[CI] | sig | ΔAP@small boot[CI] | sig |",
              "|---|---|---|---|---|"]
        for c in report["contrasts"]:
            t, s = c["ap_tail"], c["ap_small"]
            L.append(f"| {c['treatment']} vs {c['baseline']} "
                     f"| {t['boot_ci']} | {'✓' if t['boot_excludes_zero'] else '–'} "
                     f"| {s['boot_ci']} | {'✓' if s['boot_excludes_zero'] else '–'} |")
    L += ["\n> ΔAP × custo (fronteira de Pareto): custo de geração por braço a preencher "
          "(GPU-h por amostra aceita) — auditoria separada.\n"]
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
