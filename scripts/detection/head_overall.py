#!/usr/bin/env python
"""Head vs. overall AP robustness: does augmenting ONLY the tail hurt the frequent
classes? Computes ΔAP (vs. a baseline arm, default da_only) of the macro AP over the
HEAD tier (frequent classes) and over ALL classes with test GT (overall), with paired
95% CIs. Read-only; mirrors recompute_metrics.per_class_ap EXACTLY (all-size AP@0.5).

WHY: the synthetic budget is spent on rare (tail) classes only, so a fair paper must
show the head is not silently degraded. This produces the numbers behind the paper's
"Does the head suffer?" table (head/overall, both detectors).

Usage (per dataset-detector cell, on the box where dets live):
  python scripts/detection/head_overall.py --project experiments_dfg/dfg \
      --prepared data/dfg/prepared --seeds $(seq 0 13)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "detection"))

from scipy import stats as ss  # noqa: E402

from detection.report import gts_by_pid, load_runs           # noqa: E402
from detection.budget import budget_tag                        # noqa: E402
from detection.stats import paired_ci                          # noqa: E402
from detection.ap_by_size import Detection, GroundTruth, PANORAMA_SIZE_DEFAULT  # noqa: E402
from recompute_metrics import per_class_ap                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--base", default="da_only")
    ap.add_argument("--arms", nargs="+", default=[
        "real_duplicate", "photometric_full", "copy_paste",
        "diffusion_bg", "signgen_controlnet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(14)))
    ap.add_argument("--panorama-size", type=int, default=PANORAMA_SIZE_DEFAULT)
    args = ap.parse_args()

    prep = Path(args.prepared)
    subset = json.loads((prep / "subset.json").read_text())
    names = subset["names"]
    sid = {c["name"]: c["id"] for c in subset["classes"]}
    head_ids = [sid[n] for n in subset["by_tier"]["head"]]
    recs = {r["id"]: r for r in
            (json.loads(l) for l in (prep / "panoramas.jsonl").read_text().splitlines()
             if l.strip())}
    split = json.loads((prep / "splits.json").read_text())[args.eval_split]
    bm = budget_tag(args.K)
    area = float(args.panorama_size) ** 2

    gp = gts_by_pid(recs, split, sid, size=args.panorama_size)
    idof = {p: i for i, p in enumerate(gp)}
    gts = [GroundTruth(idof[p], o["class_id"], *o["box"])
           for p, lst in gp.items() for o in lst]
    overall_ids = sorted({o["class_id"] for lst in gp.values() for o in lst})

    def macro(pc, ids):
        v = [pc[c]["all"] for c in ids if c in pc and pc[c]["all"] == pc[c]["all"]]
        return st.mean(v) if v else float("nan")

    def per_seed(arm):
        out = {}
        for s, (_h, dp) in load_runs(args.project, arm, args.seeds, bm,
                                     eval_split=args.eval_split).items():
            dl = [Detection(idof[p], d["class_id"], d["conf"], *d["box"])
                  for p, x in dp.items() if p in idof for d in x]
            pc = per_class_ap(dl, gts, len(names), area)
            out[s] = {"head": macro(pc, head_ids), "overall": macro(pc, overall_ids)}
        return out

    base = per_seed(args.base)
    print(f"# head/overall — project={args.project}  base={args.base}")
    print(f"  head classes={len(head_ids)}  overall(test-GT) classes={len(overall_ids)}  seeds={len(base)}")
    print("| arm | tier | ΔAP | IC95 | p |")
    print("|---|---|---|---|---|")
    for arm in args.arms:
        ps = per_seed(arm)
        common = sorted(set(ps) & set(base))
        for tier in ("head", "overall"):
            d = [ps[s][tier] - base[s][tier] for s in common
                 if ps[s][tier] == ps[s][tier] and base[s][tier] == base[s][tier]]
            ci = paired_ci(d)
            p = float(ss.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else float("nan")
            sig = "**" if (ci["ci_low"] > 0 or ci["ci_high"] < 0) else ""
            print(f"| {arm} | {tier} | {sig}{ci['mean']:+.4f}{sig} | "
                  f"[{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}] | {p:.3f} |")


if __name__ == "__main__":
    main()
