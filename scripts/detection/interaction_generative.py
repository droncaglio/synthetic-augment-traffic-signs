#!/usr/bin/env python
"""Cross-detector INTERACTION test for the generative arm — the analog of the Finding-2
interaction that bulletproofed cheap-real (nano-only).

For a given dataset we take the generative contrast (default diffusion_bg - real_duplicate)
on BOTH detectors, pair by seed, and test whether the effect DIFFERS between detectors:

    interaction_s = (treat - base)_{11s, seed} - (treat - base)_{nano, seed}

A significantly positive interaction means generation helps the large detector MORE than the
small one -- the generative half of the double dissociation. Mirrors recompute_metrics.py's
all-size macro AP-tail EXACTLY (same per_class_ap / macro_tail). Read-only.

Usage:
  python scripts/detection/interaction_generative.py \
      --nano-project experiments_dfg/dfg --small-project experiments_dfg_11s/dfg \
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
from recompute_metrics import per_class_ap, macro_tail, paired_p  # noqa: E402


def seed_macro_by_project(project, arm, seeds, bm, eval_split, id_of, gt_list, n_names, image_area):
    """{seed -> all-size macro AP-tail} for one arm in one project."""
    out = {}
    runs = load_runs(project, arm, seeds, bm, eval_split=eval_split)
    for s, (_hl, dets_pid) in runs.items():
        det_list = [Detection(id_of[pid], d["class_id"], d["conf"], *d["box"])
                    for pid, dl in dets_pid.items() if pid in id_of for d in dl]
        pc = per_class_ap(det_list, gt_list, n_names, image_area)
        ap, _ = macro_tail(pc, TAIL_IDS, "all")
        out[s] = ap
    return out


def contrast_ci(a: dict, b: dict):
    """paired (a-b) over common seeds -> ci dict + p."""
    common = sorted(set(a) & set(b))
    deltas = [a[s] - b[s] for s in common if a[s] == a[s] and b[s] == b[s]]
    ci = paired_ci(deltas)
    return deltas, ci, paired_p(deltas)


TAIL_IDS: list = []


def main() -> None:
    global TAIL_IDS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nano-project", required=True)
    ap.add_argument("--small-project", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--treat", default="diffusion_bg")
    ap.add_argument("--base", default="real_duplicate")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(14)))
    ap.add_argument("--panorama-size", type=int, default=PANORAMA_SIZE_DEFAULT)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    subset = json.loads((prepared / "subset.json").read_text())
    names = subset["names"]
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    tail_names = subset["by_tier"]["tail"]
    TAIL_IDS = [subset_ids[n] for n in tail_names]

    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines()
                if l.strip())}
    split_ids = json.loads((prepared / "splits.json").read_text())[args.eval_split]
    bm = budget_tag(args.K)
    image_area = float(args.panorama_size) ** 2

    gts_pid = gts_by_pid(records, split_ids, subset_ids, size=args.panorama_size)
    id_of = {pid: i for i, pid in enumerate(gts_pid)}
    gt_list = [GroundTruth(id_of[pid], o["class_id"], *o["box"])
               for pid, lst in gts_pid.items() for o in lst]

    def load(project, arm):
        return seed_macro_by_project(project, arm, args.seeds, bm, args.eval_split,
                                     id_of, gt_list, len(names), image_area)

    nano_t = load(args.nano_project, args.treat)
    nano_b = load(args.nano_project, args.base)
    small_t = load(args.small_project, args.treat)
    small_b = load(args.small_project, args.base)

    _, nano_ci, nano_p = contrast_ci(nano_t, nano_b)
    _, small_ci, small_p = contrast_ci(small_t, small_b)

    # interaction: per-seed (small effect) - (nano effect), paired by seed
    common = sorted(set(nano_t) & set(nano_b) & set(small_t) & set(small_b))
    inter = [(small_t[s] - small_b[s]) - (nano_t[s] - nano_b[s]) for s in common]
    ici = paired_ci(inter)
    ip = paired_p(inter)

    print(f"\n=== {args.treat} - {args.base}  ({args.nano_project}  vs  {args.small_project}) ===")
    print(f"seeds paired: {len(common)}")
    print(f"  nano  effect : {nano_ci['mean']:+.4f}  [{nano_ci['ci_low']:+.4f},{nano_ci['ci_high']:+.4f}]  p={nano_p:.4f}")
    print(f"  11s   effect : {small_ci['mean']:+.4f}  [{small_ci['ci_low']:+.4f},{small_ci['ci_high']:+.4f}]  p={small_p:.4f}")
    print(f"  INTERACTION (11s - nano): {ici['mean']:+.4f}  [{ici['ci_low']:+.4f},{ici['ci_high']:+.4f}]  p={ip:.4f}"
          f"   {'** differs by capacity **' if ip < 0.05 else '(ns)'}")


if __name__ == "__main__":
    main()
