#!/usr/bin/env python
"""Recompute AP-tail in ALL-SIZE + per size-band, with effective class counts, and the
full contrast family with Holm-adjusted p-values.

WHY (methodology review, 2026-07-23): the published headline AP-tail is *small-bucket only*,
which on the full-201 test tail covers 125/402 = 31% of tail GT (medium 227 dominates). This
recomputes the tail metric over ALL sizes (and per band) so we know whether the ranking
survives, reports the EFFECTIVE number of non-NaN classes per variant (the small-only macro
silently drops the 7 tail classes with 0 small GT -> denominator 35, not 42), and applies
**Holm** across a frozen contrast family (the report code never did; "~5 survive Holm" was
never computed).

NON-DESTRUCTIVE: reads existing dets_<split>.json (via report.load_runs), writes only NEW files
  reports/det/metrics_allsize_v2.{json,md}
  reports/det/contrasts_holm_v2.{json,md}
It never touches ap_report_test.json / dets_test.json / weights / tiles.

Matching mirrors ap_by_size.compute_ap_by_size EXACTLY (greedy same-class IoU>=0.5, TP bucket =
matched-GT bucket, FP bucket = prediction bucket) so per-band numbers are comparable to the
published ones; the only addition is the pooled ALL-SIZE AP per class.

Usage (on the box where dets live):
  python scripts/detection/recompute_metrics.py --project experiments_full201/tt100k \
      --eval-split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from scipy import stats as _sstats  # noqa: E402

from detection.report import gts_by_pid, load_runs                       # noqa: E402
from detection.budget import budget_tag                                   # noqa: E402
from detection.stats import paired_ci                                     # noqa: E402
from detection.ap_by_size import (                                        # noqa: E402
    Detection, GroundTruth, _iou, _ap_from_pr, _assign_bucket, _px_area,
    PANORAMA_SIZE_DEFAULT,
)

BANDS = ("all", "small", "medium", "large")

# Frozen contrast family (the ones the paper/doc report + the ladder rungs present in full-201).
# Only 7 arms exist in the full-201 grid: zero_aug, da_only, real_duplicate, copy_paste,
# photometric_full, diffusion_bg, signgen_controlnet.
FAMILY = [
    ("da_only", "zero_aug"),                  # augmentation vs none
    ("real_duplicate", "da_only"),            # oversampling vs runtime-aug only
    ("photometric_full", "da_only"),          # HEADLINE contrast (was NOT in det_report code)
    ("copy_paste", "da_only"),                # real relocated vs baseline
    ("diffusion_bg", "da_only"),              # synthetic context vs baseline
    ("signgen_controlnet", "da_only"),        # synthetic appearance vs baseline
    ("photometric_full", "real_duplicate"),   # perturb sign vs pure oversampling
    ("diffusion_bg", "real_duplicate"),       # synthetic context vs oversampling (key)
    ("signgen_controlnet", "real_duplicate"),  # synthetic appearance vs oversampling (key)
    ("signgen_controlnet", "copy_paste"),     # synthetic sign vs real relocated (paired 1:1)
]


def per_class_ap(dets: list, gts: list, n_classes: int, image_area: float) -> dict:
    """Per-class AP@0.5 in ALL-SIZE and per band. Mirrors compute_ap_by_size matching."""
    gt_index: dict = defaultdict(list)
    n_all: Counter = Counter()
    n_bkt: dict = defaultdict(int)
    for gt in gts:
        gt_index[(gt.class_id, gt.image_id)].append(gt)
        n_all[gt.class_id] += 1
        n_bkt[(gt.class_id, _assign_bucket(_px_area(gt.w, gt.h, image_area)))] += 1

    matched: dict = defaultdict(set)
    tp_all: dict = defaultdict(list)
    tp_bkt: dict = defaultdict(list)
    for det in sorted(dets, key=lambda d: -d.confidence):
        cid, iid = det.class_id, det.image_id
        here = gt_index.get((cid, iid), [])
        ms = matched[(cid, iid)]
        best_iou, best = -1.0, -1
        for idx, gt in enumerate(here):
            if idx in ms:
                continue
            v = _iou((det.cx, det.cy, det.w, det.h), (gt.cx, gt.cy, gt.w, gt.h))
            if v > best_iou:
                best_iou, best = v, idx
        if best_iou >= 0.5:
            ms.add(best)
            gm = here[best]
            gbkt = _assign_bucket(_px_area(gm.w, gm.h, image_area))
            tp_all[cid].append((True, det.confidence))
            tp_bkt[(cid, gbkt)].append((True, det.confidence))
        else:
            det_bkt = _assign_bucket(_px_area(det.w, det.h, image_area))
            tp_all[cid].append((False, det.confidence))
            tp_bkt[(cid, det_bkt)].append((False, det.confidence))

    out: dict = {}
    for c in range(n_classes):
        row = {"all": _ap_from_pr(sorted(tp_all[c], key=lambda t: -t[1]), n_all[c]),
               "n_gt_all": n_all[c]}
        for b in ("small", "medium", "large"):
            row[b] = _ap_from_pr(sorted(tp_bkt[(c, b)], key=lambda t: -t[1]), n_bkt[(c, b)])
            row["n_gt_" + b] = n_bkt[(c, b)]
        out[c] = row
    return out


def macro_tail(pc_ap: dict, tail_ids: list, band: str) -> tuple[float, int]:
    """Macro mean of per-class AP over tail classes for a band; drops NaN. Returns (ap, n_eff)."""
    vals = [pc_ap[c][band] for c in tail_ids
            if c in pc_ap and pc_ap[c][band] == pc_ap[c][band]]  # drop NaN
    return (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1). Input order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        val = min(1.0, (m - k) * pvals[i])
        run = max(run, val)  # enforce monotone non-decreasing
        adj[i] = run
    return adj


def paired_p(deltas: list[float]) -> float:
    """Two-sided paired-t p-value for H0: mean(delta)=0."""
    n = len(deltas)
    if n < 2:
        return float("nan")
    res = _sstats.ttest_1samp(deltas, 0.0)
    return float(res.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="experiments_full201/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--arms", nargs="+", default=[
        "zero_aug", "da_only", "real_duplicate", "copy_paste",
        "photometric_full", "diffusion_bg", "signgen_controlnet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--out-dir", default="reports/det")
    ap.add_argument("--panorama-size", type=int, default=PANORAMA_SIZE_DEFAULT)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    subset = json.loads((prepared / "subset.json").read_text())
    names = subset["names"]
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    tail_names = subset["by_tier"]["tail"]
    tail_ids = [subset_ids[n] for n in tail_names]
    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines()
                if l.strip())}
    split_ids = json.loads((prepared / "splits.json").read_text())[args.eval_split]
    bm = budget_tag(args.K)
    image_area = float(args.panorama_size) ** 2

    # GT (shared across arms): {pid:[{class_id,box}]} -> GroundTruth list with image_id=index
    gts_pid = gts_by_pid(records, split_ids, subset_ids, size=args.panorama_size)
    id_of = {pid: i for i, pid in enumerate(gts_pid)}
    gt_list = [GroundTruth(id_of[pid], o["class_id"], *o["box"])
               for pid, lst in gts_pid.items() for o in lst]

    # tail GT size profile (context) — same for every arm
    tail_gt_profile = Counter()
    for gt in gt_list:
        if gt.class_id in tail_ids:
            tail_gt_profile[_assign_bucket(_px_area(gt.w, gt.h, image_area))] += 1

    # per (arm, seed) tail AP for each band + effective class count
    per_seed: dict = {}  # arm -> {seed -> {band -> ap}}
    n_eff: dict = {}     # arm -> {seed -> {band -> n_eff}}
    for arm in args.arms:
        runs = load_runs(args.project, arm, args.seeds, bm, eval_split=args.eval_split)
        per_seed[arm], n_eff[arm] = {}, {}
        for s, (_hl, dets_pid) in runs.items():
            det_list = [Detection(id_of[pid], d["class_id"], d["conf"], *d["box"])
                        for pid, dl in dets_pid.items() if pid in id_of for d in dl]
            pc = per_class_ap(det_list, gt_list, len(names), image_area)
            per_seed[arm][s] = {}
            n_eff[arm][s] = {}
            for band in BANDS:
                apv, ne = macro_tail(pc, tail_ids, band)
                per_seed[arm][s][band] = apv
                n_eff[arm][s][band] = ne

    # per-arm aggregate
    import statistics as st

    def agg(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return {"mean": float("nan"), "std": 0.0, "n": 0}
        return {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals)}

    per_arm = {}
    for arm in args.arms:
        seeds = sorted(per_seed[arm])
        per_arm[arm] = {"n_seeds": len(seeds), "bands": {}}
        for band in BANDS:
            per_arm[arm]["bands"][band] = agg([per_seed[arm][s][band] for s in seeds])
            per_arm[arm]["bands"][band]["n_eff_classes"] = (
                int(st.median([n_eff[arm][s][band] for s in seeds])) if seeds else 0)

    # contrasts + Holm, per band
    contrasts = {}
    for band in BANDS:
        rows = []
        for treat, base in FAMILY:
            if treat not in per_seed or base not in per_seed:
                continue
            common = sorted(set(per_seed[treat]) & set(per_seed[base]))
            deltas = [per_seed[treat][s][band] - per_seed[base][s][band] for s in common
                      if per_seed[treat][s][band] == per_seed[treat][s][band]
                      and per_seed[base][s][band] == per_seed[base][s][band]]
            if len(deltas) < 2:
                continue
            ci = paired_ci(deltas)
            rows.append({"treat": treat, "base": base, "n": len(deltas),
                         "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                         "n_pos": ci["n_pos"], "p_raw": paired_p(deltas)})
        adj = holm([r["p_raw"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            r["sig_holm"] = bool(a < 0.05)
            r["sig_ci"] = bool(r["ci_low"] > 0 or r["ci_high"] < 0)
        contrasts[band] = rows

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"eval_split": args.eval_split, "K": args.K, "project": args.project,
               "tail_n_classes": len(tail_ids), "tail_gt_size_profile": dict(tail_gt_profile),
               "family": [list(c) for c in FAMILY], "per_arm": per_arm, "contrasts": contrasts}
    (out_dir / "metrics_allsize_v2.json").write_text(json.dumps(payload, indent=2, default=lambda x: None))

    # ---- Markdown ----
    L = ["# Recompute v2 — AP-tail all-size + per-band + Holm\n",
         f"- split: **{args.eval_split}** · K={args.K} · project=`{args.project}`",
         f"- tail: **{len(tail_ids)} classes**; GT por tamanho: "
         f"small {tail_gt_profile['small']} · medium {tail_gt_profile['medium']} · "
         f"large {tail_gt_profile['large']} (small = "
         f"{100*tail_gt_profile['small']//max(1,sum(tail_gt_profile.values()))}% da cauda)\n",
         "## AP-tail por braço e faixa (mean ± dp, [n_eff classes])\n",
         "| arm | all-size | small | medium | large |",
         "|---|---|---|---|---|"]
    order = sorted(args.arms, key=lambda a: -per_arm[a]["bands"]["all"]["mean"])
    for arm in order:
        b = per_arm[arm]["bands"]
        cells = []
        for band in BANDS:
            x = b[band]
            m = "nan" if x["mean"] != x["mean"] else f"{x['mean']:.4f}±{x['std']:.3f}"
            cells.append(f"{m} [{x['n_eff_classes']}]")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("\n> `[n]` = nº efetivo de classes de cauda não-NaN naquela faixa. "
             "all-size deve dar 42 (toda classe tem test≥5); small cai p/ ~35.\n")

    for band in BANDS:
        L += [f"\n## Contrastes — faixa **{band}** (ΔAP-tail, IC t 95%, p bruto, p Holm)\n",
              "| treat − base | n | ΔAP | IC95 | p_raw | p_holm | Holm✓ |",
              "|---|---|---|---|---|---|---|"]
        for r in contrasts[band]:
            L.append(f"| {r['treat']} − {r['base']} | {r['n']} | {r['mean']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_raw']:.4f} | "
                     f"{r['p_holm']:.4f} | {'✓' if r['sig_holm'] else '—'} |")
    (out_dir / "contrasts_holm_v2.md").write_text("\n".join(L) + "\n")
    (out_dir / "metrics_allsize_v2.md").write_text("\n".join(L) + "\n")
    print("-> reports/det/metrics_allsize_v2.json")
    print("-> reports/det/metrics_allsize_v2.md  (+ contrasts_holm_v2.md)")
    print("\n".join(L))


if __name__ == "__main__":
    main()
