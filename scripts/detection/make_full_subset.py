#!/usr/bin/env python
"""Generate a subset.json that covers ALL annotated TT100K classes (open-set-free setting).

The 21-class subset created an open-set false-positive artifact: the detector fires a target
class on real signs it was never trained on (~90% of the tail-AP ceiling; see the diagnostic).
Training on ALL annotated classes removes that artifact and exposes the REAL long-tail (dozens
of classes with <10 instances). Tiers are assigned by TRAIN-instance frequency terciles
(provisional; the evaluation protocol for ultra-rare classes is decided separately).

Only classes with >= --min-instances TRAIN instances are kept (default 1: any class the model
can actually learn). Writes to a NEW path by default so the 21-class subset.json is not touched.

Usage:
  python scripts/detection/make_full_subset.py --out data/tt100k/prepared/subset_full.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_full_subset(recs: list[dict], train_ids: set, test_ids: set,
                      min_instances: int = 1, tail_train=(10, 80), tail_test_min: int = 5,
                      head_train_min: int = 80) -> dict:
    """All classes with >= min_instances TRAIN instances (the full open-set-free target set),
    with a DATA-DRIVEN evaluable-tail tier.

    Training uses every class (removing the open-set artifact), but the AP-tail metric must be
    computed over classes that are both LEARNABLE and MEASURABLE — otherwise the ~65 classes
    with 1-4 train instances drag AP-tail to 0 (the calibration bug). Tiers:
      head : train >= head_train_min (well-represented).
      tail : tail_train[0] <= train < tail_train[1] AND test >= tail_test_min  (the EVALUABLE
             rare band — genuinely long-tail yet measurable; the paper's headline).
      mid  : everything else that is trained (incl. the ultra-rare 1-9 train classes, kept as
             training targets to fix open-set but excluded from the headline tail metric).
    """
    ctr = Counter(o["category"] for r in recs if r["id"] in train_ids for o in r["objects"])
    cte = Counter(o["category"] for r in recs if r["id"] in test_ids for o in r["objects"])
    classes = sorted((c for c, n in ctr.items() if n >= min_instances), key=lambda c: (-ctr[c], c))
    ids = {c: i for i, c in enumerate(classes)}
    lo, hi = tail_train

    def tier(c):
        if ctr[c] >= head_train_min:
            return "head"
        if lo <= ctr[c] < hi and cte.get(c, 0) >= tail_test_min:
            return "tail"
        return "mid"
    tier_of = {c: tier(c) for c in classes}
    by_tier = {t: [c for c in classes if tier_of[c] == t] for t in ("head", "mid", "tail")}
    return {
        "n_classes": len(classes),
        "min_instances": min_instances,
        "tail_criterion": {"train_range": list(tail_train), "test_min": tail_test_min},
        "classes": [{"name": c, "id": ids[c], "instances": ctr[c],
                     "test_instances": cte.get(c, 0), "tier": tier_of[c]} for c in classes],
        "names": classes,
        "by_tier": by_tier,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--out", default="data/tt100k/prepared/subset_full.json",
                    help="output path (default keeps the 21-class subset.json untouched)")
    ap.add_argument("--min-instances", type=int, default=1)
    ap.add_argument("--tail-train-min", type=int, default=10)
    ap.add_argument("--tail-train-max", type=int, default=80)
    ap.add_argument("--tail-test-min", type=int, default=5)
    ap.add_argument("--head-train-min", type=int, default=80)
    args = ap.parse_args()
    prep = Path(args.prepared)
    recs = [json.loads(l) for l in (prep / "panoramas.jsonl").read_text().splitlines() if l.strip()]
    splits = json.loads((prep / "splits.json").read_text())
    sub = build_full_subset(recs, set(splits["train"]), set(splits["test"]),
                            args.min_instances, (args.tail_train_min, args.tail_train_max),
                            args.tail_test_min, args.head_train_min)
    Path(args.out).write_text(json.dumps(sub, indent=2))
    bt = sub["by_tier"]
    tail_te = sum(c["test_instances"] for c in sub["classes"] if c["tier"] == "tail")
    print(f"full subset: {sub['n_classes']} classes -> {args.out}")
    print(f"  head {len(bt['head'])} (train >= {args.head_train_min})  mid {len(bt['mid'])}  "
          f"EVALUABLE-tail {len(bt['tail'])} (train {args.tail_train_min}-{args.tail_train_max-1} "
          f"& test >= {args.tail_test_min}; {tail_te} test instances)")


if __name__ == "__main__":
    main()
