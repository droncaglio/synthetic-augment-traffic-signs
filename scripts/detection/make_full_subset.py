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


def build_full_subset(recs: list[dict], train_ids: set, min_instances: int = 1) -> dict:
    """All classes with >= min_instances TRAIN instances, tiered by train frequency terciles."""
    cnt = Counter(o["category"] for r in recs if r["id"] in train_ids for o in r["objects"])
    classes = sorted((c for c, n in cnt.items() if n >= min_instances),
                     key=lambda c: (-cnt[c], c))  # train-freq desc, stable
    ids = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    t = n // 3
    head, mid, tail = classes[:t], classes[t:2 * t], classes[2 * t:]
    tier_of = {**{c: "head" for c in head}, **{c: "mid" for c in mid}, **{c: "tail" for c in tail}}
    return {
        "n_classes": n,
        "min_instances": min_instances,
        "classes": [{"name": c, "id": ids[c], "instances": cnt[c], "tier": tier_of[c]}
                    for c in classes],
        "names": classes,
        "by_tier": {"head": head, "mid": mid, "tail": tail},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--out", default="data/tt100k/prepared/subset_full.json",
                    help="output path (default keeps the 21-class subset.json untouched)")
    ap.add_argument("--min-instances", type=int, default=1)
    args = ap.parse_args()
    prep = Path(args.prepared)
    recs = [json.loads(l) for l in (prep / "panoramas.jsonl").read_text().splitlines() if l.strip()]
    train_ids = set(json.loads((prep / "splits.json").read_text())["train"])
    sub = build_full_subset(recs, train_ids, args.min_instances)
    Path(args.out).write_text(json.dumps(sub, indent=2))
    bt = sub["by_tier"]
    inst = {c["name"]: c["instances"] for c in sub["classes"]}
    print(f"full subset: {sub['n_classes']} classes -> {args.out}")
    print(f"  head {len(bt['head'])} (>= {inst[bt['head'][-1]]} train-inst)  "
          f"mid {len(bt['mid'])}  tail {len(bt['tail'])} (<= {inst[bt['tail'][-1]]} train-inst)")


if __name__ == "__main__":
    main()
