#!/usr/bin/env python
"""CLI: compute the shared frequency water-filling allocation from the train split.

Usage:
  python scripts/detection/build_allocation.py \
      --prepared data/tt100k/prepared \
      --out data/tt100k/prepared/allocation.json [--K 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.allocation import build_allocation, save_allocation  # noqa: E402


def _load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--out", default=None,
                    help="default: <prepared>/allocation.json (derived so a non-tt100k "
                         "--prepared never writes into the tt100k spine)")
    ap.add_argument("--K", type=float, default=0.5)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    args.out = args.out or str(prepared / "allocation.json")
    rec_by_id = _load_jsonl(prepared / "panoramas.jsonl")
    subset = json.loads((prepared / "subset.json").read_text())
    subset_ids = {c["name"]: c["id"] for c in subset["classes"]}
    splits = json.loads((prepared / "splits.json").read_text())

    spec = build_allocation(rec_by_id, splits["train"], subset_ids, args.K)
    save_allocation(spec, args.out)
    print(f"B={spec['B']} (K={spec['K']}); sum(alloc)={sum(spec['alloc'].values())}")
    name_of = {c["id"]: c["name"] for c in subset["classes"]}
    top = sorted(spec["alloc"].items(), key=lambda kv: -kv[1])[:5]
    print("top-5 alloc:", [(name_of[int(k)], v) for k, v in top])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
