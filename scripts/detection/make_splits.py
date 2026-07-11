#!/usr/bin/env python
"""CLI: panorama-level split with pHash near-dup grouping (anti-leak).

Usage:
  python scripts/detection/make_splits.py \
      --prepared data/tt100k/prepared \
      --raw data/tt100k/tt100k_2021 \
      --out data/tt100k/prepared/splits.json \
      [--seed 42] [--phash-T 5] [--min-test-support 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.splits import make_splits, save_splits  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--raw", default="data/tt100k/tt100k_2021")
    ap.add_argument("--out", default="data/tt100k/prepared/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phash-T", type=int, default=5)
    ap.add_argument("--min-test-support", type=int, default=10)
    args = ap.parse_args()

    prepared = Path(args.prepared)
    records = _load_jsonl(prepared / "panoramas.jsonl")
    subset = json.loads((prepared / "subset.json").read_text())

    splits = make_splits(records, subset, args.raw, seed=args.seed,
                         phash_T=args.phash_T, min_test_support=args.min_test_support)
    save_splits(splits, args.out)
    m = splits["meta"]
    n = {s: len(splits[s]) for s in ("train", "val", "test")}
    print(f"panoramas: {sum(n.values())}  groups: {m['n_groups']}  "
          f"missing_hash: {m['n_missing_hash']}")
    print(f"split sizes: {n}")
    if m["warnings"]:
        print("warnings:", m["warnings"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
