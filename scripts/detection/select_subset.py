#!/usr/bin/env python
"""CLI: deterministically select the class subset from prepared/catalog.json.

Usage:
  python scripts/detection/select_subset.py \
      --catalog data/tt100k/prepared/catalog.json \
      --out data/tt100k/prepared/subset.json \
      [--n-classes 20] [--min-instances 80]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.subset import select_subset, save_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/tt100k/prepared/catalog.json")
    ap.add_argument("--out", default="data/tt100k/prepared/subset.json")
    ap.add_argument("--n-classes", type=int, default=20)
    ap.add_argument("--min-instances", type=int, default=80)
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    subset = select_subset(catalog, args.n_classes, args.min_instances)
    save_subset(subset, args.out)
    print(f"selected {subset['n_classes']} classes (min_instances={subset['min_instances']})")
    for t, names in subset["by_tier"].items():
        print(f"  {t:4s}: {names}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
