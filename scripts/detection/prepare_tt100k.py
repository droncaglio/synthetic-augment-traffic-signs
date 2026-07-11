#!/usr/bin/env python
"""CLI: parse TT100K annotations into prepared/ records + catalog.

Usage:
  python scripts/detection/prepare_tt100k.py \
      --annotations data/tt100k/tt100k_2021/annotations_all.json \
      --out data/tt100k/prepared [--panorama-size 2048] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.prepare import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="data/tt100k/tt100k_2021/annotations_all.json")
    ap.add_argument("--out", default="data/tt100k/prepared")
    ap.add_argument("--panorama-size", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    catalog = prepare(args.annotations, args.out, args.panorama_size, args.force)
    print(f"panoramas: {catalog['n_panoramas']}  categories: {catalog['n_categories']}")
    print(f"split_orig: {catalog['split_orig_counts']}")
    top = list(catalog["categories"].items())[:5]
    print("top-5 classes:", [(c, d["instances"]) for c, d in top])
    print(f"-> {args.out}/panoramas.jsonl + catalog.json")


if __name__ == "__main__":
    main()
