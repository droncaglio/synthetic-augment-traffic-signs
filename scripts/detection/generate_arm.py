#!/usr/bin/env python
"""CLI: generate an arm's synthetic tiles from the SHARED allocation + source manifest.

Usage:
  python scripts/detection/generate_arm.py --arm real_duplicate \
      --prepared data/tt100k/prepared --tiles data/tt100k/tiles [--seed 42]

--seed drives the SHARED source selection (same seed -> same sources for every arm,
which is what pairs the comparison). Output: data/tt100k/tiles/arms/<arm>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.generators.manifests import (  # noqa: E402
    index_instances_by_class, select_sources, save_manifest,
)
from detection.generators.real_duplicate import RealDuplicate  # noqa: E402

ARM_REGISTRY = {
    "real_duplicate": RealDuplicate,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_REGISTRY))
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--seed", type=int, default=42, help="shared source-selection seed")
    args = ap.parse_args()

    prepared, tiles = Path(args.prepared), Path(args.tiles)
    alloc_spec = json.loads((prepared / "allocation.json").read_text())
    index = index_instances_by_class(tiles / "train" / "labels")
    sources = select_sources(alloc_spec["alloc"], index, seed=args.seed)
    save_manifest(sources, prepared / f"sources_seed{args.seed}.json")  # audit/shared

    gen = ARM_REGISTRY[args.arm](tiles, seed=args.seed)
    manifest = gen.generate(sources, tiles / "arms" / args.arm)
    print(f"[{args.arm}] sources={len(sources)} tiles_written={manifest['n_tiles_written']}")
    print(f"  allocated/class={manifest['allocated_per_class']}")
    print(f"-> {tiles / 'arms' / args.arm}")


if __name__ == "__main__":
    main()
