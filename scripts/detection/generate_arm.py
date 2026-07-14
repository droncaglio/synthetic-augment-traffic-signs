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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from detection.generators.manifests import (  # noqa: E402
    index_instances_by_class, select_sources, assign_placements, save_manifest,
)
from detection.generators.real_duplicate import RealDuplicate  # noqa: E402
from detection.generators.bg_photometric import BgPhotometric  # noqa: E402
from detection.generators.bg_photometric_mask import BgPhotometricMask  # noqa: E402
from detection.generators.copy_paste import CopyPaste  # noqa: E402
from detection.generators.copy_paste_mask import CopyPasteMask  # noqa: E402
from detection.generators.diffusion_bg import DiffusionBg  # noqa: E402
from detection.notifications.telegram import load_env  # noqa: E402

ARM_REGISTRY = {
    "real_duplicate": RealDuplicate,
    "bg_photometric": BgPhotometric,
    "bg_photometric_mask": BgPhotometricMask,  # perturba fundo até a silhueta (SAM+fallback)
    "copy_paste": CopyPaste,
    "copy_paste_mask": CopyPasteMask,  # silhueta justa (sem halo retangular)
    "diffusion_bg": DiffusionBg,  # GPU/model — see --lora-dir / --scan-weights
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_REGISTRY))
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--seed", type=int, default=42, help="shared source-selection seed")
    # diffusion_bg only (GPU):
    ap.add_argument("--lora-dir", default=None, help="diffusion_bg: background-domain LoRA dir")
    ap.add_argument("--scan-weights", default=None,
                    help="diffusion_bg: light detector (e.g. baseline best.pt) for the "
                         "hallucination scan (REQUIRED for diffusion_bg unless --allow-no-scan)")
    ap.add_argument("--allow-no-scan", action="store_true",
                    help="diffusion_bg: permit running without the hallucination scan (dev only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="generate only the first N sources (0 = all) — for a quick visual QA")
    ap.add_argument("--resume", action="store_true",
                    help="skip tiles already written (crash-safe for the long diffusion pass)")
    ap.add_argument("--device", default=None,
                    help="diffusion_bg: GPU index to pin via CUDA_VISIBLE_DEVICES (e.g. 0)")
    args = ap.parse_args()
    load_env()  # pull HF_TOKEN (+ any creds) from repo-root .env for the diffusion arm
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)  # before torch/FLUX import

    prepared, tiles = Path(args.prepared), Path(args.tiles)
    alloc_spec = json.loads((prepared / "allocation.json").read_text())
    index = index_instances_by_class(tiles / "train" / "labels")
    sources = select_sources(alloc_spec["alloc"], index, seed=args.seed)
    save_manifest(sources, prepared / f"sources_seed{args.seed}.json")  # shared/audit

    # copy_paste* relocate the sign -> need recipient background tiles + placement manifest
    # (train tiles w/o subset labels). copy_paste_mask is a CopyPaste subclass (same needs).
    if args.arm in ("copy_paste", "copy_paste_mask"):
        bg_tiles = [t.stem for t in sorted((tiles / "train" / "labels").glob("*.txt"))
                    if not t.read_text().strip()]
        if not bg_tiles:
            sys.exit(f"{args.arm}: no background (empty-label) train tiles found.")
        entries = assign_placements(sources, bg_tiles, seed=args.seed)
        save_manifest(entries, prepared / f"placements_{args.arm}_seed{args.seed}.json")
    else:
        entries = sources

    if args.arm == "diffusion_bg":
        if not args.scan_weights and not args.allow_no_scan:
            sys.exit("diffusion_bg requires --scan-weights (hallucination scan) — it can "
                     "invent unlabeled signs otherwise. Pass a baseline best.pt, or "
                     "--allow-no-scan for a dev run.")
        gen = DiffusionBg(tiles, seed=args.seed, lora_dir=args.lora_dir,
                          scan_weights=args.scan_weights)
    else:
        gen = ARM_REGISTRY[args.arm](tiles, seed=args.seed)
    if args.limit:
        entries = entries[:args.limit]
    manifest = gen.generate(entries, tiles / "arms" / args.arm, resume=args.resume)
    print(f"[{args.arm}] sources={len(sources)} tiles_written={manifest['n_tiles_written']}")
    print(f"  allocated/class={manifest['allocated_per_class']}")
    print(f"-> {tiles / 'arms' / args.arm}")


if __name__ == "__main__":
    main()
