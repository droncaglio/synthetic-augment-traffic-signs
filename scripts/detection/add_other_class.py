#!/usr/bin/env python
"""Add (or remove) the open-set 'other' distractor class in subset.json, idempotently.

Why: the detector trains on a 21-class SUBSET of TT100K's ~200 classes. Out-of-subset signs
are otherwise painted out in training, so the model never learns "sign-but-not-a-target" and
at test fires a target (tail) class on real non-target / unlabeled signs — an open-set
false-positive confound that dominates the tail AP ceiling (see the diagnostic). Adding a
single 'other' class (trained on the ~16.5k annotated out-of-subset signs via tiling.py) gives
the model a place to put non-target signs. Eval ignores 'other' automatically (GT is built
from the original category names, never 'other'; the AP loop only scores the target names).

Usage:
  python scripts/detection/add_other_class.py [--subset data/tt100k/prepared/subset.json]
  python scripts/detection/add_other_class.py --remove      # revert to target-only subset
After changing subset.json, regenerate the TRAIN (and VAL) tiles:
  python scripts/detection/tile_panoramas.py --split train ...
  python scripts/detection/tile_panoramas.py --split val ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OTHER = "other"


def add_other(subset: dict) -> dict:
    """Append the 'other' class with the next contiguous id (idempotent)."""
    if any(c["name"] == OTHER for c in subset["classes"]):
        return subset
    oid = max(c["id"] for c in subset["classes"]) + 1
    subset["classes"].append({"name": OTHER, "id": oid, "instances": None,
                              "tier": "distractor"})
    subset["names"] = list(subset.get("names", [])) + [OTHER]
    subset["n_classes"] = len(subset["classes"])
    subset.setdefault("by_tier", {})["distractor"] = [OTHER]
    return subset


def remove_other(subset: dict) -> dict:
    """Drop the 'other' class (idempotent). Ids of the target classes are unchanged."""
    subset["classes"] = [c for c in subset["classes"] if c["name"] != OTHER]
    subset["names"] = [n for n in subset.get("names", []) if n != OTHER]
    subset["n_classes"] = len(subset["classes"])
    subset.get("by_tier", {}).pop("distractor", None)
    return subset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="data/tt100k/prepared/subset.json")
    ap.add_argument("--remove", action="store_true", help="revert: drop the 'other' class")
    args = ap.parse_args()
    p = Path(args.subset)
    sub = json.loads(p.read_text())
    n0 = sub["n_classes"]
    sub = remove_other(sub) if args.remove else add_other(sub)
    p.write_text(json.dumps(sub, indent=2))
    where = f"id={sub['names'].index(OTHER)}" if OTHER in sub["names"] else "absent"
    print(f"subset.json: {n0} -> {sub['n_classes']} classes ('other' {where})")


if __name__ == "__main__":
    main()
