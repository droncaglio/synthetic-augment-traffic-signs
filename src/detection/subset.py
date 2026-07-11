"""Deterministic subset selection: pick ~15-25 TT100K classes (head/mid/tail).

Rule (no randomness, no AP — reproducible from the catalog + config):
  1. Eligible = classes with total instances >= ``min_instances`` (a floor that
     makes >= min_test_support plausible in a ~15% test split; the HARD >=10-in-test
     constraint is enforced later by splits.py).
  2. Partition eligible (sorted by instances desc) into 3 equal bands: head / mid / tail.
  3. From each band pick ``k = round(n_classes/3)`` classes at evenly-spaced rank
     positions (covers the spread of each band without RNG).
  4. Final subset = union, ordered by instances desc, contiguous ids 0..K-1, tier tagged.

Never selects by observed AP (anti-leak: selection must not peek at test performance).
"""
from __future__ import annotations

import json
from pathlib import Path

TIERS = ("head", "mid", "tail")


def _pick_evenly(band: list[str], k: int) -> list[str]:
    """Pick k names from an ordered band at evenly-spaced rank positions."""
    n = len(band)
    if k >= n:
        return list(band)
    if k <= 1:
        return [band[0]]
    idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    # rounding collisions may yield < k unique indices; backfill from unused ranks
    if len(idxs) < k:
        for j in range(n):
            if j not in idxs:
                idxs.append(j)
                if len(idxs) == k:
                    break
        idxs = sorted(idxs)
    return [band[i] for i in idxs]


def _three_bands(names_desc: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split an instances-desc ordered list into head/mid/tail (as equal as possible)."""
    n = len(names_desc)
    b = n // 3
    r = n % 3
    # distribute the remainder to head then mid so tail is never the largest
    h = b + (1 if r >= 1 else 0)
    m = b + (1 if r >= 2 else 0)
    return names_desc[:h], names_desc[h:h + m], names_desc[h + m:]


def select_subset(catalog: dict, n_classes: int = 20, min_instances: int = 80) -> dict:
    """Deterministically select the class subset. Returns a subset spec dict."""
    cats = catalog["categories"]  # already ordered by (-instances, name) from prepare
    eligible = [c for c, d in cats.items() if d["instances"] >= min_instances]
    if len(eligible) < 3:
        raise ValueError(
            f"select_subset: only {len(eligible)} classes >= {min_instances} instances; "
            f"lower min_instances or check the catalog."
        )

    head_b, mid_b, tail_b = _three_bands(eligible)
    per_band = max(1, round(n_classes / 3))
    picked_by_tier = {
        "head": _pick_evenly(head_b, per_band),
        "mid": _pick_evenly(mid_b, per_band),
        "tail": _pick_evenly(tail_b, per_band),
    }
    tier_of = {name: tier for tier, names in picked_by_tier.items() for name in names}

    # final order: instances desc (eligible already desc), contiguous ids
    chosen = [c for c in eligible if c in tier_of]
    classes = [
        {"name": c, "id": i, "instances": cats[c]["instances"], "tier": tier_of[c]}
        for i, c in enumerate(chosen)
    ]
    return {
        "n_classes": len(classes),
        "min_instances": min_instances,
        "classes": classes,
        "names": [c["name"] for c in classes],
        "by_tier": {t: [c["name"] for c in classes if c["tier"] == t] for t in TIERS},
    }


def save_subset(subset: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(subset, indent=2))


def load_subset(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
