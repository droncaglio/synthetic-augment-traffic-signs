"""Frequency water-filling allocation of the synthetic budget across subset classes.

B = round(K * N_train_subset_instances) synthetic instances are distributed to the
RAREST classes first, raising them toward a common floor (water level). Classes
already above the floor get 0. The allocation is a pure function of (train counts, B),
so it is IDENTICAL across all arms — the shared budget that isolates the content axis.

Anti-leak: train counts are computed from the TRAIN split only.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path


def train_instance_counts(records_by_id: dict[str, dict], train_ids: list[str],
                          subset_ids: dict[str, int]) -> dict[int, int]:
    """Per-class subset-instance counts in the TRAIN panoramas only."""
    counts = {cid: 0 for cid in subset_ids.values()}
    for pid in train_ids:
        for o in records_by_id[pid]["objects"]:
            cid = subset_ids.get(o["category"])
            if cid is not None:
                counts[cid] += 1
    return counts


def budget_from_K(counts: dict[int, int], K: float) -> int:
    """B = round(K * total train subset instances)."""
    return int(round(K * sum(counts.values())))


def compute_water_fill(counts: dict[int, int], B: int,
                       alpha: float | None = None, g_max: int | None = None
                       ) -> dict[int, int]:
    """Distribute up to B units to the lowest effective-count classes (water-filling),
    with optional per-class caps (reviewer's g_c = min(deficit, alpha*n_c, g_max)).

    * alpha: cap generation at alpha * real_count (e.g. alpha=3 -> a class with 10 real
      instances gets at most 30 synthetic), so tiny classes never dominate the budget.
    * g_max: optional absolute cap per class.
    Deterministic (heap ties break by class id). sum(result) == B unless every class
    hits its cap first (then sum < B; the caller reports the effective budget).
    """
    alloc = {c: 0 for c in counts}
    if B <= 0:
        return alloc

    def cap_of(c: int) -> int | None:
        caps = []
        if alpha is not None:
            caps.append(int(alpha * counts[c]))
        if g_max is not None:
            caps.append(int(g_max))
        return min(caps) if caps else None

    heap = [(counts[c], c) for c in counts
            if cap_of(c) is None or cap_of(c) > 0]   # (effective_count, class_id)
    heapq.heapify(heap)
    allocated = 0
    while allocated < B and heap:
        e, c = heapq.heappop(heap)
        cp = cap_of(c)
        if cp is not None and alloc[c] >= cp:
            continue                      # class already capped -> drop from the fill
        alloc[c] += 1
        allocated += 1
        if cp is None or alloc[c] < cp:
            heapq.heappush(heap, (e + 1, c))
    return alloc


def build_allocation(records_by_id: dict[str, dict], train_ids: list[str],
                     subset_ids: dict[str, int], K: float = 0.5,
                     alpha: float | None = 3.0, g_max: int | None = None) -> dict:
    """Compute the shared allocation spec from the train split.

    Default alpha=3.0 caps per-class generation at 3x its real count (reviewer's
    recommendation), so extremely small classes do not dominate the budget.
    """
    counts = train_instance_counts(records_by_id, train_ids, subset_ids)
    B = budget_from_K(counts, K)
    alloc = compute_water_fill(counts, B, alpha=alpha, g_max=g_max)
    return {
        "K": K, "B": B, "B_effective": sum(alloc.values()),
        "alpha": alpha, "g_max": g_max,
        "train_counts": {str(k): v for k, v in counts.items()},
        "alloc": {str(k): v for k, v in alloc.items()},
    }


def save_allocation(spec: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec, indent=2))


def load_allocation(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
