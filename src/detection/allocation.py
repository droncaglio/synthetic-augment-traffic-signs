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


def compute_water_fill(counts: dict[int, int], B: int) -> dict[int, int]:
    """Distribute B units to the lowest effective-count classes first (water-filling).

    Deterministic (heap ties break by class id). Guarantees sum(result) == B.
    """
    alloc = {c: 0 for c in counts}
    if B <= 0:
        return alloc
    heap = [(n, c) for c, n in counts.items()]   # (effective_count, class_id)
    heapq.heapify(heap)
    for _ in range(B):
        e, c = heapq.heappop(heap)
        alloc[c] += 1
        heapq.heappush(heap, (e + 1, c))
    return alloc


def build_allocation(records_by_id: dict[str, dict], train_ids: list[str],
                     subset_ids: dict[str, int], K: float = 0.5) -> dict:
    """Compute the shared allocation spec from the train split."""
    counts = train_instance_counts(records_by_id, train_ids, subset_ids)
    B = budget_from_K(counts, K)
    alloc = compute_water_fill(counts, B)
    return {
        "K": K, "B": B,
        "train_counts": {str(k): v for k, v in counts.items()},
        "alloc": {str(k): v for k, v in alloc.items()},
    }


def save_allocation(spec: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec, indent=2))


def load_allocation(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
