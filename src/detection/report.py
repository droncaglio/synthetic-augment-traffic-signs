"""Pure report helpers (testable): build eval GTs, load grid runs, aggregate per-arm.

The CLI (scripts/detection/det_report.py) wires these + the bootstrap into report.md.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from detection.prepare import xyxy_to_yolo
from detection.tiling import tile_grid, clip_visibility
from detection.run_naming import experiment_name


def gts_by_pid(records: dict, split_ids: list, subset_ids: dict, size: int = 2048,
               keep: float = 0.6) -> dict:
    """{pid: [{class_id, box}]} for labelable subset signs (same filter as evaluate)."""
    grid = tile_grid(size, size, 640, 128)
    out: dict = {}
    for pid in split_ids:
        lst = []
        for o in records[pid]["objects"]:
            cid = subset_ids.get(o["category"])
            if cid is None:
                continue
            xyxy = tuple(o["xyxy"])
            if not any(clip_visibility(xyxy, t)[1] >= keep for t in grid):
                continue
            lst.append({"class_id": cid, "box": xyxy_to_yolo(xyxy, size, size)})
        out[pid] = lst
    return out


def load_runs(project: str | Path, arm: str, seeds, bm: str) -> list[tuple[dict, dict]]:
    """Per-seed (headline, dets_by_pid) for an arm; skips seeds without ap_report+dets."""
    runs = []
    for s in seeds:
        d = Path(project) / experiment_name(arm, s, budget_tag=bm)
        rep, dets = d / "ap_report.json", d / "dets.json"
        if rep.exists() and dets.exists():
            hl = json.loads(rep.read_text()).get("headline", {})
            dj = {pid: [{"class_id": x["class_id"], "conf": x["conf"], "box": tuple(x["box"])}
                        for x in dl] for pid, dl in json.loads(dets.read_text()).items()}
            runs.append((hl, dj))
    return runs


def aggregate_arm(runs: list[tuple[dict, dict]]) -> dict | None:
    """Mean ± std of AP-tail / AP@small(macro) over an arm's seeds."""
    if not runs:
        return None
    tail = [hl.get("ap_tail", 0.0) for hl, _ in runs]
    small = [hl.get("ap_small_macro", 0.0) for hl, _ in runs]
    return {
        "n_seeds": len(runs),
        "ap_tail_mean": round(st.mean(tail), 4),
        "ap_tail_std": round(st.pstdev(tail), 4) if len(tail) > 1 else 0.0,
        "ap_small_mean": round(st.mean(small), 4),
        "ap_small_std": round(st.pstdev(small), 4) if len(small) > 1 else 0.0,
    }
