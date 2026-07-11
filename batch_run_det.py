#!/usr/bin/env python
"""Batch runner for the detection grid (arms x seeds) — resumable.

Runs run_det.py once per (arm, seed) as a subprocess, tracking status in a JSON so
the grid can be resumed after a crash (done runs are skipped; --retry-failed re-runs
failures). Order follows the batch config (baselines + cheap arms first, diffusion last).

Usage:
  python batch_run_det.py --batch configs/detection/batches/full_grid_det.yaml \
      --device 0 --base-epochs 25 [--retry-failed] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from detection.run_naming import experiment_name  # noqa: E402
from detection.budget import budget_tag  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_status(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"runs": {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="configs/detection/batches/full_grid_det.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--base-epochs", type=int, default=25)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--status-file", default="batch_status_det.json")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.batch).read_text())
    K = float(cfg.get("K", 0.5))
    bm = budget_tag(K)
    dataset = cfg.get("dataset", "tt100k")
    runs = [(arm, seed) for arm in cfg["arms"] for seed in cfg["seeds"]]

    status_path = Path(args.status_file)
    status = _load_status(status_path)
    status["batch"] = cfg.get("name", Path(args.batch).stem)

    print(f"grid: {len(runs)} runs ({len(cfg['arms'])} arms x {len(cfg['seeds'])} seeds), "
          f"K={K}, base_epochs={args.base_epochs}")
    for arm, seed in runs:
        exp = experiment_name(arm, seed, budget_tag=bm)
        rid = f"{dataset}_{exp}"
        ap_report = Path(args.project) / exp / "ap_report.json"
        prev = status["runs"].get(rid, {})
        done = ap_report.exists() and prev.get("status") == "done"
        if done and not (args.retry_failed and prev.get("status") == "failed"):
            print(f"skip (done): {rid}")
            continue
        if args.dry_run:
            print(f"would run: {rid}")
            continue

        print(f"RUN: {rid}")
        status["runs"][rid] = {"arm": arm, "seed": seed, "status": "running",
                               "started_at": _now()}
        status_path.write_text(json.dumps(status, indent=2))
        cmd = [sys.executable, str(ROOT / "run_det.py"), "--arm", arm, "--seed", str(seed),
               "--K", str(K), "--device", args.device, "--base-epochs", str(args.base_epochs),
               "--project", args.project]
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and ap_report.exists():
            rep = json.loads(ap_report.read_text())
            hl = rep.get("headline", {})
            status["runs"][rid].update({"status": "done", "finished_at": _now(),
                                        "ap_small_macro": hl.get("ap_small_macro"),
                                        "ap_tail": hl.get("ap_tail"),
                                        "converged": rep.get("meta", {}).get("converged")})
        else:
            status["runs"][rid].update({"status": "failed", "finished_at": _now(),
                                        "returncode": proc.returncode})
        status_path.write_text(json.dumps(status, indent=2))

    n_done = sum(1 for r in status["runs"].values() if r.get("status") == "done")
    n_fail = sum(1 for r in status["runs"].values() if r.get("status") == "failed")
    print(f"\nbatch done: {n_done} ok, {n_fail} failed -> {status_path}")


if __name__ == "__main__":
    main()
