#!/usr/bin/env python
"""Batch runner for the detection grid (arms x seeds) — resumable, ONE command.

For each (arm, seed) it: (1) generates the arm's synthetic tiles once if missing
(ENIAC-style auto-prep embedded in the batch — baselines skip this; diffusion resumes
and uses the already-trained zero_aug seed 0 as its anti-hallucination scanner), then
(2) runs run_det.py as a subprocess. Status is tracked in a JSON so the whole pipeline
resumes after a crash (done runs skipped; --retry-failed re-runs failures). The config
order (zero_aug first ... diffusion_bg last) makes the scanner available in time.

Usage (generates all arms + trains all 42 runs in one go):
  python batch_run_det.py --batch configs/detection/batches/full_grid_det.yaml \
      --device 0 --base-epochs 25 [--retry-failed] [--skip-generate] [--dry-run]
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


# content arms need their synthetic tiles generated ONCE before training (ENIAC-style
# auto-prep embedded in the batch). Baselines (zero_aug/da_only) train on raw tiles.
CONTENT_ARMS = ("real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg")


def _arm_generated(tiles: str, arm: str) -> bool:
    return (Path(tiles) / "arms" / arm / "generation_manifest.json").exists()


def _default_scanner(project: str, bm: str) -> str | None:
    """diffusion_bg scanner = the already-trained zero_aug seed 0 (it runs first in the
    grid, so by the time diffusion is reached its weights exist)."""
    for w in ("last.pt", "best.pt"):
        p = Path(project) / experiment_name("zero_aug", 0, budget_tag=bm) / "weights" / w
        if p.exists():
            return str(p)
    return None


def _generate_arm(arm: str, args, bm: str, dry: bool) -> None:
    """Generate an arm's synthetic tiles (idempotent; diffusion resumes + needs a scanner)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "detection" / "generate_arm.py"),
           "--arm", arm, "--tiles", args.tiles, "--prepared", args.prepared]
    if arm == "diffusion_bg":
        scanner = args.scan_weights or _default_scanner(args.project, bm)
        if not scanner and not dry:  # at run time the scanner must exist (zero_aug ran first)
            raise SystemExit("diffusion_bg generation needs a scanner — pass --scan-weights "
                             "or ensure zero_aug seed 0 trained earlier in the grid.")
        if dry:
            print(f"would generate: {arm} (scanner={scanner or 'zero_aug seed0 @ runtime'})")
            return
        cmd += ["--scan-weights", scanner, "--resume", "--device", args.device]
    if dry:
        print(f"would generate: {arm}")
        return
    print(f"GEN: {arm}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default="configs/detection/batches/full_grid_det.yaml")
    ap.add_argument("--device", default="0")
    ap.add_argument("--base-epochs", type=int, default=25)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--scan-weights", default=None,
                    help="diffusion_bg scanner (default: zero_aug seed 0 weights from this grid)")
    ap.add_argument("--status-file", default="batch_status_det.json")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--skip-generate", action="store_true",
                    help="assume all content-arm tiles are already generated")
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
    ensured: set = set()  # arms whose generation we've handled this invocation
    for arm, seed in runs:
        exp = experiment_name(arm, seed, budget_tag=bm)
        rid = f"{dataset}_{exp}"
        ap_report = Path(args.project) / exp / "ap_report.json"
        prev = status["runs"].get(rid, {})
        done = ap_report.exists() and prev.get("status") == "done"
        if done and not (args.retry_failed and prev.get("status") == "failed"):
            print(f"skip (done): {rid}")
            continue

        # ENIAC-style: ensure the arm's synthetic tiles exist before its first run.
        if (not args.skip_generate and arm in CONTENT_ARMS and arm not in ensured
                and not _arm_generated(args.tiles, arm)):
            _generate_arm(arm, args, bm, dry=args.dry_run)
            ensured.add(arm)

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
                                        "loss_smoke_ok": rep.get("meta", {}).get("loss_smoke_ok")})
        else:
            status["runs"][rid].update({"status": "failed", "finished_at": _now(),
                                        "returncode": proc.returncode})
        status_path.write_text(json.dumps(status, indent=2))

    n_done = sum(1 for r in status["runs"].values() if r.get("status") == "done")
    n_fail = sum(1 for r in status["runs"].values() if r.get("status") == "failed")
    print(f"\nbatch done: {n_done} ok, {n_fail} failed -> {status_path}")


if __name__ == "__main__":
    main()
