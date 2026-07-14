#!/usr/bin/env python
"""Eval-only pass: re-evaluate ALREADY-TRAINED runs on a given split (no retrain).

The grid trained + evaluated on --eval-split val (run_det default), so each run dir has
val ap_report.json/dets.json. This loads each run's last.pt, predicts the requested split
(reconstruct on the 2048 panorama + global NMS), and writes SPLIT-SPECIFIC artifacts
`ap_report_<split>.json` + `dets_<split>.json` (non-destructive: val files untouched).
Then `det_report.py --eval-split <split>` picks them up (load_runs prefers split files).

Usage (on the workstation, where weights + tiles live):
  python scripts/detection/eval_runs.py --eval-split test          # all 42 runs
  python scripts/detection/eval_runs.py --eval-split test --arms diffusion_bg --seeds 0   # 1 run smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))  # import predict helper from run_det.py

from detection.budget import budget_tag            # noqa: E402
from detection.run_naming import experiment_name   # noqa: E402
from detection.evaluate import evaluate_split       # noqa: E402
from detection.ap_by_size import nan_safe_dumps     # noqa: E402
from run_det import predict_dets_by_panorama        # noqa: E402

ARMS = ["zero_aug", "da_only", "real_duplicate", "bg_photometric", "copy_paste", "diffusion_bg"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="experiments/tt100k")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    project = Path(args.project).resolve()
    prepared, tiles_dir = Path(args.prepared), Path(args.tiles)
    subset = json.loads((prepared / "subset.json").read_text())
    records = {r["id"]: r for r in
               (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines() if l.strip())}
    split_ids = json.loads((prepared / "splits.json").read_text())[args.eval_split]
    bm = budget_tag(args.K)

    done, skipped = 0, 0
    for arm in args.arms:
        for s in args.seeds:
            exp = experiment_name(arm, s, budget_tag=bm)
            run_dir = project / exp
            weights = run_dir / "weights" / "last.pt"
            if not weights.exists():
                print(f"[skip] {exp}: no weights/last.pt")
                skipped += 1
                continue
            t0 = time.time()
            dets = predict_dets_by_panorama(weights, tiles_dir, args.eval_split,
                                            conf=args.conf, nms_iou=args.nms_iou, panorama_size=2048)
            result = evaluate_split(records, split_ids, subset, dets, panorama_size=2048)
            result["meta"] = {"arm": arm, "seed": s, "K": args.K, "eval_split": args.eval_split,
                              "eval_only": True}
            (run_dir / f"ap_report_{args.eval_split}.json").write_text(nan_safe_dumps(result))
            (run_dir / f"dets_{args.eval_split}.json").write_text(json.dumps(
                {pid: [{"class_id": d["class_id"], "conf": d["conf"], "box": list(d["box"])}
                       for d in dl] for pid, dl in dets.items()}))
            hl = result["headline"]
            print(f"[ok] {arm} s{s}  tail={hl.get('ap_tail'):.4f} "
                  f"small={hl.get('ap_small_macro'):.4f}  ({time.time()-t0:.0f}s)")
            done += 1
    print(f"\n[done] {done} runs avaliados no split '{args.eval_split}' | {skipped} pulados")


if __name__ == "__main__":
    main()
