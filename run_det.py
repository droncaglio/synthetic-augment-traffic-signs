#!/usr/bin/env python
"""Run ONE detection experiment (arm x seed): train + panorama-level evaluation.

Trains YOLO on the arm's 640 tiles with an equalized optimizer-step budget, then
evaluates by reconstructing per-tile predictions onto the panorama + global NMS
(detection.reconstruct) and computing AP@small / AP-tail (detection.evaluate).
Writes <project>/<dataset>/<experiment>/ap_report.json.

Training needs a GPU (use --device); everything else is CPU. Content arms
(real_duplicate/copy_paste/bg_photometric/diffusion_bg) require Stage-2 generators
to have written the arm's synthetic tiles — until then only zero_aug/da_only run.

Example (on the workstation):
  python run_det.py --arm zero_aug --seed 0 --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from detection.train_harness import (
    equalized_plan, total_steps_from_reference, train_arm, resolve_arm_train_dirs,
    loss_plateaued,
)
from detection.reconstruct import reconstruct_panorama
from detection.evaluate import evaluate_split
from detection.ap_by_size import nan_safe_dumps
from detection.run_naming import experiment_name
from detection.budget import budget_tag
from detection.notifications.telegram import TelegramNotifier

ROOT = Path(__file__).resolve().parent


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def _count_images(d: Path) -> int:
    return sum(1 for _ in d.glob("*.jpg")) if d.exists() else 0


def build_dataset_yaml(tiles_dir: Path, train_dirs: list[Path], subset: dict, out: Path) -> Path:
    """Write an Ultralytics dataset.yaml. train may be a LIST of dirs (real + synthetic).
    Our AP is computed separately (panorama-level); Ultralytics' val is only to let
    training run (and is skipped entirely when --val is off)."""
    names = {c["id"]: c["name"] for c in subset["classes"]}
    out.write_text(yaml.safe_dump({
        "path": str(tiles_dir.resolve()),
        "train": [str(Path(d).resolve()) for d in train_dirs],
        "val": str((tiles_dir / "val" / "images").resolve()),
        "names": names,
    }, sort_keys=False))
    return out


def predict_dets_by_panorama(weights: Path, tiles_dir: Path, split: str,
                             conf: float, nms_iou: int, panorama_size: int) -> dict:
    """Predict every tile of `split`, group by panorama, reconstruct + global NMS."""
    from ultralytics import YOLO

    tile_index = json.loads((tiles_dir / split / "tile_index.json").read_text())
    by_pano: dict[str, list[dict]] = {}
    model = YOLO(str(weights))

    # group tiles per panorama
    per_pano_tiles: dict[str, list[dict]] = {}
    for e in tile_index:
        per_pano_tiles.setdefault(e["panorama_id"], []).append(e)

    img_dir = tiles_dir / split / "images"
    lab_dir = tiles_dir / split / "labels"
    for pid, entries in per_pano_tiles.items():
        # batch-predict all tiles of this panorama in one call (much faster than 1-by-1)
        paths, kept = [], []
        for e in entries:
            img = img_dir / f"{e['tile']}.jpg"
            if img.exists():
                paths.append(str(img))
                kept.append(e)
        if not paths:
            by_pano[pid] = []
            continue
        results = model.predict(paths, conf=conf, iou=0.45, verbose=False, save=False)
        tiles_payload = []
        for e, r in zip(kept, results):
            b = r.boxes
            dets = []
            if b is not None and len(b) > 0:
                xywhn = b.xywhn.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                cf = b.conf.cpu().numpy()
                for j in range(len(cls)):
                    dets.append({"class_id": int(cls[j]), "conf": float(cf[j]),
                                 "box": (float(xywhn[j, 0]), float(xywhn[j, 1]),
                                         float(xywhn[j, 2]), float(xywhn[j, 3]))})
            ig_path = lab_dir / f"{e['tile']}.ignore.json"
            ignores = json.loads(ig_path.read_text()) if ig_path.exists() else []
            tiles_payload.append({"entry": e, "dets": dets, "ignores": ignores})
        by_pano[pid] = reconstruct_panorama(tiles_payload, panorama_size, nms_iou)
    return by_pano


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--base-epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default="", help="'ram' | 'disk' | '' (off)")
    ap.add_argument("--val", action="store_true",
                    help="keep Ultralytics per-epoch val + best.pt (convergence probe); "
                         "default off -> fixed-budget, evaluate last.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prepared", default="data/tt100k/prepared")
    ap.add_argument("--tiles", default="data/tt100k/tiles")
    ap.add_argument("--configs", default="configs/detection")
    ap.add_argument("--project", default="experiments/tt100k")
    args = ap.parse_args()

    prepared, tiles_dir, cfgs = Path(args.prepared), Path(args.tiles), Path(args.configs)
    subset = json.loads((prepared / "subset.json").read_text())
    arm_cfg = _load_yaml(cfgs / "arm" / f"{args.arm}.yaml")
    model_cfg = _load_yaml(cfgs / "model" / "yolo11n.yaml")

    exp = experiment_name(args.arm, args.seed, smoke=args.smoke, budget_tag=budget_tag(args.K))
    notifier = TelegramNotifier.from_env()
    notifier.send_separator()
    notifier.send_start(exp, {"dataset": "tt100k", "arm": args.arm, "seed": args.seed,
                              "smoke": args.smoke})
    t0 = time.time()
    try:
        # content arms train on real + synthetic tiles; baselines on raw train tiles.
        train_dirs = resolve_arm_train_dirs(args.arm, tiles_dir)
        n_arm = sum(_count_images(d) for d in train_dirs)
        n_ref = _count_images(tiles_dir / "train" / "images")
        base_epochs = 2 if args.smoke else args.base_epochs
        total_steps = total_steps_from_reference(n_ref, args.batch, base_epochs)
        plan = equalized_plan(n_arm, args.batch, total_steps)
        # Equalized-steps fairness is central to the paper: for official runs, abort
        # (don't just warn) if the realized step budget drifts out of tolerance.
        if not plan["within_tol"]:
            msg = f"step budget deviation {plan['deviation']:.3f} for {args.arm} (tol 2%)"
            if args.smoke:
                print(f"[warn smoke] {msg}")
            else:
                raise RuntimeError(f"{msg} — aborting official run (fairness invariant).")

        ds_yaml = build_dataset_yaml(tiles_dir, train_dirs, subset,
                                     tiles_dir / f"dataset_{args.arm}.yaml")
        weights = train_arm(ds_yaml, model_cfg["weights"], args.project, exp,
                            epochs=plan["epochs"], batch=args.batch, imgsz=args.imgsz,
                            seed=args.seed, runtime_aug=arm_cfg["runtime_aug"], device=args.device,
                            val=args.val, workers=args.workers,
                            cache=(args.cache or False))

        # panorama-level evaluation
        records = {r["id"]: r for r in
                   (json.loads(l) for l in (prepared / "panoramas.jsonl").read_text().splitlines() if l.strip())}
        splits = json.loads((prepared / "splits.json").read_text())
        dets = predict_dets_by_panorama(weights, tiles_dir, args.eval_split,
                                        conf=0.001, nms_iou=0.5, panorama_size=2048)
        result = evaluate_split(records, splits[args.eval_split], subset, dets, panorama_size=2048)
        # val-free convergence check (train-loss plateau) — flags subtraining per run
        converged, loss_info = loss_plateaued(weights.parent.parent / "results.csv")
        if not converged and not args.smoke:
            print(f"[warn] loss não platô p/ {args.arm} ({loss_info}) — possível subtreino; "
                  f"considere subir --base-epochs.")
        result["meta"] = {"arm": args.arm, "seed": args.seed, "K": args.K,
                          "eval_split": args.eval_split, "epochs": plan["epochs"],
                          "steps": plan["realized_steps"], "deviation": plan["deviation"],
                          "within_tol": plan["within_tol"], "n_train_tiles": n_arm,
                          "converged": converged, "loss_check": loss_info}

        out = weights.parent.parent / "ap_report.json"  # alongside the trained weights
        out.write_text(nan_safe_dumps(result))
        # save per-panorama reconstructed detections so det_report can run the paired
        # bootstrap CI (stats.bootstrap_delta_ap needs the raw dets, not just the AP).
        (weights.parent.parent / "dets.json").write_text(json.dumps(
            {pid: [{"class_id": d["class_id"], "conf": d["conf"], "box": list(d["box"])}
                   for d in dl] for pid, dl in dets.items()}))
        hl = result["headline"]
        hl_meta = {**hl, "epochs": plan["epochs"],
                   "train_time_hours": (time.time() - t0) / 3600}
        notifier.send_success(exp, hl_meta)
        print(f"AP@small(macro)={hl['ap_small_macro']:.4f}  AP-tail={hl['ap_tail']:.4f}  -> {out}")
    except Exception as e:  # noqa: BLE001
        notifier.send_failure(exp, e)
        raise


if __name__ == "__main__":
    main()
