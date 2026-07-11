"""Training harness with EQUALIZED optimizer steps across arms.

Adding synthetic tiles enlarges the train set, so at a fixed epoch count the
augmented arms would take more optimizer steps than Zero-Aug — and a gain could
then come from more optimization, not from the content. We instead fix a TOTAL
optimizer-step budget (anchored to Zero-Aug) and pick per-arm epochs so every arm
consumes ~the same number of updates (early stopping OFF).

  steps_per_epoch(n) = ceil(n / batch)                 # Ultralytics keeps the last partial batch
  TOTAL_STEPS        = base_epochs * steps_per_epoch(n_zero_aug)
  epochs_arm         = round(TOTAL_STEPS / steps_per_epoch(n_arm))
  realized_steps     = epochs_arm * steps_per_epoch(n_arm)   # asserted within `tol` of TOTAL_STEPS

The step math is pure/unit-tested; train_arm() wraps ultralytics YOLO.train (GPU).
"""
from __future__ import annotations

import math
from pathlib import Path


BASELINE_ARMS = frozenset({"zero_aug", "da_only"})


def resolve_arm_train_dirs(arm: str, tiles_dir: str | Path) -> list[Path]:
    """Train-image dir(s) for an arm. Baselines use raw train tiles; content arms use
    the raw train tiles PLUS their synthetic tiles (dataset.yaml lists both). Content
    arms MUST have Stage-2 tiles — raise otherwise (never silently train a content arm
    on raw tiles only, which would be a zero_aug run mislabeled as that arm)."""
    tiles_dir = Path(tiles_dir)
    train = tiles_dir / "train" / "images"
    if arm in BASELINE_ARMS:
        return [train]
    synth = tiles_dir / "arms" / arm / "images"
    if synth.exists():
        return [train, synth]  # real + synthetic
    raise FileNotFoundError(
        f"arm '{arm}' has no synthetic tiles at {synth} — run the Stage-2 generator "
        f"first (refusing to train it on raw tiles as a mislabeled zero_aug)."
    )


def loss_plateaued(results_csv: str | Path, last_k: int = 5,
                   rel_drop_tol: float = 0.02) -> tuple[bool, dict]:
    """Val-free convergence check from Ultralytics results.csv (train loss).

    Returns (plateaued, info). plateaued=True if the total train loss dropped by less
    than rel_drop_tol (relative) over the last `last_k` epochs — i.e. training flattened
    within the fixed budget. Lets every grid run self-report convergence without val.
    """
    import csv
    rows = list(csv.DictReader(open(results_csv)))
    loss_keys = [k for k in (rows[0].keys() if rows else [])
                 if "train/" in k and "loss" in k]
    if len(rows) < last_k + 1 or not loss_keys:
        return True, {"note": "too few epochs / no loss cols to judge"}
    losses = [sum(float(r[k]) for k in loss_keys) for r in rows]
    window = losses[-last_k - 1:]
    rel_drop = (window[0] - window[-1]) / max(abs(window[0]), 1e-9)
    return (rel_drop < rel_drop_tol,
            {"final_loss": round(losses[-1], 4), "recent_rel_drop": round(rel_drop, 4),
             "last_k": last_k, "n_epochs": len(losses)})


def steps_per_epoch(n_tiles: int, batch: int) -> int:
    return max(1, math.ceil(n_tiles / batch))


def total_steps_from_reference(n_ref_tiles: int, batch: int, base_epochs: int) -> int:
    """Anchor the shared step budget to the reference (Zero-Aug) tile count."""
    return base_epochs * steps_per_epoch(n_ref_tiles, batch)


def epochs_for_budget(n_tiles: int, batch: int, total_steps: int) -> int:
    return max(1, round(total_steps / steps_per_epoch(n_tiles, batch)))


def realized_steps(n_tiles: int, batch: int, epochs: int) -> int:
    return epochs * steps_per_epoch(n_tiles, batch)


def equalized_plan(n_tiles: int, batch: int, total_steps: int, tol: float = 0.02) -> dict:
    """Per-arm epoch plan that matches the shared optimizer-step budget."""
    epochs = epochs_for_budget(n_tiles, batch, total_steps)
    realized = realized_steps(n_tiles, batch, epochs)
    dev = abs(realized - total_steps) / total_steps if total_steps else 0.0
    return {
        "n_tiles": n_tiles, "batch": batch, "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch(n_tiles, batch),
        "epochs": epochs, "realized_steps": realized,
        "deviation": dev, "within_tol": dev <= tol,
    }


def train_arm(dataset_yaml: str | Path, weights: str | Path, project: str | Path,
              name: str, epochs: int, batch: int, imgsz: int, seed: int,
              runtime_aug: dict, device: int | str = 0, val: bool = False,
              workers: int = 16, cache: str | bool = False) -> Path:
    """Train one arm for a FIXED epoch budget. Returns the checkpoint to evaluate.

    val=False (default): skip Ultralytics' per-epoch validation (its best.pt is picked
    by a tile-mAP proxy != our panorama metric, and it costs ~37% of each epoch on the
    25k val tiles). We train a fixed step budget and evaluate the FINAL model (last.pt)
    with our own panorama-level AP. val=True keeps per-epoch val + best.pt (for a
    convergence probe to choose base_epochs).
    runtime_aug keys must be valid Ultralytics augmentation args (fliplr, hsv_h, ...).
    """
    import numpy as np
    import torch
    from ultralytics import YOLO

    # Seed global RNGs before YOLO (Ultralytics seeds its own, but this closes
    # end-to-end reproducibility for any numpy/torch use around it).
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = YOLO(str(weights))
    model.train(
        data=str(dataset_yaml), epochs=epochs, batch=batch, imgsz=imgsz,
        seed=seed, deterministic=True, patience=epochs,  # patience>=epochs => no early stop
        project=str(project), name=name, exist_ok=True, verbose=False,
        device=device, val=val, workers=workers, cache=cache, **runtime_aug,
    )
    # Ask Ultralytics where it saved (it prepends runs/detect/ to a relative project).
    # Fixed-budget training -> evaluate last.pt; best.pt only meaningful with val.
    return Path(model.trainer.best if val else model.trainer.last)
