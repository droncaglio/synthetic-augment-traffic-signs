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
              runtime_aug: dict, device: int = 0) -> Path:
    """Train one arm with a fixed epoch budget (early stopping disabled). Returns best.pt.

    runtime_aug keys must be valid Ultralytics augmentation args (fliplr, hsv_h, ...).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.train(
        data=str(dataset_yaml), epochs=epochs, batch=batch, imgsz=imgsz,
        seed=seed, deterministic=True, patience=epochs,  # patience>=epochs => no early stop
        project=str(project), name=name, exist_ok=True, verbose=False,
        device=device, **runtime_aug,
    )
    return Path(project) / name / "weights" / "best.pt"
