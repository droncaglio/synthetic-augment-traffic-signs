"""Composition of experiment/run names for the detection pipeline.

Single source of truth, used by run_det.py, batch_run_det.py, reproduce.py and
report scripts. Keeps experiment_dir, the status JSON key and the reverse parser
synchronized.

Adapted from the Paper 1 (synthetic-allocation-derma) cls run_naming. Two
differences for detection:
  * ``arm`` replaces ``technique`` (the 6 content-ladder arms).
  * NO fold: detection replicates over seeds only (7 seeds), not k-fold.

Formats:
  experiment_name = "{arm}[_{bm_tag}]_seed{S}[_smoke]"
  run_id          = "{dataset}_" + experiment_name

Reverse parser:
  parse_run_dir("diffusion_bg_bm050_seed42") ->
    {"arm": "diffusion_bg", "budget_tag": "bm050", "seed": 42, "smoke": False}
"""
from __future__ import annotations

import re
from typing import Optional

# arm: lazy [a-z0-9_]+? so it does not swallow the _bm### / _seed suffixes.
_RUN_RX = re.compile(
    r"^(?P<arm>[a-z0-9_]+?)"
    r"(?:_(?P<budget_tag>bm\d{3}))?"
    r"_seed(?P<seed>\d+)"
    r"(?P<smoke>_smoke)?$"
)


def experiment_name(
    arm: str,
    seed: int,
    smoke: bool = False,
    budget_tag: Optional[str] = None,
) -> str:
    """Experiment directory name (under experiments/{dataset}/).

    Returns e.g. "diffusion_bg_bm050_seed42" or "zero_aug_seed7_smoke".

    Raises:
        ValueError: seed < 0.
    """
    if seed < 0:
        raise ValueError(f"experiment_name: seed must be >=0, got {seed}")
    bm_suffix = f"_{budget_tag}" if budget_tag else ""
    smoke_suffix = "_smoke" if smoke else ""
    return f"{arm}{bm_suffix}_seed{seed}{smoke_suffix}"


def run_id(
    dataset: str,
    arm: str,
    seed: int,
    smoke: bool = False,
    budget_tag: Optional[str] = None,
) -> str:
    """Unique run key for the batch status JSON (dataset + experiment_name)."""
    return f"{dataset}_{experiment_name(arm, seed, smoke, budget_tag)}"


def parse_run_dir(dirname: str) -> Optional[dict]:
    """Reverse parser: experiment directory name -> dict of fields.

    Returns {arm, budget_tag (None if absent), seed, smoke} or None if no match.

    Examples:
        >>> parse_run_dir("zero_aug_seed42")
        {'arm': 'zero_aug', 'budget_tag': None, 'seed': 42, 'smoke': False}
        >>> parse_run_dir("copy_paste_bm050_seed7_smoke")
        {'arm': 'copy_paste', 'budget_tag': 'bm050', 'seed': 7, 'smoke': True}
    """
    m = _RUN_RX.match(dirname)
    if not m:
        return None
    return {
        "arm": m["arm"],
        "budget_tag": m["budget_tag"],
        "seed": int(m["seed"]),
        "smoke": bool(m["smoke"]),
    }


def display_arm(arm: str, budget_tag: Optional[str]) -> str:
    """Arm name for display in reports (arm or arm_bm###)."""
    return f"{arm}_{budget_tag}" if budget_tag else arm
