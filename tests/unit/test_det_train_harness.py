"""Unit tests for detection.train_harness (equalized optimizer-step math)."""
import pytest

from detection.train_harness import (
    steps_per_epoch, total_steps_from_reference, epochs_for_budget,
    realized_steps, equalized_plan, resolve_arm_train_dir,
)


def test_resolve_arm_train_dir(tmp_path):
    # baselines -> raw train tiles
    assert resolve_arm_train_dir("zero_aug", tmp_path) == tmp_path / "train" / "images"
    assert resolve_arm_train_dir("da_only", tmp_path) == tmp_path / "train" / "images"
    # content arm WITHOUT Stage-2 tiles -> hard error (confound guard #2)
    with pytest.raises(FileNotFoundError):
        resolve_arm_train_dir("diffusion_bg", tmp_path)
    # content arm WITH tiles -> returns the combined dir
    d = tmp_path / "arms" / "copy_paste" / "images"
    d.mkdir(parents=True)
    assert resolve_arm_train_dir("copy_paste", tmp_path) == d


def test_steps_per_epoch_ceil():
    assert steps_per_epoch(5000, 16) == 313   # ceil(5000/16)=313
    assert steps_per_epoch(16, 16) == 1
    assert steps_per_epoch(1, 16) == 1        # floor guard -> >=1


def test_reference_budget():
    # Zero-Aug anchor: 5000 tiles, batch 16, 100 base epochs
    total = total_steps_from_reference(5000, 16, 100)
    assert total == 100 * 313


def test_epochs_and_realized_roundtrip():
    total = total_steps_from_reference(5000, 16, 100)   # 31300
    e = epochs_for_budget(5000, 16, total)
    assert e == 100                                     # reference arm -> exactly base_epochs
    assert realized_steps(5000, 16, e) == total


@pytest.mark.parametrize("n_arm", [5000, 6200, 7500, 9000, 4200])
def test_equalized_plan_within_tolerance(n_arm):
    total = total_steps_from_reference(5000, 16, 100)
    plan = equalized_plan(n_arm, 16, total, tol=0.02)
    assert plan["epochs"] >= 1
    assert plan["within_tol"], f"n={n_arm} deviation={plan['deviation']:.4f}"


def test_equalized_plan_bigger_arm_fewer_epochs():
    total = total_steps_from_reference(5000, 16, 100)
    small = equalized_plan(5000, 16, total)   # reference
    big = equalized_plan(9000, 16, total)     # more tiles -> fewer epochs, ~same steps
    assert big["epochs"] < small["epochs"]
    assert abs(big["realized_steps"] - small["realized_steps"]) / total < 0.02
