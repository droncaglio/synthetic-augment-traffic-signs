"""Unit tests for detection.train_harness (equalized optimizer-step math)."""
import pytest

from detection.train_harness import (
    steps_per_epoch, total_steps_from_reference, epochs_for_budget,
    realized_steps, equalized_plan, resolve_arm_train_dirs, loss_plateaued,
)


def test_resolve_arm_train_dirs(tmp_path):
    train = tmp_path / "train" / "images"
    # baselines -> raw train tiles only
    assert resolve_arm_train_dirs("zero_aug", tmp_path) == [train]
    assert resolve_arm_train_dirs("da_only", tmp_path) == [train]
    # content arm WITHOUT Stage-2 tiles -> hard error (confound guard #2)
    with pytest.raises(FileNotFoundError):
        resolve_arm_train_dirs("diffusion_bg", tmp_path)
    # content arm WITH tiles -> real train + synthetic dir
    d = tmp_path / "arms" / "copy_paste" / "images"
    d.mkdir(parents=True)
    assert resolve_arm_train_dirs("copy_paste", tmp_path) == [train, d]


def test_loss_plateaued(tmp_path):
    csv_path = tmp_path / "results.csv"
    hdr = "epoch,train/box_loss,train/cls_loss,train/dfl_loss\n"
    # flat loss -> plateaued
    csv_path.write_text(hdr + "\n".join(f"{i},1.0,1.0,1.0" for i in range(10)))
    ok, _ = loss_plateaued(csv_path)
    assert ok
    # steadily dropping -> NOT plateaued (subtraining)
    csv_path.write_text(hdr + "\n".join(f"{i},{3.0-0.2*i},{3.0-0.2*i},{3.0-0.2*i}"
                                        for i in range(10)))
    ok2, info2 = loss_plateaued(csv_path)
    assert not ok2 and info2["recent_rel_drop"] > 0.02


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
