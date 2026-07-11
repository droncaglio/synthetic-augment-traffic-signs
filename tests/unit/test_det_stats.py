"""Unit tests for detection.stats (paired bootstrap CI of delta-AP)."""
import pytest

from detection.stats import (
    _percentile, ap_small_overall, paired_seed_deltas, bootstrap_delta_ap, ci_excludes_zero,
)

CLASS_NAMES = ["A"]
PIDS = ["p0", "p1", "p2", "p3"]
# 0.01*2048 ~= 20px -> COCO small bucket (ap_small_overall metric)
GTS = {pid: [{"class_id": 0, "box": (0.5, 0.5, 0.01, 0.01)}] for pid in PIDS}
# baseline detects nothing; treatment detects A perfectly in every panorama
BASELINE = {pid: [] for pid in PIDS}
TREATMENT = {pid: [{"class_id": 0, "conf": 0.9, "box": (0.5, 0.5, 0.01, 0.01)}] for pid in PIDS}


def test_percentile():
    v = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(v, 0) == 0.0
    assert _percentile(v, 100) == 4.0
    assert _percentile(v, 50) == 2.0


def test_ap_small_overall_nan_guard():
    assert ap_small_overall({"overall": {"small": {"ap50": float("nan")}}}) == 0.0
    assert ap_small_overall({"overall": {"small": {"ap50": 0.7}}}) == 0.7


def test_paired_seed_deltas_positive():
    out = paired_seed_deltas([BASELINE], [TREATMENT], GTS, CLASS_NAMES, PIDS)
    assert out["deltas"][0] == pytest.approx(1.0)   # 1.0 - 0.0
    assert out["n_positive"] == 1 and out["n_seeds"] == 1


def test_bootstrap_ci_excludes_zero_for_clear_win():
    boot = bootstrap_delta_ap([BASELINE], [TREATMENT], GTS, CLASS_NAMES, PIDS,
                              n_boot=200, seed=0)
    # treatment beats baseline on every panorama -> every replica delta == 1.0
    assert boot["ci_low"] == pytest.approx(1.0)
    assert boot["ci_high"] == pytest.approx(1.0)
    assert ci_excludes_zero(boot)


def test_bootstrap_ci_includes_zero_for_null():
    boot = bootstrap_delta_ap([TREATMENT], [TREATMENT], GTS, CLASS_NAMES, PIDS,
                              n_boot=200, seed=0)  # identical arms -> delta 0
    assert boot["mean"] == pytest.approx(0.0)
    assert not ci_excludes_zero(boot)


def test_bootstrap_deterministic():
    args = ([BASELINE], [TREATMENT], GTS, CLASS_NAMES, PIDS)
    a = bootstrap_delta_ap(*args, n_boot=100, seed=7)
    b = bootstrap_delta_ap(*args, n_boot=100, seed=7)
    assert a == b
