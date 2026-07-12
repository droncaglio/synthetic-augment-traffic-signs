"""Unit tests for the pure report helpers (aggregation, GT build, run loading)."""
import json

import pytest

from detection.report import gts_by_pid, load_runs, aggregate_arm
from detection.run_naming import experiment_name


def test_aggregate_arm():
    runs = [({"ap_tail": 0.2, "ap_small_macro": 0.5}, {}),
            ({"ap_tail": 0.4, "ap_small_macro": 0.7}, {})]
    a = aggregate_arm(runs)
    assert a["n_seeds"] == 2
    assert a["ap_tail_mean"] == pytest.approx(0.3)
    assert a["ap_small_mean"] == pytest.approx(0.6)
    assert aggregate_arm([]) is None


def test_gts_by_pid_filters_nonsubset_and_unlabelable():
    records = {"p0": {"objects": [
        {"category": "A", "xyxy": [100, 100, 120, 120]},   # subset, labelable
        {"category": "Z", "xyxy": [0, 0, 5, 5]},            # non-subset -> dropped
    ]}}
    g = gts_by_pid(records, ["p0"], {"A": 0}, size=1000)
    assert len(g["p0"]) == 1 and g["p0"][0]["class_id"] == 0
    assert g["p0"][0]["box"] == pytest.approx((0.11, 0.11, 0.02, 0.02))


def test_load_runs_skips_missing_seeds(tmp_path):
    d = tmp_path / experiment_name("real_duplicate", 0, budget_tag="bm050")
    d.mkdir(parents=True)
    (d / "ap_report.json").write_text(json.dumps(
        {"headline": {"ap_tail": 0.3, "ap_small_macro": 0.5}}))
    (d / "dets.json").write_text(json.dumps(
        {"p0": [{"class_id": 0, "conf": 0.9, "box": [0.5, 0.5, 0.1, 0.1]}]}))
    runs = load_runs(tmp_path, "real_duplicate", [0, 1], "bm050")  # seed 1 absent
    assert set(runs) == {0}                 # keyed by seed -> pairs on intersection
    hl, dj = runs[0]
    assert hl["ap_tail"] == 0.3
    assert dj["p0"][0]["box"] == (0.5, 0.5, 0.1, 0.1)  # tuple, ready for bootstrap
