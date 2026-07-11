"""Unit tests for detection.run_naming (arm/seed naming, no fold)."""
import pytest

from detection.run_naming import experiment_name, run_id, parse_run_dir, display_arm


def test_experiment_name_basic():
    assert experiment_name("zero_aug", 42) == "zero_aug_seed42"
    assert experiment_name("copy_paste", 7, budget_tag="bm050") == "copy_paste_bm050_seed7"
    assert experiment_name("da_only", 1, smoke=True) == "da_only_seed1_smoke"


def test_run_id_prefixes_dataset():
    assert run_id("tt100k", "diffusion_bg", 42, budget_tag="bm050") == "tt100k_diffusion_bg_bm050_seed42"


def test_negative_seed_raises():
    with pytest.raises(ValueError):
        experiment_name("zero_aug", -1)


@pytest.mark.parametrize("arm,seed,bm,smoke", [
    ("zero_aug", 42, None, False),
    ("copy_paste", 7, "bm050", False),
    ("bg_photometric", 3, "bm025", True),
    ("diffusion_bg", 100, "bm100", False),
])
def test_parse_roundtrip(arm, seed, bm, smoke):
    name = experiment_name(arm, seed, smoke=smoke, budget_tag=bm)
    parsed = parse_run_dir(name)
    assert parsed == {"arm": arm, "budget_tag": bm, "seed": seed, "smoke": smoke}


def test_parse_does_not_swallow_budget_into_arm():
    parsed = parse_run_dir("real_duplicate_bm050_seed42")
    assert parsed["arm"] == "real_duplicate"
    assert parsed["budget_tag"] == "bm050"


def test_parse_invalid_returns_none():
    assert parse_run_dir("not a run dir") is None       # spaces
    assert parse_run_dir("zero_aug") is None            # missing _seedN
    assert parse_run_dir("zero_aug_seedX") is None      # non-numeric seed


def test_display_arm():
    assert display_arm("copy_paste", "bm050") == "copy_paste_bm050"
    assert display_arm("zero_aug", None) == "zero_aug"
