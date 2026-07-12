"""Unit tests for batch_run_det generation-completeness guard."""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "batch_run_det", Path(__file__).resolve().parents[2] / "batch_run_det.py")
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


def _write(tiles, prepared, arm, n_manifest, n_full):
    (tiles / "arms" / arm).mkdir(parents=True, exist_ok=True)
    (tiles / "arms" / arm / "generation_manifest.json").write_text(
        json.dumps({"n_sources": n_manifest}))
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "sources_seed42.json").write_text(
        json.dumps([{"class_id": 0, "source_tile": f"t{i}", "bbox": [0.5, 0.5, 0.1, 0.1]}
                    for i in range(n_full)]))


def test_partial_qa_manifest_is_not_generated(tmp_path):
    """A --limit QA lot (40 of 3787) must NOT count as generated -> grid regenerates."""
    tiles, prepared = tmp_path / "tiles", tmp_path / "prepared"
    _write(tiles, prepared, "diffusion_bg", n_manifest=40, n_full=3787)
    assert batch._arm_generated(str(tiles), str(prepared), "diffusion_bg") is False


def test_complete_manifest_is_generated(tmp_path):
    tiles, prepared = tmp_path / "tiles", tmp_path / "prepared"
    _write(tiles, prepared, "copy_paste", n_manifest=3787, n_full=3787)
    assert batch._arm_generated(str(tiles), str(prepared), "copy_paste") is True


def test_missing_manifest_is_not_generated(tmp_path):
    assert batch._arm_generated(str(tmp_path / "tiles"), str(tmp_path / "prepared"),
                                "real_duplicate") is False
