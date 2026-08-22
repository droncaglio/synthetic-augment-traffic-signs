#!/usr/bin/env python
"""
reproduce.py — single orchestrator to reproduce Paper 3 (WVC) end to end.

Long-tail traffic-sign detection with a budget-matched cost ladder of 7 augmentation
arms — zero_aug (No-Aug), da_only (Standard-Aug), real_duplicate, photometric_full,
copy_paste, diffusion_bg, signgen_controlnet (SignGen) — at K=0.5, with panorama-
reconstruction + global-NMS evaluation and optimizer steps equalized across arms.
The full paper grid is 2 datasets (TT100K, DFG) x 2 detectors (yolo11n, yolo11s) x
7 arms x 14 seeds. Each (dataset x detector) is a CELL with isolated tiles /
experiments / status / report paths; `--dataset`/`--model` pick a cell (default: all).
One command runs the whole grid, cell by cell:
  check -> [per cell] download -> prepare -> generate -> train -> report.

Steps:
  check     — validate the environment (conda, GPU, core deps, AND the diffusion
              extras: bitsandbytes + HF_TOKEN in .env — the arm that costs ~17h, so
              we fail fast before it, not 15h in).
  download  — verify the TT100K 2021 raw files; auto-fetch the zip if missing (direct
              URL), else print instructions.
  prepare   — the data spine (idempotent): prepare_tt100k -> select_subset ->
              make_splits (panorama split + pHash near-dup + assert_no_leak) ->
              tile_panoramas {train,val,test} -> build_allocation.
  generate  — materialize the content arms' synthetic tiles (diffusion resumes; its
              scanner = the trained zero_aug seed 0). Optional: `train` also auto-preps
              each arm on demand, so `generate` is only for running the long diffusion
              pass on its own / overnight.
  train     — batch_run_det.py (embeds per-arm generation; resumable status JSON).
  report    — det_report.py: per-arm AP + paired-bootstrap CIs for the primary
              contrasts (metric = the reported macro), on the chosen eval split.
  all       — check -> download -> prepare -> train -> report (generation happens
              inside train).

Filters / modes:
  --arm A [A ...]   restrict train/generate to a subset of arms.
  --seeds S [S ...] restrict to a subset of seeds (default: the batch config's).
  --smoke           1 seed, 2 epochs, arms {zero_aug, real_duplicate} — validates the
                    spine end to end (numbers NOT valid; diffusion excluded — too slow).
  --force           re-run prepare steps whose outputs already exist.
  --dry-run         print what would run without executing.
  --skip-download   assume the raw files are already present.
  --clean           delete generated outputs (experiments/, reports/det, batch_status)
                    for a from-scratch run (keeps raw + prepared). Asks unless --yes.
  --device          GPU index (default 0).   --eval-split {val,test} for report.

Examples:
  python reproduce.py                                   # all cells, all steps
  python reproduce.py --dataset tt100k --model yolo11n  # one cell, full pipeline
  python reproduce.py --smoke                           # fast spine validation (tt100k/11n)
  python reproduce.py --step prepare --dataset dfg      # build the DFG data spine only
  python reproduce.py --step generate --arm diffusion_bg   # the ~17h diffusion pass
  python reproduce.py --step report --eval-split test   # final contrasts on test
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from detection.notifications.telegram import LogCapture, TelegramNotifier, load_env  # noqa: E402

CONDA_ENV = "augment-traffic-signs"
VALID_STEPS = ["check", "download", "prepare", "build_masks", "generate", "train", "report", "all"]

# The paper's cost ladder — 7 arms (paper name in parentheses):
#   zero_aug (No-Aug), da_only (Standard-Aug), real_duplicate (Real-Duplicate),
#   photometric_full (Photometric-Full), copy_paste (Copy-Paste),
#   diffusion_bg (Diffusion-BG), signgen_controlnet (SignGen).
ALL_ARMS = ["zero_aug", "da_only", "real_duplicate", "photometric_full",
            "copy_paste", "diffusion_bg", "signgen_controlnet"]
CONTENT_ARMS = ["real_duplicate", "photometric_full", "copy_paste",
                "diffusion_bg", "signgen_controlnet"]
DEFAULT_SEEDS = list(range(14))            # the paper uses 14 seeds
DATASETS = ["tt100k", "dfg"]               # two long-tailed benchmarks
MODELS = ["yolo11n", "yolo11s"]            # two detector capacities
BATCH_YAML = PROJECT_ROOT / "configs" / "detection" / "batches" / "full_grid_det.yaml"

# TT100K 2021 raw is auto-downloadable. DFG images come from scripts/fetch_dfg.py; DFG
# train.json/test.json (annotations) are user-provided under data/dfg/ (see README).
TT100K_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/tt100k_2021.zip"
TT100K_ZIP = PROJECT_ROOT / "data" / "tt100k" / "tt100k_2021.zip"

# ─── per-cell paths (set by configure_cell for each dataset x detector) ─────────
DATASET = MODEL = CELL = None
RAW_DIR = RAW_ANNOTATIONS = PREPARED = TILES = PROJECT_EXP = STATUS_FILE = REPORT_DIR = None


def configure_cell(dataset: str, model: str) -> None:
    """Point every per-cell path at one (dataset x detector) cell, so cells have
    isolated tiles / experiments / status / report paths and never clobber each other."""
    global DATASET, MODEL, CELL, RAW_DIR, RAW_ANNOTATIONS, PREPARED, TILES
    global PROJECT_EXP, STATUS_FILE, REPORT_DIR
    DATASET, MODEL, CELL = dataset, model, f"{dataset}_{model}"
    base = PROJECT_ROOT / "data" / dataset
    PREPARED = base / "prepared"
    TILES = base / "tiles"
    if dataset == "tt100k":
        RAW_DIR = base / "tt100k_2021"
        RAW_ANNOTATIONS = RAW_DIR / "annotations_all.json"
    else:  # dfg: raw root holds train.json/test.json; images live under data/dfg/images
        RAW_DIR = base
        RAW_ANNOTATIONS = base / "train.json"
    PROJECT_EXP = PROJECT_ROOT / "experiments" / CELL
    STATUS_FILE = PROJECT_ROOT / f"batch_status_{CELL}.json"
    REPORT_DIR = PROJECT_ROOT / "reports" / "det" / CELL


configure_cell("tt100k", "yolo11n")   # default cell (backward compatible)


# ─── logging ───────────────────────────────────────────────────────────────────
def log(msg: str, prefix: str = "•") -> None:
    print(f"{prefix} {msg}", flush=True)


def section(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def ok(m): log(m, "✓")
def warn(m): log(m, "⚠")
def fail(m): log(m, "✗")


def run_cmd(cmd: list, dry_run: bool = False) -> int:
    log("$ " + " ".join(str(c) for c in cmd), "→")
    if dry_run:
        log("(dry-run — not executed)", " ")
        return 0
    return subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT)).returncode


# ─── check ─────────────────────────────────────────────────────────────────────
def step_check() -> bool:
    section("CHECK — environment")
    all_ok = True
    pv = sys.version_info
    (ok if pv >= (3, 11) else warn)(f"Python {pv.major}.{pv.minor}.{pv.micro}")

    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env == CONDA_ENV:
        ok(f"conda env: {env}")
    else:
        warn(f"conda env: {env or '(none)'} — expected {CONDA_ENV} (conda activate {CONDA_ENV})")

    missing = []
    for pkg in ["torch", "ultralytics", "yaml", "numpy", "pandas", "scipy", "cv2",
                "imagehash", "albumentations"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        fail(f"core packages missing: {missing} — conda env update -f env/environment.yml")
        all_ok = False
    else:
        ok("core packages OK (torch, ultralytics, imagehash, albumentations, ...)")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.device_count()}x — primary: {torch.cuda.get_device_name(0)}")
        else:
            warn("GPU NOT available — training/diffusion will be unusably slow on CPU")
    except Exception as e:  # noqa: BLE001
        warn(f"GPU check failed: {e}")

    # diffusion extras — fail fast BEFORE the ~17h diffusion pass, not 15h in.
    dif_missing = [p for p in ("diffusers", "bitsandbytes") if not _importable(p)]
    if dif_missing:
        warn(f"diffusion extras missing: {dif_missing} — the diffusion_bg arm will fail "
             f"(pip install {' '.join(dif_missing)}). OK to ignore if you skip diffusion_bg.")
    else:
        ok("diffusion extras OK (diffusers, bitsandbytes)")
    load_env()  # pull HF_TOKEN from .env into the environment
    if os.environ.get("HF_TOKEN", "").strip():
        ok("HF_TOKEN present (.env) — FLUX.1-Fill download will authenticate")
    else:
        warn("HF_TOKEN absent — diffusion_bg cannot pull gated FLUX.1-Fill-dev. Add it to "
             ".env (see .env.example) and accept the model license on HuggingFace.")

    (ok if BATCH_YAML.exists() else fail)(f"batch YAML: {BATCH_YAML.relative_to(PROJECT_ROOT)}")
    all_ok = all_ok and BATCH_YAML.exists()
    return all_ok


def _importable(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:  # noqa: BLE001
        return False


# ─── download ──────────────────────────────────────────────────────────────────
def step_download(dry_run: bool) -> bool:
    section(f"DOWNLOAD — {DATASET} raw files")
    return _download_tt100k(dry_run) if DATASET == "tt100k" else _download_dfg(dry_run)


def _download_tt100k(dry_run: bool) -> bool:
    if RAW_ANNOTATIONS.exists():
        ok(f"raw present: {RAW_ANNOTATIONS.relative_to(PROJECT_ROOT)}")
        return True
    warn(f"missing: {RAW_ANNOTATIONS.relative_to(PROJECT_ROOT)}")
    if dry_run:
        log(f"would download {TT100K_URL} -> {TT100K_ZIP.relative_to(PROJECT_ROOT)} + unzip", "→")
        return True
    TT100K_ZIP.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not TT100K_ZIP.exists():
            log(f"downloading {TT100K_URL} (~large, minutes) ...", "→")
            urllib.request.urlretrieve(TT100K_URL, TT100K_ZIP)
        log(f"unzipping -> {TT100K_ZIP.parent.relative_to(PROJECT_ROOT)}", "→")
        rc = run_cmd(["unzip", "-q", "-o", str(TT100K_ZIP), "-d", str(TT100K_ZIP.parent)])
        if rc != 0 or not RAW_ANNOTATIONS.exists():
            raise RuntimeError("unzip did not yield annotations_all.json")
        ok("TT100K raw ready")
        return True
    except Exception as e:  # noqa: BLE001
        fail(f"auto-download failed ({e}). Fetch manually:\n"
             f"    {TT100K_URL}\n"
             f"  extract so that {RAW_ANNOTATIONS.relative_to(PROJECT_ROOT)} exists.")
        return False


def _download_dfg(dry_run: bool) -> bool:
    """DFG images auto-fetch via scripts/fetch_dfg.py (~7.5 GB, no auth). The DFG-TSD
    train.json/test.json annotations are user-provided under data/dfg/ (see README)."""
    train_j, test_j, imgs = RAW_DIR / "train.json", RAW_DIR / "test.json", RAW_DIR / "images"
    if not (train_j.exists() and test_j.exists()):
        fail(f"DFG annotations missing — place train.json + test.json under "
             f"{RAW_DIR.relative_to(PROJECT_ROOT)}/ (the DFG-TSD split; see README).")
        return False
    if imgs.exists() and any(imgs.glob("*.jpg")):
        ok(f"DFG raw present: {RAW_DIR.relative_to(PROJECT_ROOT)}")
        return True
    if dry_run:
        log("would run scripts/fetch_dfg.py (downloads ~7.5 GB DFG images)", "→")
        return True
    if run_cmd(["python", "scripts/fetch_dfg.py", "--root", str(RAW_DIR)]) != 0:
        fail("DFG image fetch failed — run scripts/fetch_dfg.py manually.")
        return False
    ok("DFG raw ready")
    return True


# ─── prepare (data spine) ──────────────────────────────────────────────────────
def _step(cmd: list, sentinel: Path, label: str, force: bool, dry_run: bool) -> bool:
    if sentinel.exists() and not force:
        ok(f"{label}: already done ({sentinel.relative_to(PROJECT_ROOT)})")
        return True
    log(f"{label} ...")
    rc = run_cmd(cmd + (["--force"] if force and "--force" not in cmd else []), dry_run=dry_run)
    if rc != 0:
        fail(f"{label} failed (rc={rc})")
        return False
    ok(f"{label} done")
    return True


def step_prepare(force: bool, dry_run: bool) -> bool:
    section(f"PREPARE — {DATASET} data spine")
    S = "scripts/detection"
    subset = PREPARED / "subset.json"
    if DATASET == "tt100k":
        # full-201 subset (make_full_subset), not the pre-pivot 21-class select_subset:
        # the paper trains on ALL annotated classes to remove the open-set FP artifact.
        chain = [
            (["python", f"{S}/prepare_tt100k.py", "--annotations", str(RAW_ANNOTATIONS),
              "--out", str(PREPARED)], PREPARED / "catalog.json", "prepare_tt100k"),
            (["python", f"{S}/make_full_subset.py", "--prepared", str(PREPARED),
              "--out", str(subset)], subset, "make_full_subset (full-201)"),
            (["python", f"{S}/make_splits.py", "--prepared", str(PREPARED), "--raw", str(RAW_DIR),
              "--out", str(PREPARED / "splits.json")], PREPARED / "splits.json", "make_splits"),
        ]
    else:  # dfg: prepare_dfg writes panoramas/catalog/splits (official split); then full-201.
        chain = [
            (["python", f"{S}/prepare_dfg.py", "--raw", str(RAW_DIR), "--out", str(PREPARED),
              "--val-frac", "0.15"], PREPARED / "splits.json", "prepare_dfg"),
            (["python", f"{S}/make_full_subset.py", "--prepared", str(PREPARED),
              "--out", str(subset)], subset, "make_full_subset (full-201)"),
        ]
    for cmd, sentinel, label in chain:
        if not _step(cmd, sentinel, label, force, dry_run):
            return False
    for split in ("train", "val", "test"):
        cmd = ["python", f"{S}/tile_panoramas.py", "--split", split, "--prepared", str(PREPARED),
               "--raw", str(RAW_DIR), "--out", str(TILES)]
        if not _step(cmd, TILES / split / "images", f"tile_panoramas[{split}]", force, dry_run):
            return False
    if not _step(["python", f"{S}/build_allocation.py", "--prepared", str(PREPARED),
                  "--out", str(PREPARED / "allocation.json"), "--K", "0.5"],
                 PREPARED / "allocation.json", "build_allocation", force, dry_run):
        return False
    ok("data spine ready (prepared/ + tiles/ + allocation.json)")
    return True


# ─── generate (explicit content-arm materialization) ──────────────────────────
def step_build_masks(device: str, dry_run: bool) -> bool:
    """Precompute SAM sign masks (opt-in; only the mask arms need them). Idempotent/resumível."""
    section("BUILD_MASKS — SAM sign masks (for copy_paste_mask & future mask arms)")
    cmd = ["python", "scripts/detection/precompute_sam_masks.py",
           "--tiles", str(TILES), "--out", str(TILES.parent / "masks"), "--resume"]
    if device:
        cmd += ["--device", device]
    if run_cmd(cmd, dry_run=dry_run) != 0:
        fail("SAM mask precompute failed")
        return False
    ok("SAM masks precomputed")
    return True


def step_generate(arms: list, device: str, scan_weights: Optional[str], dry_run: bool) -> bool:
    section("GENERATE — content-arm synthetic tiles")
    S = "scripts/detection"
    todo = [a for a in arms if a in CONTENT_ARMS]
    if not todo:
        warn("no content arms selected (baselines need no generation).")
        return True
    for arm in todo:
        if _arm_complete(arm):
            ok(f"{arm}: already generated")
            continue
        cmd = ["python", f"{S}/generate_arm.py", "--arm", arm,
               "--tiles", str(TILES), "--prepared", str(PREPARED)]
        if arm == "diffusion_bg":
            sc = scan_weights or _default_scanner()
            if not sc and not dry_run:
                fail("diffusion_bg needs a scanner — pass --scan-weights or train zero_aug first.")
                return False
            cmd += ["--scan-weights", sc or "<zero_aug seed0 @ runtime>", "--resume",
                    "--device", device]
        if run_cmd(cmd, dry_run=dry_run) != 0:
            fail(f"generation failed for {arm}")
            return False
    ok("content arms generated")
    return True


def _arm_complete(arm: str, seed: int = 42) -> bool:
    """True only if the arm's generation manifest is COMPLETE (n_sources == full source
    list) — a partial --limit QA lot must not count as generated."""
    mf = TILES / "arms" / arm / "generation_manifest.json"
    if not mf.exists():
        return False
    n_have = json.loads(mf.read_text()).get("n_sources", 0)
    src = PREPARED / f"sources_seed{seed}.json"
    return n_have > 0 if not src.exists() else n_have >= len(json.loads(src.read_text()))


def _default_scanner() -> Optional[str]:
    for w in ("last.pt", "best.pt"):
        p = PROJECT_EXP / "zero_aug_bm050_seed0" / "weights" / w
        if p.exists():
            return str(p)
    return None


# ─── train ─────────────────────────────────────────────────────────────────────
def step_train(arms: Optional[list], seeds: Optional[list], base_epochs: int,
               device: str, scan_weights: Optional[str], dry_run: bool) -> bool:
    section(f"TRAIN — batch_run_det.py ({CELL}, generation embedded)")
    cmd = ["python", "batch_run_det.py", "--batch", str(BATCH_YAML),
           "--device", device, "--base-epochs", str(base_epochs),
           "--project", str(PROJECT_EXP), "--tiles", str(TILES), "--prepared", str(PREPARED),
           "--model", MODEL, "--status-file", str(STATUS_FILE)]
    if arms:
        cmd += ["--arms", *arms]
    if seeds:
        cmd += ["--seeds", *[str(s) for s in seeds]]
    if scan_weights:
        cmd += ["--scan-weights", scan_weights]
    if dry_run:
        cmd.append("--dry-run")
    rc = run_cmd(cmd, dry_run=False)  # batch_run_det handles its own dry-run flag
    if rc != 0:
        fail(f"batch_run_det rc={rc} — check {STATUS_FILE.name}")
        return False
    ok("training grid finished")
    return True


# ─── report ────────────────────────────────────────────────────────────────────
def step_report(eval_split: str, seeds: Optional[list], dry_run: bool) -> bool:
    section(f"REPORT — det_report.py ({CELL}, eval={eval_split})")
    cmd = ["python", "scripts/detection/det_report.py", "--project", str(PROJECT_EXP),
           "--prepared", str(PREPARED), "--eval-split", eval_split, "--out", str(REPORT_DIR),
           "--seeds", *[str(s) for s in (seeds or DEFAULT_SEEDS)]]
    if run_cmd(cmd, dry_run=dry_run) != 0:
        fail("det_report failed")
        return False
    ok(f"report ready: {REPORT_DIR.relative_to(PROJECT_ROOT)}/report.md")
    return True


# ─── clean ─────────────────────────────────────────────────────────────────────
def step_clean(dry_run: bool, assume_yes: bool) -> bool:
    section("CLEAN — delete generated outputs (keeps raw + prepared)")
    targets = [p for p in (PROJECT_EXP, REPORT_DIR, STATUS_FILE, TILES / "arms") if p.exists()]
    if not targets:
        ok("nothing to clean")
        return True
    print("Will be DELETED:")
    for t in targets:
        print(f"  - [{'dir ' if t.is_dir() else 'file'}] {t.relative_to(PROJECT_ROOT)}")
    print("Preserved: data/tt100k/tt100k_2021 (raw), prepared/, tiles/{train,val,test}\n")
    if dry_run:
        warn("dry-run — nothing deleted")
        return True
    if not assume_yes and input("Confirm deletion? type 'yes': ").strip().lower() != "yes":
        warn("clean cancelled")
        return False
    for t in targets:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
        ok(f"removed {t.relative_to(PROJECT_ROOT)}")
    return True


# ─── step wrapper (Telegram + exceptions) ──────────────────────────────────────
def _run_step(notifier, name: str, fn: Callable[[], bool], fatal: bool, dry_run: bool):
    if not dry_run:
        notifier.send_message(f"▶️ <b>STEP: {name}</b>")
    t0 = time.time()
    try:
        good = fn()
    except Exception as exc:  # noqa: BLE001
        fail(f"step {name} crashed: {exc!r}")
        if not dry_run:
            notifier.send_message(f"❌ <b>STEP FAILED: {name}</b>\n<pre>{repr(exc)[:400]}</pre>")
        return False, fatal
    if not dry_run:
        icon, lbl = ("✅", "DONE") if good else ("❌", "FAILED")
        notifier.send_message(f"{icon} <b>STEP {lbl}: {name}</b> "
                              f"({int(time.time() - t0)}s)")
    return good, (fatal and not good)


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Paper 3 (traffic-sign detection) end to end.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--step", choices=VALID_STEPS, default="all")
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all",
                   help="which dataset cell(s) to run (default: all)")
    p.add_argument("--model", choices=[*MODELS, "all"], default="all",
                   help="which detector cell(s) to run (default: all)")
    p.add_argument("--arm", dest="arms", nargs="+", default=None,
                   help=f"restrict to a subset of arms {ALL_ARMS}")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--base-epochs", type=int, default=25)
    p.add_argument("--eval-split", choices=["val", "test"], default="test")
    p.add_argument("--scan-weights", default=None, help="diffusion_bg scanner override")
    p.add_argument("--device", default="0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    step = args.step
    arms, seeds, base_epochs = args.arms, args.seeds, args.base_epochs
    if args.smoke:  # fast spine validation: exclude the slow diffusion arm
        arms = arms or ["zero_aug", "real_duplicate"]
        seeds = seeds or [0]
        base_epochs = 2

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models = MODELS if args.model == "all" else [args.model]
    if args.smoke:  # smoke validates a single cell's spine only
        datasets, models = ["tt100k"], ["yolo11n"]
    cells = [(d, m) for d in datasets for m in models]

    section("REPRODUCE — synthetic-augment-traffic-signs (Paper 3 / WVC)")
    for k, v in [("step", step), ("cells", ", ".join(f"{d}/{m}" for d, m in cells)),
                 ("arms", arms or "all"), ("seeds", seeds or "config"),
                 ("base_epochs", base_epochs), ("eval_split", args.eval_split),
                 ("smoke", args.smoke), ("force", args.force), ("dry_run", args.dry_run)]:
        log(f"{k:12}: {v}")

    notifier = TelegramNotifier.from_env()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"reproduce_{step}_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rc_all = 0

    with LogCapture(log_path, notifier, label=f"reproduce_{step}_{ts}"):
        if not args.dry_run:
            notifier.send_separator()
            notifier.send_message(f"🧪 <b>REPRODUCE START</b>{' [SMOKE]' if args.smoke else ''}\n"
                                  f"<code>{step}</code> · cells={len(cells)} · arms={arms or 'all'}")

        # The environment check is cell-independent — run it once, up front.
        if step in ("check", "all"):
            _run_step(notifier, "check", step_check, False, args.dry_run)

        for dataset, model in cells:
            configure_cell(dataset, model)
            section(f"CELL — {dataset} / {model}")

            if args.clean:
                _, abort = _run_step(notifier, f"clean[{CELL}]",
                                     lambda: step_clean(args.dry_run, args.yes), True, args.dry_run)
                if abort:
                    return 1

            if step in ("download", "all") and not args.skip_download:
                good, _ = _run_step(notifier, f"download[{CELL}]",
                                    lambda: step_download(args.dry_run),
                                    fatal=(step == "download"), dry_run=args.dry_run)
                if not good and step == "download":
                    return 1
                if not good:
                    warn("raw files missing — later steps may fail")
                    rc_all = 1

            if step in ("prepare", "all"):
                _, abort = _run_step(notifier, f"prepare[{CELL}]",
                                     lambda: step_prepare(args.force, args.dry_run), True, args.dry_run)
                if abort:
                    return 1

            if step == "build_masks":  # explicit/opt-in — only mask arms need SAM masks
                _, abort = _run_step(notifier, f"build_masks[{CELL}]",
                                     lambda: step_build_masks(args.device, args.dry_run),
                                     True, args.dry_run)
                if abort:
                    return 1

            if step == "generate":  # explicit only; `all`/`train` auto-prep inside the batch
                _, abort = _run_step(notifier, f"generate[{CELL}]",
                                     lambda: step_generate(arms or ALL_ARMS, args.device,
                                                           args.scan_weights, args.dry_run),
                                     True, args.dry_run)
                if abort:
                    return 1

            if step in ("train", "all"):
                _, abort = _run_step(notifier, f"train[{CELL}]",
                                     lambda: step_train(arms, seeds, base_epochs, args.device,
                                                        args.scan_weights, args.dry_run),
                                     True, args.dry_run)
                if abort:
                    return 1

            if step in ("report", "all"):
                if args.smoke:
                    warn("SMOKE — report skipped (2-epoch/1-seed runs are not valid numbers)")
                else:
                    good, _ = _run_step(notifier, f"report[{CELL}]",
                                        lambda: step_report(args.eval_split, seeds, args.dry_run),
                                        False, args.dry_run)
                    if not good:
                        rc_all = 1

        section("DONE")
        log(f"cells       : {', '.join(f'{d}/{m}' for d, m in cells)}")
        log("reports     : reports/det/<dataset>_<model>/report.md")
        if not args.dry_run:
            icon = "✅" if rc_all == 0 else "⚠️"
            notifier.send_message(f"{icon} <b>REPRODUCE DONE</b> <code>{step}</code> · "
                                  f"{len(cells)} cell(s)")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
