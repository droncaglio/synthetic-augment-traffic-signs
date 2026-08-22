# Data Augmentation for Long-Tail Traffic-Sign Detection: A Budget-Matched Cost Ladder

Code and reproduction pipeline for the paper of the same topic (under double-blind
review). The repository reproduces every table and figure end to end, from the raw
public datasets to the final consolidated per-cell reports.

## What the paper studies

Traffic signs follow a severe long-tailed distribution, and data augmentation is the
usual remedy — but *which kind is worth its cost on the tail* is rarely measured under
controlled conditions. We place augmentation strategies on a single **cost ladder**:
seven arms that share one added-instance budget and one training protocol, so that only
the augmentation *mechanism* and its *generation cost* vary.

- **Cost ladder (7 arms; paper name in parentheses):**
  `zero_aug` (No-Aug) · `da_only` (Standard-Aug) · `real_duplicate` (Real-Duplicate) ·
  `photometric_full` (Photometric-Full) · `copy_paste` (Copy-Paste) ·
  `diffusion_bg` (Diffusion-BG) · `signgen_controlnet` (SignGen).
- **Two long-tailed benchmarks:** TT100K (Tsinghua-Tencent 100K) and DFG-TSD.
- **Two detector capacities:** YOLO11n and YOLO11s.
- **14 seeds**, all contrasts paired by seed, Holm-corrected within each
  dataset×detector cell.

**Metric:** AP-tail (macro AP@0.5 over the *evaluable tail*), computed on the
reconstructed panorama (per-tile predictions merged + global NMS), all object sizes.

## Quickstart

```bash
# 1. Environment (diffusion deps are heavy — dedicated env)
conda env create -f env/environment.yml
conda activate augment-traffic-signs

# 2. Reproduce everything: 2 datasets x 2 detectors x 7 arms x 14 seeds
python reproduce.py                 # all cells: check -> download -> prepare -> train -> report
```

`reproduce.py` orchestrates the whole grid **cell by cell** (a *cell* = one
dataset × one detector). Steps run in order and are resumable; every step can be run in
isolation and every cell filtered:

```bash
python reproduce.py --dataset tt100k --model yolo11n   # one cell, full pipeline
python reproduce.py --dataset dfg    --model yolo11s    # another cell
python reproduce.py --smoke                             # 1 seed / 2 epochs spine check (no diffusion)
python reproduce.py --step prepare  --dataset dfg       # build the DFG data spine only
python reproduce.py --step report   --eval-split test   # final contrasts on the test split
python reproduce.py --dry-run                           # print the plan, run nothing
python reproduce.py --clean --dataset tt100k --model yolo11n   # wipe a cell's generated outputs
```

| step | what it does |
|---|---|
| `check` | validate env / GPU / core deps **and** the diffusion extras (`diffusers`, `bitsandbytes`, `HF_TOKEN`) — fails fast before the long passes |
| `download` | TT100K auto-downloads; DFG images fetch via `scripts/fetch_dfg.py` (annotations user-provided, see below) |
| `prepare` | idempotent data spine: full-catalog subset → leak-safe split → 640-px tiling → water-filling allocation (K=0.5) |
| `train` | `batch_run_det.py` — the arm×seed grid with optimizer steps equalized across arms; per-cell resumable status JSON; content-arm generation is embedded and resumes |
| `report` | `det_report.py` — per-arm AP-tail + within-cell contrasts with paired CIs, Holm-corrected |

Per cell the grid is 7 arms × 14 seeds = **98 training runs**; the full paper is four
cells (**392 runs**) plus one-off synthetic generation. Generation dominates wall-clock
(diffusion is hours; copy-paste/oversample are seconds) — see the paper's cost analysis.
All experiments were run on a single NVIDIA RTX 4500 Ada (24 GB). Use
`--dataset`/`--model`/`--arm`/`--smoke` to scope a partial run first.

## The two datasets

Neither dataset is redistributed here (only code and configs are versioned).

| Dataset | Source | Obtaining it |
|---|---|---|
| **TT100K 2021** | [Tsinghua-Tencent 100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/) | auto-downloaded by `reproduce.py --step download` |
| **DFG-TSD** | [DFG traffic-sign dataset](https://www.vicos.si/resources/dfg/) | images via `scripts/fetch_dfg.py` (~7.5 GB, CC BY-NC-SA); place the `train.json` / `test.json` annotations under `data/dfg/` |

DFG images extract flat into `data/dfg/images/`; its official train/test split lives in
the annotation JSON, from which we carve a seeded validation slice.

**Diffusion arms need a Hugging Face token.** `diffusion_bg` uses the *gated*
`black-forest-labs/FLUX.1-Fill-dev`, which requires accepting the model license and an
`HF_TOKEN`. Copy `.env.example` to `.env` at the repo root and fill in `HF_TOKEN`;
`reproduce.py --step check` verifies it and fails fast otherwise. `signgen_controlnet`
uses Stable Diffusion 1.5 + a Canny ControlNet and trains its own class verifier on
demand. The cheap arms (oversample / photometric / copy-paste) need none of this.
(`setup_env.py` is a separate helper for the optional Telegram run-notifications.)

## What `reproduce.py` produces

- `reports/det/<dataset>_<model>/report.md` — per-cell consolidated results: per-arm
  AP-tail (mean ± sd over 14 seeds) and the within-cell contrasts with paired CIs and
  Holm-corrected p-values.
- `experiments/<dataset>_<model>/<arm>_<seed>/` — per-run checkpoints, predictions, and
  `ap_report.json`.
- `batch_status_<dataset>_<model>.json` — resumable grid status per cell.

## Paper ↔ code map

| Paper artifact | Produced by |
|---|---|
| Table 1 (AP-tail, mean ± sd) | `det_report.py` → `reports/det/<cell>/report.md` |
| Table 2 (within-cell contrasts, Holm) | `det_report.py` (paired difference + CI + Holm) |
| Capacity-by-reuse / capacity-by-diversity interaction | `scripts/detection/interaction_generative.py` |
| Head / overall AP (head-cost check, Sec. Results) | `scripts/detection/head_overall.py` |
| Cost-ladder sample figure | `scripts/detection/paper_samples.py` |
| Panorama-reconstruction + global-NMS evaluation | `run_det.py` + `src/detection/{reconstruct,evaluate}.py` |
| Water-filling budget allocation (K=0.5, α=3) | `scripts/detection/build_allocation.py` + `src/detection/budget.py` |
| Leak-safe split (pHash groups / official split) | `scripts/detection/make_splits.py`, `prepare_dfg.py` + `src/detection/splits.py` |

## Repository layout

```
reproduce.py                       # single end-to-end orchestrator (per-cell)
run_det.py / batch_run_det.py      # single-run / batch training + panorama eval
configs/detection/arm/             # one YAML per augmentation arm
configs/detection/batches/         # full_grid_det.yaml = the paper grid (7 arms x 14 seeds)
configs/detection/model/           # yolo11n.yaml, yolo11s.yaml
src/detection/                     # importable library: splits, tiling, reconstruct,
                                   #   evaluate, stats, budget, generators/
scripts/detection/                 # CLIs: prepare_tt100k, prepare_dfg, make_full_subset,
                                   #   make_splits, tile_panoramas, build_allocation,
                                   #   generate_arm, det_report, interaction_generative, ...
scripts/fetch_dfg.py               # DFG image downloader
env/environment.yml                # conda environment
tests/unit/                        # unit suite (run with pytest)
```

## Reproducibility notes

- **Seeds.** Splits use a fixed `seed 42`; training uses 14 seeds (`0..13`), and every
  contrast is paired by seed.
- **Closed-set, full catalog.** Detectors train on *every* annotated class (this removes
  an open-set false-positive artifact seen with a small hand-picked subset). The
  **evaluable tail** is the set of classes that are rare yet learnable and measurable
  (≥10 and <80 train instances, ≥5 test instances); only it enters the metric.
- **Leakage control.** TT100K panoramas are split at the panorama level with near-
  duplicate groups (perceptual-hash Hamming ≤ 5) assigned wholesale to one split; DFG
  keeps its official train/test split with a seeded validation slice. `assert_no_leak`
  guards train/val/test disjointness.
- **Variable image size.** Tiling, reconstruction, and evaluation normalize by each
  image's own dimensions; the `2048` panorama size is a TT100K-specific fallback, so
  DFG's variable-size images (~1920×1080) are handled correctly.
- **Fair training.** Adding instances enlarges the training set, so epochs per arm are
  chosen to equalize the total optimizer-step budget; early stopping is disabled. Any
  cost difference between arms therefore comes from generation, not training.
- **Evaluation.** AP@0.5 with class-wise 101-point interpolation on the reconstructed
  panorama after global NMS (IoU 0.5). Contrasts are read only within a dataset×detector
  cell, never across.

## Tests

```bash
pytest tests/unit
```

The suite covers the integrity-critical logic: leak-safe split loading, tiling and
panorama reconstruction, the budget/allocation tag logic, metric and AP computation, the
run/experiment naming, batch fold expansion, and the generators (copy-paste, photometric,
diffusion, SignGen) and their manifests.

## License

Code released under the MIT License (see `LICENSE`). The datasets retain their original
licenses (TT100K; DFG-TSD is CC BY-NC-SA 4.0, academic/non-commercial) — consult each
source before use. Model weights downloaded at runtime (FLUX.1-Fill, Stable Diffusion 1.5,
ControlNet, YOLO11) retain their respective licenses.

## Citation

A citation entry will be added on publication. (Anonymous during review.)
