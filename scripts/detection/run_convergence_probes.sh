#!/usr/bin/env bash
# Convergence probes: a val-ON run of EACH arm (seed 0) to a SEPARATE project, so the
# official grid (experiments/tt100k) is never touched. Each probe trains the arm for its
# EQUALIZED step budget (--base-epochs 25 -> run_det computes the per-arm epochs) with
# Ultralytics per-epoch validation on, so we get the val mAP curve and can prove each arm
# plateaus within its budget (the base_epochs justification, per-arm).
#
# Idempotent: skips an arm whose results.csv is already in reports/qa/probe/. The arm's
# synthetic tiles must already exist (the grid generates them) — probes do NOT generate.
#
# Usage:  bash scripts/detection/run_convergence_probes.sh [DEVICE]
set -euo pipefail

ARMS=(zero_aug da_only real_duplicate bg_photometric bg_photometric_mask photometric_full \
      copy_paste copy_paste_mask diffusion_bg)
DEVICE="${1:-0}"
PROBE_PROJ="experiments/tt100k_probe"
OUT="reports/qa/probe"
mkdir -p "$OUT"

for arm in "${ARMS[@]}"; do
  csv="${OUT}/${arm}_seed0_results.csv"
  if [[ -f "$csv" ]]; then
    echo "skip (have): $arm"
    continue
  fi
  echo "=== PROBE (val-on): $arm ==="
  # notifica o Telegram (start/done por braço) — probe é longo, é bom saber quando acaba.
  python run_det.py --arm "$arm" --seed 0 --val --base-epochs 25 \
    --K 0.5 --device "$DEVICE" --project "$PROBE_PROJ"
  cp "${PROBE_PROJ}/${arm}_bm050_seed0/results.csv" "$csv"
  echo "-> $csv"
done
echo "all convergence probes done -> ${OUT}/  (rsync these back to analyze)"
