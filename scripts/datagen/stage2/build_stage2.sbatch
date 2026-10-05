#!/bin/bash -l
#SBATCH --job-name=datagen_stage2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=10:00:00
#
# Stage-2 dynamic-task datagen:
#   1. build the QA datasets via configs/sample_instances/stage2.yaml -> data/stage2/
#      (6 per-task train arrows + val + test)
#   2. concatenate the per-task train arrows into data/stage2/train_all.arrow
#      (natural proportions; is_checkmate is intentionally smaller, capped by the
#       stalemate supply)
#
# Positions are the feature-gated samples under data/lichess/ (generic / check /
# have_check / checkmate / check_near_mate / stalemate) — no sampling step here.
#
# Usage:  sbatch scripts/datagen/stage2/build_stage2.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage2.yaml}"

# Existing position pools are reused; no data is deleted by this launcher.
OUT="$($PY -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one --------------------------
"$PY" -m datagen.build_train_all \
  --in-dir "$REPO/$OUT" \
  --note   "concatenated stage-2 per-task arrows (natural proportions)"
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
