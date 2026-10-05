#!/bin/bash -l
#SBATCH --job-name=datagen_stage3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=10:00:00
#
# Stage-3 forward-task datagen (build only; positions come from
# scripts/datagen/stage3/sample_stage3.sbatch -> data/lichess/forward/):
#   1. build the QA datasets via configs/sample_instances/stage3.yaml -> data/stage3/
#      (7 per-task train arrows + val + test; every record is a forward
#       "After the moves <seq>, <question>" example answered on the final board)
#   2. concatenate the 7 per-task train arrows into data/stage3/train_all.arrow
#      (equal sizes => uniform 1/7; DataLoader shuffles)
#
# Usage:  sbatch scripts/datagen/stage3/build_stage3.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage3.yaml}"

# Existing position pools are reused; no data is deleted by this launcher.
OUT="$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one (uniform 1/7) ------------
"$PY" -m datagen.build_train_all --in-dir "$REPO/$OUT" \
  --note "concatenated stage-3 forward per-task arrows"
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
