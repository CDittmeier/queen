#!/bin/bash -l
#SBATCH --job-name=datagen_stage4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=10:00:00
#
# Stage-4 dynamic forward-task datagen (build only; positions come from
# sample_forward.sbatch + sample_forward_special.sbatch):
#   1. build the QA datasets via configs/sample_instances/stage4.yaml -> data/stage4/
#      (6 per-task train arrows + val + test; every record is a forward
#       "After the moves <seq>, <dynamic question>" answered on the final board)
#   2. concatenate the per-task train arrows into data/stage4/train_all.arrow
#      (natural proportions, like stage 2; is_checkmate is intentionally smaller)
#
# Usage:  sbatch scripts/datagen/stage4/build_stage4.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage4.yaml}"

# Existing position pools are reused; no data is deleted by this launcher.
OUT="$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one (natural proportions) ----
"$PY" -m datagen.build_train_all \
  --in-dir "$REPO/$OUT" \
  --note   "concatenated stage-4 forward dynamic-task arrows (natural proportions)"
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
