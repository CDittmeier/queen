#!/bin/bash -l
#SBATCH --job-name=datagen_stage2
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=10:00:00
# Wait for the have_check position sampling (job 9923115) to finish OK — the
# checking_moves mix sources data/lichess/have_check/. Update the id if that job
# is resubmitted; drop this line once have_check/ is already on disk.
#SBATCH --dependency=afterok:9923115
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
# Usage:  sbatch scripts/datagen/datagen_stage2.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage2.yaml}"

# Fresh start for the build output (positions in data/lichess/ are kept).
OUT="$($PY -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"
rm -rf "$REPO/$OUT"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one --------------------------
"$PY" -m datagen.build_train_all \
  --in-dir "$REPO/$OUT" \
  --note   "concatenated stage-2 per-task arrows (natural proportions)"
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
