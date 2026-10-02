#!/bin/bash -l
#SBATCH --job-name=datagen_stage4
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=10:00:00
# Wait for the forward feature-gated sampling (job 10214418 ->
# data/lichess/forward_{check,have_check,checkmate,check_near_mate,stalemate}/) to
# finish OK — the mixed tasks source those pools. The generic forward pool
# (data/lichess/forward/) is already on disk from stage 3. Update the id if that
# job is resubmitted; drop this line once the forward_* pools are on disk.
#SBATCH --dependency=afterok:10214418
#
# Stage-4 dynamic forward-task datagen (build only; positions come from
# sample_forward.sbatch + sample_forward_special.sbatch):
#   1. build the QA datasets via configs/sample_instances/stage4.yaml -> data/stage4/
#      (6 per-task train arrows + val + test; every record is a forward
#       "After the moves <seq>, <dynamic question>" answered on the final board)
#   2. concatenate the per-task train arrows into data/stage4/train_all.arrow
#      (natural proportions, like stage 2; is_checkmate is intentionally smaller)
#
# Usage:  sbatch scripts/datagen/datagen_stage4.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage4.yaml}"

# Fresh start for the build output (positions in data/lichess/ are kept).
OUT="$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"
rm -rf "$REPO/$OUT"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one (natural proportions) ----
"$PY" -m datagen.build_train_all \
  --in-dir "$REPO/$OUT" \
  --note   "concatenated stage-4 forward dynamic-task arrows (natural proportions)"
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
