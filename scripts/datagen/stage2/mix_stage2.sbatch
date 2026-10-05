#!/bin/bash -l
#SBATCH --job-name=build_stage2_mix
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#
# Build the stage-2 training mixture (90% stage-2 train_all + 10% stage-1) into
# data/stage2/train_with_stage1_mix.arrow. See datagen/mix_arrows.py.
#   sbatch scripts/datagen/stage2/mix_stage2.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"

"$PY" -m datagen.mix_arrows \
  --inputs "$REPO/data/stage2/train_all.arrow" \
           "$REPO/data/stage1/train_all.arrow" \
  --dist   9 1 \
  --out    "$REPO/data/stage2/train_with_stage1_mix.arrow" \
  --seed   0
