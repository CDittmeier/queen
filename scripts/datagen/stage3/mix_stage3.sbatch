#!/bin/bash -l
#SBATCH --job-name=build_stage3_mix
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#
# Build the stage-3 training mixture (86% stage-3 forward + 8% stage-2 + 6%
# stage-1) into data/stage3/train_with_mix.arrow. See datagen/mix_arrows.py:
# stage-3 is the anchor (largest weight) and is used whole; stage-2/stage-1 are
# uniformly subsampled to hit the 8/6 shares against it.
#   sbatch scripts/datagen/stage3/mix_stage3.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"

"$PY" -m datagen.mix_arrows \
  --inputs "$REPO/data/stage3/train_all.arrow" \
           "$REPO/data/stage2/train_all.arrow" \
           "$REPO/data/stage1/train_all.arrow" \
  --dist   86 8 6 \
  --out    "$REPO/data/stage3/train_with_mix.arrow" \
  --seed   0
