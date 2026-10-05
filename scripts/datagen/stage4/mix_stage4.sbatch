#!/bin/bash -l
#SBATCH --job-name=build_stage4_mix
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#
# Build the stage-4 training mixture (80% stage-4 forward-dynamic + 8% stage-3 +
# 6% stage-2 + 6% stage-1) into data/stage4/train_with_mix.arrow. See
# datagen/mix_arrows.py: stage-4 is the anchor (largest weight) and is used whole;
# stage-3/2/1 are uniformly subsampled to hit the 8/6/6 shares against it.
#   sbatch scripts/datagen/stage4/mix_stage4.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"

"$PY" -m datagen.mix_arrows \
  --inputs "$REPO/data/stage4/train_all.arrow" \
           "$REPO/data/stage3/train_all.arrow" \
           "$REPO/data/stage2/train_all.arrow" \
           "$REPO/data/stage1/train_all.arrow" \
  --dist   80 8 6 6 \
  --out    "$REPO/data/stage4/train_with_mix.arrow" \
  --seed   0
