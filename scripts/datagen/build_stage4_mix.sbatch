#!/bin/bash -l
#SBATCH --job-name=build_stage4_mix
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00
#
# Build the stage-4 training mixture (80% stage-4 forward-dynamic + 8% stage-3 +
# 6% stage-2 + 6% stage-1) into data/stage4/train_with_mix.arrow. See
# datagen/mix_arrows.py: stage-4 is the anchor (largest weight) and is used whole;
# stage-3/2/1 are uniformly subsampled to hit the 8/6/6 shares against it.
#   sbatch scripts/datagen/build_stage4_mix.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"

"$PY" -m datagen.mix_arrows \
  --inputs "$REPO/data/stage4/train_all.arrow" \
           "$REPO/data/stage3/train_all.arrow" \
           "$REPO/data/stage2/train_all.arrow" \
           "$REPO/data/stage1/train_all.arrow" \
  --dist   80 8 6 6 \
  --out    "$REPO/data/stage4/train_with_mix.arrow" \
  --seed   0
