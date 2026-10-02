#!/bin/bash -l
#SBATCH --job-name=build_stage2_mix
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=01:00:00
#
# Build the stage-2 training mixture (90% stage-2 train_all + 10% stage-1) into
# data/stage2/train_with_stage1_mix.arrow. See datagen/mix_arrows.py.
#   sbatch scripts/datagen/build_stage2_mix.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"

"$PY" -m datagen.mix_arrows \
  --inputs "$REPO/data/stage2/train_all.arrow" \
           "$REPO/data/stage1/train_all.arrow" \
  --dist   9 1 \
  --out    "$REPO/data/stage2/train_with_stage1_mix.arrow" \
  --seed   0
