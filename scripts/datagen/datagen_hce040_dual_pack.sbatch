#!/bin/bash -l
#SBATCH --job-name=hce040_dual_pack
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=04:00:00
#
# Pack the hce040 game + puzzle shards into both stage-5 training arrows
# (game-only and game+100k-puzzles), the common test/val split, and the
# 10k held-out puzzle eval set.
#   sbatch scripts/datagen/datagen_hce040_dual_pack.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

"$REPO/.venv/bin/python" helpers/build_stage5_dual_arrows.py
echo "[hce040_dual_pack] done"
