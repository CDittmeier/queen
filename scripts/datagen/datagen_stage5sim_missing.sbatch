#!/usr/bin/env bash
#SBATCH --job-name=s5sim_missing
#SBATCH --account=danqic
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%j-%x.out
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"

REPAIR_OUT=${REPAIR_OUT:?set REPAIR_OUT to the completed repair output root}
MISSING_OUT=${MISSING_OUT:?set MISSING_OUT to the final manifest directory}

python -u helpers/build_stage5sim_missing.py \
    --game-primary data/stage5/sim/game_shards \
    --puzzle-primary data/stage5/sim/puzzle_shards \
    --repair-root "$REPAIR_OUT" \
    --out "$MISSING_OUT"
