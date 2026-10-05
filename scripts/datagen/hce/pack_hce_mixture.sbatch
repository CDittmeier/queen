#!/bin/bash -l
#SBATCH --job-name=hce040_dual_pack
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#
# Pack the hce040 game + puzzle shards into both stage-5 training arrows
# (game-only and game+100k-puzzles), the common test/val split, and the
# 10k held-out puzzle eval set.
#   sbatch scripts/datagen/hce/pack_hce_mixture.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

"${PYTHON:-$REPO/.venv/bin/python}" -m datagen.tree.pack_mixture \
  --game-shards "data/stage5/0-0-40/shards/shard_*.jsonl" \
  --puzzle-shards "data/stage5/puzzles-0-0-40/shards/shard_*.jsonl" \
  --out-game data/stage5/0-0-40 --out-mix data/stage5/0-0-40-puzzles100k \
  --heldout-out data/stage5/puzzles-0-0-40/heldout_eval \
  --test-n 500 --val-n 200 --heldout-n 10000 --mix-n 100000 --seed 0
echo "[hce040_dual_pack] done"
