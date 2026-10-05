#!/bin/bash -l
#SBATCH --job-name=hce040_puzzles
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#
# Lichess-puzzle HCE-rationale dataset: 300k puzzle roots (pool built by
# datagen.tree.prepare_puzzle_pool from puzzle shards 001-003) through the same 0/0/40
# tree pipeline as the game data. NO de-skew (puzzles are tactical by design; keep all).
# 12-way array, 25k kept per task. Resumable like the game run.
#   sbatch scripts/datagen/hce/build_hce_puzzles.sbatch
# Then once all shards are complete:
#   sbatch scripts/datagen/hce/pack_hce_puzzles.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$REPO/data/engines/cudalibs:${LD_LIBRARY_PATH:-}"   # lc0 needs libcublas at load
mkdir -p "$REPO/joblog"

"${PYTHON:-$REPO/.venv/bin/python}" -m datagen.tree.parallelize --mode run \
  --pool "data/lichess/puzzles/pool/pool-*.jsonl" \
  --out  "data/stage5/puzzles-0-0-40/shards" \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 12 --target 25000 --flush 2000 \
  --cover 0.0 --cover-deep 0.0 --max-nodes 40 --no-deskew
