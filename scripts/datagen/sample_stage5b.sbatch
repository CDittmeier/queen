#!/bin/bash -l
#SBATCH --job-name=sample_stage5b
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=pli-c
#SBATCH --time=16:00:00
#SBATCH --array=0-23
#
# Stage-5.b simulation-position sampling: each array task takes one train
# shard of the stage-5 PGN allocation, samples one position per game
# (generic, >= START_PLY), plays 100 engine-pool games per position across
# the task's CPUs (lc0 / maia3 on the GPU, Stockfish on CPU), and stores the
# games in the output records (configs/sample_positions/simulate.yaml).
# Output is appended + flushed per position, so a requeued task resumes.
#
#   sbatch scripts/datagen/sample_stage5b.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/joblog" "$REPO/data/lichess/stage5b"

i="$SLURM_ARRAY_TASK_ID"
pgn=$(ls "$REPO"/data/lichess_dbs/allocated/stage5/train_shardrated_*.pgn.zst | sort | sed -n "$((i + 1))p")
out=$(printf '%s/data/lichess/stage5b/train-%03d.jsonl' "$REPO" "$i")

"$PY" -m datagen.sample_positions "$pgn" \
    --config "$REPO/configs/sample_positions/simulate.yaml" \
    --output "$out" --seed $((100 + i))
