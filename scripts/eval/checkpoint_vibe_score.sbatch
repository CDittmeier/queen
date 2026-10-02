#!/usr/bin/env bash
#SBATCH --job-name=checkpoint_score
#SBATCH --account=danqic
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/lab/checkpoint_vibe/puzzle_eval_100_s20260808/score-%j.out
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
OUT=${OUT:-$REPO/lab/checkpoint_vibe/puzzle_eval_100_s20260808}

cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec "$REPO/vllm_venv/bin/python" -u helpers/eval_checkpoint_puzzles.py \
    --score-only --output "$OUT" --sf-workers 8
