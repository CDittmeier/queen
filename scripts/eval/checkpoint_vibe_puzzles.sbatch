#!/usr/bin/env bash
#SBATCH --job-name=checkpoint_puzzles
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/lab/checkpoint_vibe/puzzle_eval_100_s20260808/slurm-%j.out
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
OUT=${OUT:-$REPO/lab/checkpoint_vibe/puzzle_eval_100_s20260808}
PY=${VLLM_PYTHON:-$REPO/vllm_venv/bin/python}

cd "$REPO"
mkdir -p "$OUT"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec "$PY" -u helpers/eval_checkpoint_puzzles.py \
    --run --output "$OUT" \
    --sf-workers 1 --gpu-memory-utilization 0.80
