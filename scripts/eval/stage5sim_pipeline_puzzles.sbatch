#!/usr/bin/env bash
#SBATCH --job-name=simpipe_puzzles
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/lab/checkpoint_vibe/puzzle_eval_100_s20260808/pipeline-%j.out
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
export MT_FORCE_NODES=10000 MT_TACTIC_NODES=10000 MT_ORACLE_NODES=10000

"$PY" -u helpers/eval_sim_pipeline_puzzles.py \
    --output "$OUT" --concurrency 100 --max-batch 128 \
    --max-num-seqs 128 --sf-judges 8

if [[ "${SKIP_SCORE:-0}" != 1 ]]; then
    "$PY" -u helpers/eval_checkpoint_puzzles.py \
        --score-only --output "$OUT" --sf-workers 1
fi
