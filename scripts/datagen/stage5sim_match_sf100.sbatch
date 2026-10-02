#!/usr/bin/env bash
#SBATCH --job-name=s5sim_vs_sf100
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%j-%x.out
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MT_FORCE_NODES=${MT_FORCE_NODES:-10000}

OUT=${OUT:-$REPO/lab/stage5sim_ladder/startpos_vs_sf100}
SF_NODES=${SF_NODES:-100}
PY=${VLLM_PYTHON:-$REPO/vllm_venv/bin/python}
mkdir -p "$REPO/joblog" "$OUT"

HISTORY_ARGS=()
if [[ "${INCLUDE_HISTORY:-0}" == "1" ]]; then
    HISTORY_ARGS+=(--history)
fi

exec "$PY" -u helpers/run_stage5sim_match.py \
    --out "$OUT" --max-plies 400 \
    --max-batch 128 --max-num-seqs 128 --sf-judges 4 \
    --opponent-nodes "$SF_NODES" \
    "${HISTORY_ARGS[@]}"
