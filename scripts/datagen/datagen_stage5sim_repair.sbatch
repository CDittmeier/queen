#!/usr/bin/env bash
#SBATCH --job-name=s5sim_repair
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
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
export MT_TACTIC_NODES=${MT_TACTIC_NODES:-10000}
export MT_ORACLE_NODES=${MT_ORACLE_NODES:-10000}

SNAPSHOT=${SNAPSHOT:?set SNAPSHOT to the repair snapshot directory}
REPAIR_OUT=${REPAIR_OUT:-$SNAPSHOT/outputs}
SAMPLE_OUT=${SAMPLE_OUT:-$REPO/lab/two_networks/reports/stage5sim_repair_sample10.md}
PY=${VLLM_PYTHON:-$REPO/vllm_venv/bin/python}

run_source() {
    local source=$1 pool=$2 total=$3
    local manifest="$SNAPSHOT/${source}_indices.txt"
    local out="$REPAIR_OUT/$source"
    mkdir -p "$out"
    "$PY" -u -m datagen.sim.parallelize run \
        --shard 0 --num-shards 1 --pool "$pool" --out "$out" \
        --total "$total" --indices "$manifest" \
        --concurrency 128 --max-batch 128 --max-num-seqs 128 --sf-judges 4
}

run_source game "$REPO/data/lichess/stage5/train-*.jsonl" 912000
run_source puzzle "$REPO/data/lichess/puzzles/pool/pool-*.jsonl" 48000

"$PY" -u helpers/render_stage5sim_samples.py \
    --repairs "$REPAIR_OUT" --out "$SAMPLE_OUT" --count 10
