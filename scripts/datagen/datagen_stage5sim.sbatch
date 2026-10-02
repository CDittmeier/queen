#!/usr/bin/env bash
#SBATCH --job-name=s5sim_960k
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --array=0-39
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A_%a-%x.out
#
# Stage-5.sim production array: 960k positions in a 95/5 game/puzzle mix.
#
#   tasks  0..37: 912k game positions / 38 logical shards = 24k each
#   tasks 38..39:  48k puzzle positions /  2 logical shards = 24k each
#
# Logical sharding is over accepted global position indices, not source files.
# The 38 game tasks are therefore disjoint even though the source pool has 25
# physical train files. Each record is appended and flushed immediately. Since
# --total fixes each logical shard's population, resubmitting a failed array
# task skips completed indices and processes only its unfinished remainder.
#
# The puzzle pool was built from Lichess puzzles rated 600--3000 by
# helpers/build_puzzle_shards.py and converted to the same triple format by
# helpers/build_puzzle_pool.py.
#
# Submit only after the production greenlight:
#   sbatch scripts/datagen/datagen_stage5sim.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stockfish at 10k nodes (validated: 10k and 100k pick the same move wherever
# either calls one clearly best); override via MT_* only for an intentional run.
export MT_FORCE_NODES=${MT_FORCE_NODES:-10000}
export MT_TACTIC_NODES=${MT_TACTIC_NODES:-10000}
export MT_ORACLE_NODES=${MT_ORACLE_NODES:-10000}

PER_TASK=${PER_TASK:-24000}
GAME_SHARDS=38
PUZZLE_SHARDS=2
ARRAY_ID=${SLURM_ARRAY_TASK_ID:?must run as a Slurm array task}

if (( ARRAY_ID < GAME_SHARDS )); then
    SOURCE=game
    SHARD=$ARRAY_ID
    NUM_SHARDS=$GAME_SHARDS
    TOTAL=$((PER_TASK * GAME_SHARDS))
    POOL="$REPO/data/lichess/stage5/train-*.jsonl"
    OUT="$REPO/data/stage5/sim/game_shards"
else
    SOURCE=puzzle
    SHARD=$((ARRAY_ID - GAME_SHARDS))
    NUM_SHARDS=$PUZZLE_SHARDS
    TOTAL=$((PER_TASK * PUZZLE_SHARDS))
    POOL="$REPO/data/lichess/puzzles/pool/pool-*.jsonl"
    OUT="$REPO/data/stage5/sim/puzzle_shards"
fi

mkdir -p "$REPO/joblog" "$OUT"
echo "[stage5sim] source=$SOURCE array_id=$ARRAY_ID logical_shard=$SHARD/$NUM_SHARDS total=$TOTAL per_task=$PER_TASK"

# vLLM-capable Python (the repo .venv has no vLLM).
PY=${VLLM_PYTHON:-$REPO/vllm_venv/bin/python}

exec "$PY" -u -m datagen.sim.parallelize run \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" \
    --pool "$POOL" --out "$OUT" --total "$TOTAL" \
    --concurrency "${K:-128}" --max-batch "${B:-128}" \
    --max-num-seqs "${S:-128}" --sf-judges "${SFJ:-4}"
