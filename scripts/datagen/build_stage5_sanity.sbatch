#!/usr/bin/env bash
#SBATCH --job-name=stage5-sanity
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --array=0-3

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
source "$REPO/lab/vllm_nightly_venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

SMALL=()
if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  SMALL=(--include-small-splits)
fi

cd "$REPO"
python helpers/build_stage5_sanity.py build \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  "${SMALL[@]}"
