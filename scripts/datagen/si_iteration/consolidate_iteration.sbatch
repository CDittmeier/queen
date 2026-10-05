#!/usr/bin/env bash
#SBATCH --job-name=consolidate_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

# One Qwen3.8-27B consolidation shard per array index; material hints off, boundary instructions unchanged.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
exec "$PYTHON" -u -m datagen.self_distill_stages consolidate \
  --config "$ROUND_CONFIG/consolidate.yaml" --start-index "$INDEX" --end-index "$((INDEX + 1))"
