#!/usr/bin/env bash
#SBATCH --job-name=mine_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# One mining task per array index; recursive FIFO, batching, and resume are in the existing miner.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
exec "$PYTHON" -u -m datagen.self_distill_stages mine \
  --config "$ROUND_CONFIG/mine.yaml" --start-index "$INDEX" --end-index "$((INDEX + 1))"
