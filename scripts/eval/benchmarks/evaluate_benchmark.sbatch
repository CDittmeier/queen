#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=evaluate_benchmark
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
exec "$PYTHON" -u -m eval.benchmark --config "${1:-configs/eval/benchmark.yaml}" \
  --dataset-index "${SLURM_ARRAY_TASK_ID:-0}"
