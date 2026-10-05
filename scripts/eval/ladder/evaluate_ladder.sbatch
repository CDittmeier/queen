#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=evaluate_ladder
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# All 32 games are batched; resubmit to resume unfinished games, not restart them.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
exec "$PYTHON" -u -m eval.model_ladder --config "${1:-configs/eval/model_ladder.yaml}" \
  --output "${2:-runs/self_distill/iteration/eval/ladder}" --max-seconds 3300
