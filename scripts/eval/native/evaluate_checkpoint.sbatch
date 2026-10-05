#!/usr/bin/env bash
#SBATCH --job-name=evaluate_checkpoint
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Evaluate a checkpoint on an Arrow test set; --regret adds Stockfish move scoring.
# sbatch scripts/eval/native/evaluate_checkpoint.sbatch --config CONFIG.yaml --ckpt CHECKPOINT \
#   --test-dataset TEST.arrow --out results.json
# For narrative evaluations, explicitly choose --temperature and --max-new-tokens.
# This invokes eval.py, not the separate historical 100/1000-position benchmark helpers.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u eval.py "$@"
