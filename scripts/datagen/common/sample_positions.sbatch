#!/usr/bin/env bash
#SBATCH --job-name=sample_positions
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu

# Sample human games, puzzles, or model play using the sampler's existing CLI.
# sbatch scripts/datagen/common/sample_positions.sbatch 'data/games/*.pgn.zst' \
#   --config configs/sample_positions/random.yaml --output data/positions.jsonl
# For play, request --partition=gpu --gres=gpu:1 at submission
# using the same .venv as training. Use disjoint inputs/seeds per job.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.sample_positions "$@"
