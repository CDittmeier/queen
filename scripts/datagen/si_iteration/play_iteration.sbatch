#!/usr/bin/env bash
#SBATCH --job-name=play_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Batched model-vs-Stockfish play, strength log-uniform over 100–100k nodes.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

mkdir -p "$(dirname "$(yaml_value "$ROUND_CONFIG/play.yaml" output)")"
exec "$PYTHON" -u -m datagen.sample_positions \
  --play-games "$(round_value play_games)" --config "$ROUND_CONFIG/play.yaml"
