#!/usr/bin/env bash
#SBATCH --job-name=score_iteration_play
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu

# Stockfish audit and subsampling; the existing sampler deletes raw games after publication.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

exec "$PYTHON" -u -m datagen.sample_positions \
  --score-play "$(yaml_value "$ROUND_CONFIG/play.yaml" output)" --config "$ROUND_CONFIG/play.yaml"
