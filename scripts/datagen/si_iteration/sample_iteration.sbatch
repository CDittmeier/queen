#!/usr/bin/env bash
#SBATCH --job-name=sample_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --array=0-1

# Task 0: five positions/game from the allocated human source. Task 1: 35k puzzles.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    CONFIG="$ROUND_CONFIG/human.yaml"
    SOURCE=("$(round_value human_source)")
    ;;
  1)
    CONFIG="$ROUND_CONFIG/puzzles.yaml"
    SOURCE=(--puzzles "$(round_value puzzle_source)")
    ;;
  *) echo "sample_iteration expects task 0 (human) or 1 (puzzles)" >&2; exit 1 ;;
esac
mkdir -p "$(dirname "$(yaml_value "$CONFIG" output)")"
exec "$PYTHON" -u -m datagen.sample_positions "${SOURCE[@]}" --config "$CONFIG"
