#!/usr/bin/env bash
#SBATCH --job-name=sample_hce_puzzles
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
# Convert original puzzle shards 001, 002, 003 (300k rows) to the HCE tree pool.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
exec "$PYTHON" -m datagen.tree.prepare_puzzle_pool
