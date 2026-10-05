#!/usr/bin/env bash
#SBATCH --job-name=pack_iteration
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"
exec "$PYTHON" -u -m utils.pack_self_distill --config "$ROUND_CONFIG/packing.yaml"
