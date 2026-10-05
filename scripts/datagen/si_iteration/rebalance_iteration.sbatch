#!/usr/bin/env bash
#SBATCH --job-name=rebalance_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# Deduplicate and repartition all accepted records after every mining task has completed.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

exec "$PYTHON" -u -m datagen.self_distill_stages rebalance --config "$ROUND_CONFIG/rebalance.yaml"
