#!/usr/bin/env bash
#SBATCH --job-name=pack_hce
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# Pack HCE shards into position-disjoint train/validation/test Arrow datasets.
# sbatch scripts/datagen/hce/pack_hce.sbatch --out data/stage5/0-0-40/shards \
#   --arrow-out data/stage5/0-0-40 --test-n 500 --val-n 200
# Game and puzzle datasets use the same launcher with different paths.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.tree.parallelize --mode pack "$@"
