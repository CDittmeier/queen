#!/usr/bin/env bash
#SBATCH --job-name=pack_instances
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# Concatenate a curriculum stage's per-task Arrow datasets into train_all.arrow.
# sbatch scripts/datagen/common/pack_instances.sbatch --in-dir data/stage1
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.build_train_all "$@"
