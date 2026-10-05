#!/usr/bin/env bash
#SBATCH --job-name=mix_datasets
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# Mix Arrow datasets; source paths, proportions, and destination are explicit.
# sbatch scripts/datagen/common/mix_datasets.sbatch --inputs data/stage2/train_all.arrow \
#   data/stage1/train_all.arrow --dist 9 1 --out data/stage2/train_with_stage1_mix.arrow --seed 0
# Historical recipes, inputs newest stage first:
# stage 2: [stage2, stage1]                 -> --dist 9 1
# stage 3: [stage3, stage2, stage1]         -> --dist 86 8 6
# stage 4: [stage4, stage3, stage2, stage1] -> --dist 80 8 6 6
# Stages 3/4 write data/stageN/train_with_mix.arrow.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.mix_arrows "$@"
