#!/bin/bash -l
#SBATCH --job-name=hce040_regen_pack
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#
# Pack the regenerated 0/0/40 game shards into the training arrows (test 500 / val 200 / train rest).
#   sbatch scripts/datagen/hce/pack_hce_games.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

"${PYTHON:-$REPO/.venv/bin/python}" -m datagen.tree.parallelize --mode pack \
  --out "data/stage5/0-0-40/shards" --arrow-out "data/stage5/0-0-40" --test-n 500 --val-n 200
echo "[hce040_regen] packed -> data/stage5/0-0-40/{train_all,val_dataset,test_dataset}.arrow"
