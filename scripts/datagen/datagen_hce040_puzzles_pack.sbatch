#!/bin/bash -l
#SBATCH --job-name=hce040_puzzles_pack
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00
#
# Pack the 300k puzzle shards into their own arrows (mixed into training later at 0/5/10/20%).
#   sbatch scripts/datagen/datagen_hce040_puzzles_pack.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

"$REPO/.venv/bin/python" -m datagen.tree.parallelize --mode pack \
  --out "data/stage5/puzzles-0-0-40/shards" --arrow-out "data/stage5/puzzles-0-0-40" --test-n 500 --val-n 200
echo "[hce040_puzzles] packed -> data/stage5/puzzles-0-0-40/{train_all,val_dataset,test_dataset}.arrow"
