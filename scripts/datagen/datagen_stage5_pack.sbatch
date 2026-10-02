#!/bin/bash -l
#SBATCH --job-name=stage5_pack
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00
#
# Pack the stage-5 shard jsonls (data/stage5/shards/shard_*.jsonl) into the final HF arrow dataset
# (same {fen,history,prompt,response,extra} schema as build_qa_dataset): split by global idx into
# test (500) / val (200) / train (rest), all position-disjoint. TOLERATES an in-progress array — packs
# whatever is on disk and warns if incomplete, so it can be run early for a partial dataset and re-run
# once the array completes (overwrites). Uses the build_qa_dataset --stage5-pack entry point.
#   sbatch scripts/datagen/datagen_stage5_pack.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

"$REPO/.venv/bin/python" -m datagen.build_qa_dataset \
  --config configs/sample_instances/stage5.yaml --stage5-pack
echo "[stage5] packed -> data/stage5/{test_dataset,val_dataset}.arrow + training_datasets/hce_rationale.arrow"
