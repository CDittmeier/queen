#!/bin/bash -l
#SBATCH --job-name=hce040_regen
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --array=0-71
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=16:00:00
#
# Stage-5 HCE-rationale REGENERATION at the production 0/0/40 tree config, now with tactic-motif
# labeling (MultiPV-2 judge + tag_tactics; measured cost 1.00x vs the old pipeline). Same pool,
# same 72-way split, same de-skew and target as the original run -> same data composition.
# Resumable: re-submit a task to continue where it stopped.
#   sbatch scripts/datagen/datagen_hce040_regen.sbatch                # launch all 72
#   sbatch --array=3,17 scripts/datagen/datagen_hce040_regen.sbatch   # resume specific shards
# Then once all shards are complete:
#   sbatch scripts/datagen/datagen_hce040_regen_pack.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$REPO/data/engines/cudalibs:${LD_LIBRARY_PATH:-}"   # lc0 needs libcublas at load
mkdir -p "$REPO/joblog"

"$REPO/.venv/bin/python" -m datagen.tree.parallelize --mode run \
  --pool "data/lichess/stage5/train-*.jsonl" \
  --out  "data/stage5/0-0-40/shards" \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 72 --target 17362 --flush 2000 \
  --cover 0.0 --cover-deep 0.0 --max-nodes 40
