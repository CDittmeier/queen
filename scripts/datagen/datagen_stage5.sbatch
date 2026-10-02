#!/bin/bash -l
#SBATCH --job-name=stage5_shard
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
# Stage-5 machine-readable HCE-rationale dataset — the heavy compute, as a 72-way slurm array.
# Each task K processes pool positions with global_idx % 72 == K, applies the de-skew filter
# (|NNUE eval| > 2 pawns -> drop 25%), get_trees the kept ones, and STREAMS them to
# data/stage5/shards/shard_KKKK.jsonl with flush+fsync every 2000. RESUMABLE: re-submitting a task
# skips already-written idxs and continues until its target (~17,362 kept) is met, so a task that
# hits the 16h wall just needs re-submitting. Measured ~3.1s/position on 4 cpus -> ~17.6k/16h.
#
#   sbatch scripts/datagen/datagen_stage5.sbatch                 # launch all 72
#   sbatch --array=3,17 scripts/datagen/datagen_stage5.sbatch    # re-run specific shards (resume)
# Then, once all shards are complete:
#   sbatch scripts/datagen/datagen_stage5_pack.sbatch            # merge -> data/stage5/*.arrow
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# lc0 needs libcublas*.so.13 at load time even on the eigen/CPU backend.
export LD_LIBRARY_PATH="$REPO/data/engines/cudalibs:${LD_LIBRARY_PATH:-}"
mkdir -p "$REPO/joblog"

"$REPO/.venv/bin/python" -m datagen.tree.parallelize --mode run \
  --pool "data/lichess/stage5/train-*.jsonl" \
  --out  "data/stage5/shards" \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 72 --target 17362 --flush 2000
