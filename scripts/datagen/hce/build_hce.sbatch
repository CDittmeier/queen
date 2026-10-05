#!/usr/bin/env bash
#SBATCH --job-name=build_hce
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --partition=cpu

# Build resumable HCE JSONL shards; defaults to the selected 0/0/40 tree recipe.
# sbatch --array=0-71 scripts/datagen/hce/build_hce.sbatch \
#   --pool 'data/lichess/stage5/train-*.jsonl' --out data/stage5/0-0-40/shards \
#   --nshards 72 --target 17362 --flush 2000
# Puzzles: use their pool/output, --array=0-11 --nshards 12 --target 25000 --no-deskew.
# Explicit --shard overrides the array index. Other tree recipes use CLI overrides.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.tree.parallelize --mode run \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --cover 0 --cover-deep 0 --max-nodes 40 "$@"
