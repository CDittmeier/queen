#!/bin/bash -l
#SBATCH --job-name=sample_stage5
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=03:00:00
#
# Sample stage-5 positions from the stage-5 allocation (renamed from the never-used stage-2
# block): the 25 train_ shards of rated_2018-10 [00050-00074] — disjoint from stage-1/2, whose
# jsonls come from [00000-00024]. Generic present-position sampling (per_game=5), same 3-field
# format as stage 1: [start_fen, moves, end_fen]. Feeds configs/sample_instances/stage5.yaml,
# whose val/test splits are carved position-disjoint from these train shards by the builder.
#
#   sbatch scripts/datagen/sample_stage5.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/joblog"

ALLOC="$REPO/data/lichess_dbs/allocated/stage5"
OUT="$REPO/data/lichess/stage5"
SAMPLE_CFG="$REPO/configs/sample_positions/random.yaml"   # per_game lives here
mkdir -p "$OUT"

sample_one() {
  local pgn="$1" out="$2"
  if [ -s "$out" ]; then echo "[skip] $out ($(wc -l < "$out") lines)"; return; fi
  "$PY" -m datagen.sample_positions "$pgn" --config "$SAMPLE_CFG" --output "$out" --seed 1
}
export -f sample_one
export PY SAMPLE_CFG

jobs_file="$(mktemp)"
i=0
for pgn in $(ls "$ALLOC"/train_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\n' "$pgn" "$(printf '%s/train-%03d.jsonl' "$OUT" "$i")" >> "$jobs_file"
  i=$((i + 1))
done

echo "[sample_stage5] sampling $(wc -l < "$jobs_file") train shards (16-way, cfg=$SAMPLE_CFG)..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1"'
rm -f "$jobs_file"
echo "[sample_stage5] train positions: $(cat "$OUT"/train-*.jsonl | wc -l)"
echo "[sample_stage5] done -> $OUT"
