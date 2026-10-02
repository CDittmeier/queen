#!/bin/bash -l
#SBATCH --job-name=sample_special
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00
#
# Sample the four feature-gated position types (check / checkmate / stalemate /
# check_near_mate) from the stage-1 allocated PGNs, 1 jsonl per PGN shard, into
#   data/lichess/<type>/train_NNNN.jsonl       (25 train shards)
#   data/lichess/<type>/val-test_NNNN.jsonl    (1 val_test shard)
# Same game-level split as the generic sample: train and val_test come from
# disjoint allocated PGN sets. Each (type, shard) is one sample_positions run,
# dispatched 16-way.
#
# Usage:  sbatch scripts/datagen/sample_special.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
ALLOC="$REPO/data/lichess_dbs/allocated/stage1"
TYPES="check checkmate stalemate check_near_mate"
mkdir -p "$REPO/joblog"

sample_one() {
  local pgn="$1" out="$2" cfg="$3"
  if [ -s "$out" ]; then echo "[skip] $out ($(wc -l < "$out") lines)"; return; fi
  "$PY" -m datagen.sample_positions "$pgn" --config "$cfg" --output "$out" --seed 1
}
export -f sample_one
export PY

jobs_file="$(mktemp)"
for type in $TYPES; do
  cfg="$REPO/configs/sample_positions/$type.yaml"
  out="$REPO/data/lichess/$type"
  mkdir -p "$out"
  i=0
  for pgn in $(ls "$ALLOC"/train_shardrated_*.pgn.zst | sort); do
    printf '%s\t%s\t%s\n' "$pgn" "$(printf '%s/train_%04d.jsonl' "$out" "$i")" "$cfg" >> "$jobs_file"
    i=$((i + 1))
  done
  j=0
  for pgn in $(ls "$ALLOC"/val_test_shardrated_*.pgn.zst | sort); do
    printf '%s\t%s\t%s\n' "$pgn" "$(printf '%s/val-test_%04d.jsonl' "$out" "$j")" "$cfg" >> "$jobs_file"
    j=$((j + 1))
  done
done

echo "[sample] $(wc -l < "$jobs_file") jobs (4 types x 26 shards), 16-way..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1" "$2"'
rm -f "$jobs_file"

echo "[sample] done. Per-type totals:"
for type in $TYPES; do
  d="$REPO/data/lichess/$type"
  echo "  $type: $(cat "$d"/*.jsonl 2>/dev/null | wc -l) positions across $(ls "$d"/*.jsonl 2>/dev/null | wc -l) shards"
done
