#!/bin/bash -l
#SBATCH --job-name=sample_have_check
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00
#
# Sample the have_check feature (side to move has >=1 legal checking move) from the
# stage-1 allocated PGNs, 1 jsonl per PGN shard, into
#   data/lichess/have_check/train_NNNN.jsonl      (25 train shards, 000-024)
#   data/lichess/have_check/val-test_0000.jsonl   (1 val_test shard)
# Same game-level split as the other samples (train and val_test from disjoint
# allocated PGN sets). Each shard is one sample_positions run, dispatched 16-way.
#
# Usage:  sbatch scripts/datagen/sample_have_check.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
ALLOC="$REPO/data/lichess_dbs/allocated/stage1"
CFG="$REPO/configs/sample_positions/have_check.yaml"
OUT="$REPO/data/lichess/have_check"
mkdir -p "$REPO/joblog" "$OUT"

sample_one() {
  local pgn="$1" out="$2" cfg="$3"
  if [ -s "$out" ]; then echo "[skip] $out ($(wc -l < "$out") lines)"; return; fi
  "$PY" -m datagen.sample_positions "$pgn" --config "$cfg" --output "$out" --seed 1
}
export -f sample_one
export PY

jobs_file="$(mktemp)"
i=0
for pgn in $(ls "$ALLOC"/train_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\t%s\n' "$pgn" "$(printf '%s/train_%04d.jsonl' "$OUT" "$i")" "$CFG" >> "$jobs_file"
  i=$((i + 1))
done
j=0
for pgn in $(ls "$ALLOC"/val_test_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\t%s\n' "$pgn" "$(printf '%s/val-test_%04d.jsonl' "$OUT" "$j")" "$CFG" >> "$jobs_file"
  j=$((j + 1))
done

echo "[sample] $(wc -l < "$jobs_file") jobs (have_check x $((i + j)) shards), 16-way..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1" "$2"'
rm -f "$jobs_file"

echo "[sample] done. have_check: $(cat "$OUT"/*.jsonl 2>/dev/null | wc -l) positions across $(ls "$OUT"/*.jsonl 2>/dev/null | wc -l) shards"
