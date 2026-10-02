#!/bin/bash -l
#SBATCH --job-name=sample_forward_special
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=06:00:00
#
# Stage-4 forward feature-gated position sampling. The five stage-2 special types
# (check / check_near_mate / checkmate / have_check / stalemate) but each gate is
# applied to the END of a 1-8 ply forward variation (lookahead_ply: [1,8]), so
# every record is a 5-field forward triple (presented Q = P - k + history, the k
# saved moves Q->P, final FEN P) gating the final board P. Same stage-1 PGN
# allocation and game-level split as the other samples, 16-way parallel, into
#   data/lichess/forward_<type>/train_NNNN.jsonl       (train shards)
#   data/lichess/forward_<type>/val-test_NNNN.jsonl    (val_test shard)
# train_/val-test_ naming matches the {split} placeholder used by the QA builder.
# QA-dataset build is NOT run here (separate stage-4 build step).
#
# Usage:  sbatch scripts/datagen/sample_forward_special.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
ALLOC="$REPO/data/lichess_dbs/allocated/stage1"
TYPES="check check_near_mate checkmate have_check stalemate"
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
  cfg="$REPO/configs/sample_positions/forward_$type.yaml"
  out="$REPO/data/lichess/forward_$type"
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

echo "[sample] $(wc -l < "$jobs_file") jobs (5 forward types x shards), 16-way..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1" "$2"'
rm -f "$jobs_file"

echo "[sample] done. Per-type totals:"
for type in $TYPES; do
  d="$REPO/data/lichess/forward_$type"
  echo "  forward_$type: $(cat "$d"/*.jsonl 2>/dev/null | wc -l) positions across $(ls "$d"/*.jsonl 2>/dev/null | wc -l) shards"
done
