#!/bin/bash -l
#SBATCH --job-name=sample_forward
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#
# Stage-3 forward position sampling — same scale/counts as stage-1 step 1
# (configs/sample_positions/random.yaml, per_game=5, the stage-1 PGN allocation,
# 16-way parallel) but with lookahead [1,8] (configs/sample_positions/forward.yaml).
# Output is the forward 5-field records (Q + history, k saved moves, final FEN):
#   train_*    -> data/lichess/forward/train-NNN.jsonl
#   val_test_* -> data/lichess/forward/val_test-000.jsonl
# QA-dataset build is deliberately NOT run here — that needs the per-task
# _render_forward implementations (stage-3 task work, separate step).
#
# Usage:  sbatch scripts/datagen/stage3/sample_stage3.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export PYTHONPATH="$REPO"
PY="${PYTHON:-$REPO/.venv/bin/python}"

ALLOC="$REPO/data/lichess_dbs/allocated/stage1"
OUT="$REPO/data/lichess/forward"
SAMPLE_CFG="$REPO/configs/sample_positions/forward.yaml"
mkdir -p "$OUT" "$REPO/joblog"


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
for pgn in $(ls "$ALLOC"/val_test_shardrated_*.pgn.zst | sort); do
  printf '%s\t%s\n' "$pgn" "$OUT/val_test-000.jsonl" >> "$jobs_file"
done

echo "[forward] sampling $(wc -l < "$jobs_file") shards (16-way, cfg=$SAMPLE_CFG)..."
xargs -P 16 -L 1 -a "$jobs_file" bash -c 'sample_one "$0" "$1"'
rm -f "$jobs_file"
echo "[forward] train positions: $(cat "$OUT"/train-*.jsonl | wc -l)"
echo "[forward] val_test positions: $(wc -l < "$OUT"/val_test-000.jsonl)"
echo "[forward] done -> $OUT"
