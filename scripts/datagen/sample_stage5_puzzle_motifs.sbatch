#!/bin/bash -l
#SBATCH --job-name=sample_s5_puzmot
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY="$REPO/.venv/bin/python"
CFG="$REPO/configs/sample_positions/puzzle_motifs.yaml"
OUT="$REPO/data/lichess/stage5_motifs/puzzles"
PARTIALS="$REPO/lab/stage5_motifs/puzzle_partials"

mkdir -p "$OUT" "$PARTIALS" "$REPO/joblog"
i="$SLURM_ARRAY_TASK_ID"
limit=25000
global_start=$((i * limit))
source_index=$((global_start / 100000 + 1))
source_offset=$((global_start % 100000))
source=$(printf '%s/data/lichess/puzzles/shard-%03d.jsonl' "$REPO" "$source_index")
out=$(printf '%s/puzzle-%03d.jsonl' "$OUT" "$i")
partial=$(printf '%s/puzzle-%03d.job-%s.jsonl' "$PARTIALS" "$i" "$SLURM_JOB_ID")

if [ -s "$out" ]; then
  echo "[skip] completed output already exists: $out"
  exit 0
fi
if [ -e "$partial" ]; then
  echo "refusing to overwrite partial output: $partial" >&2
  exit 1
fi

echo "[sample_stage5_puzzle_motifs] source=$source offset=$source_offset limit=$limit output=$out"
"$PY" -m datagen.sample_positions \
  --puzzles "$source" \
  --config "$CFG" \
  --output "$partial" \
  --skip-puzzles "$source_offset" \
  --max-puzzles "$limit"
mv -- "$partial" "$out"
echo "[sample_stage5_puzzle_motifs] complete: $(wc -l < "$out") rows -> $out"
