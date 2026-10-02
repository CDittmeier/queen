#!/bin/bash -l
#SBATCH --job-name=sample_s5_motifs
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=08:00:00

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY="$REPO/.venv/bin/python"
ALLOC="$REPO/data/lichess_dbs/allocated/stage5_motifs"
CFG="$REPO/configs/sample_positions/motifs.yaml"
OUT="$REPO/data/lichess/stage5_motifs"
PARTIALS="$REPO/lab/stage5_motifs/partials"

mkdir -p "$OUT" "$PARTIALS" "$REPO/joblog"
mapfile -t shards < <(find "$ALLOC" -maxdepth 1 -type f \
  -name 'train_shardrated_2021-03-*.pgn.zst' | sort)
if [ "${#shards[@]}" -lt 10 ]; then
  echo "expected at least 10 allocated shards, found ${#shards[@]}" >&2
  exit 1
fi

i="$SLURM_ARRAY_TASK_ID"
pgn="${shards[$i]}"
out=$(printf '%s/train-%03d.jsonl' "$OUT" "$i")
partial=$(printf '%s/train-%03d.job-%s.jsonl' "$PARTIALS" "$i" "$SLURM_JOB_ID")
seed=$((i + 1))

if [ -s "$out" ]; then
  echo "[skip] completed output already exists: $out"
  exit 0
fi
if [ -e "$partial" ]; then
  echo "refusing to overwrite partial output: $partial" >&2
  exit 1
fi

echo "[sample_stage5_motifs] shard=$pgn output=$out seed=$seed"
"$PY" -m datagen.sample_positions "$pgn" \
  --config "$CFG" \
  --output "$partial" \
  --seed "$seed"
mv -- "$partial" "$out"
echo "[sample_stage5_motifs] complete: $(wc -l < "$out") rows -> $out"
