#!/usr/bin/env bash
#SBATCH --job-name=game_probe
#SBATCH --account=danqic
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/lab/checkpoint_vibe/game_tactic_probe_100_s20260808/slurm-%j.out
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
OUT=$REPO/lab/checkpoint_vibe/game_tactic_probe_100_s20260808

cd "$REPO"
mkdir -p "$OUT"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

exec "$REPO/.venv/bin/python" -u helpers/build_game_tactic_probe.py \
    --output "$OUT" --n 100 --candidates 2000 \
    --seed 20260808 --sf-nodes 10000 --workers 16
