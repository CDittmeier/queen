#!/usr/bin/env bash
#SBATCH --job-name=pack-stage5-sanity
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
source "$REPO/lab/vllm_nightly_venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO"
python helpers/build_stage5_sanity.py pack --num-shards 4 --device cpu
