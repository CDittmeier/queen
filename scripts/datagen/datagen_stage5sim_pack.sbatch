#!/bin/bash -l
#SBATCH --job-name=stage5sim_pack
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=04:00:00
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$REPO/joblog"

exec "$REPO/.venv/bin/python" -u helpers/build_stage5sim_arrows.py
