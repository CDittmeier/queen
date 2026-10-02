#!/bin/bash -l
#SBATCH --job-name=build_s5_motif
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%j-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=02:00:00

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG=${CONFIG:-$REPO/configs/sample_instances/stage5_motif.yaml}
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

"$REPO/.venv/bin/python" -m datagen.build_qa_dataset \
  --config "$CONFIG"
