#!/usr/bin/env bash
#SBATCH --job-name=build_instances
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --partition=cpu

# Build template QA instances from existing position pools (no position sampling).
# sbatch scripts/datagen/common/build_instances.sbatch --config configs/sample_instances/stage1.yaml
# Also supports the builder's --stage5-pack flag for the legacy HCE layout.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

exec "$PYTHON" -u -m datagen.build_qa_dataset "$@"
