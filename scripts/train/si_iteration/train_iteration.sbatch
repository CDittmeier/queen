#!/usr/bin/env bash
#SBATCH --job-name=train_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=25:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Fixed four-GPU SFT recipe; latest-checkpoint resume is handled by train_sft.sbatch.
# Resubmit this same script to continue; no automatic dependency submissions.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

CONFIG="${1:-configs/train/self_distill.yaml}"
[[ "${SLURM_GPUS_ON_NODE:-4}" == 4 ]] || { echo "This recipe requires 4 GPUs" >&2; exit 1; }
# Require packed datasets, not raw consolidation JSONL.
TRAIN=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["train_dataset"])' "$CONFIG")
VALID=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["eval_dataset"])' "$CONFIG")
[[ -d "$TRAIN" && -d "$VALID" ]] || { echo "Missing packed Arrow datasets: $TRAIN / $VALID" >&2; exit 1; }
exec bash "$REPO/scripts/train/common/train_sft.sbatch" "$CONFIG"
