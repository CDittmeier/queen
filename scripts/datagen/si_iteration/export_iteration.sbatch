#!/usr/bin/env bash
#SBATCH --job-name=export_iteration
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu

# Export the selected unmerged parent as Flamingo for vLLM. CPU-only.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"

exec "$PYTHON" -u -m models.vllm.merge_weights \
  --arch flamingo --save-dtype bf16 \
  --ckpt "$(round_value source_checkpoint)" \
  --train-dataset "$(round_value source_train_dataset)" \
  --base "$(round_value base_model)" --out "$(round_value export_model)"
