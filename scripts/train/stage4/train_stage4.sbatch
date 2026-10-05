#!/usr/bin/env bash
#SBATCH --job-name=train_stage4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=23:59:00
#SBATCH --partition=gpu

# Selected stage-4 Flamingo recipe from the original sweep, not the sweep itself.
# Global batch: 8 examples/GPU x 16 accumulation steps x 2 GPUs = 256.
# Config retains the historical training budget; later stages use selected checkpoints.
# sbatch scripts/train/stage4/train_stage4.sbatch
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export REPO PYTHONPATH="$REPO:${PYTHONPATH:-}"
source "$REPO/scripts/environment.sh"
CONFIG="$REPO/configs/train/stage4/flamingo.yaml"
export WANDB_MODE=offline WANDB_PROJECT=chesslm-curriculum WANDB_DIR="$REPO/runs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WANDB_DIR"

# Resume this experiment and queue the next slot until its configured budget is reached.
EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN="$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP"
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(ls -d "$RUN"/step_* 2>/dev/null | sort -V | tail -1 || true)
STEP=0; [ -n "$LATEST" ] && STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))
if [ "$STEP" -ge "$NSTEPS" ]; then echo "done: $STEP >= $NSTEPS"; exit 0; fi
if [[ -n "${PREV_STEP+x}" && "$STEP" -le "$PREV_STEP" ]]; then
  echo "No checkpoint progress since $PREV_STEP; stopping continuation chain" >&2
  exit 1
fi
NGPU="${SLURM_GPUS_ON_NODE:-2}"
[[ "$NGPU" == 2 ]] || { echo "This recipe requires 2 GPUs for global batch 256" >&2; exit 1; }
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,PREV_STEP="$STEP" \
    "$REPO/scripts/train/stage4/train_stage4.sbatch" >/dev/null
fi
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 "$REPO/train.py" \
  --config "$CONFIG" "${RESUME[@]}"
