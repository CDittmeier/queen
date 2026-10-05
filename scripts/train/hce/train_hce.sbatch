#!/usr/bin/env bash
#SBATCH --job-name=train_stage5_flamingo_hce
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# Resumable four-H100 Stage-5 Flamingo HCE training. A cold start overlays the
# Stage-4 weights named by init_from; later slots resume the newest Stage-5
# checkpoint. Resume slots are prequeued explicitly with Slurm dependencies;
# once n_steps is reached, any surplus tail slot exits immediately.
#
#   sbatch scripts/train/hce/train_hce.sbatch CONFIG.yaml
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export REPO
CONFIG="${1:-$REPO/configs/train/stage5/flamingo_stage5_040_puzzles100k_unfrozen.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
source "$REPO/scripts/environment.sh"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_PROJECT=chesslm-stage5
export WANDB_DIR="$REPO/runs"

EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN="$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP"
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(ls -d "$RUN"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
STEP=0; [ -n "$LATEST" ] && STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))

if [ "$STEP" -ge "$NSTEPS" ]; then
  echo "done: step $STEP >= $NSTEPS"
  exit 0
fi

RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NGPU" -ne 4 ]; then
  echo "expected 4 GPUs, found $NGPU" >&2
  exit 1
fi
echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "$REPO/train.py" \
  --config "$CONFIG" "${RESUME[@]}" "$@"
