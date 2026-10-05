#!/usr/bin/env bash
#SBATCH --job-name=train_stage1_llava
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# Stage-1 LLaVA training on 4 GPUs (single node). Keep global batch 256 by setting
# the config's batch_size so batch_size * grad_accum * 4 == 256 (e.g. 4 * 16 * 4).
#   sbatch scripts/train/stage1/train_stage1_llava_4gpu.sbatch <config>
# Auto-resumes from the newest checkpoint and self-chains across 8h GPU slots
# until n_steps; the chain stops at completion or if a slot makes no progress.
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export REPO
CONFIG="${1:-$REPO/configs/train/stage1/llava.yaml}"
[ $# -gt 0 ] && shift   # pass any remaining args (e.g. --resume-from <ckpt>) through
source "$REPO/scripts/environment.sh"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# wandb: offline (works without compute-node internet access); sync later with `wandb sync`
export WANDB_MODE=offline
export WANDB_PROJECT=chesslm-stage1
export WANDB_DIR="$REPO/runs"

# --- auto-resume + self-chaining across 8h GPU slots ---
EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN="$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP"
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(ls -d "$RUN"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
STEP=0; [ -n "$LATEST" ] && STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))
if [ "$STEP" -ge "$NSTEPS" ]; then echo "done: step $STEP >= $NSTEPS"; exit 0; fi
if [ "${PREV_STEP:-0}" -gt 0 ] && [ "$STEP" -le "$PREV_STEP" ]; then
  echo "no checkpoint progress since step $PREV_STEP; halting chain"; exit 1; fi
# queue the next slot to continue after this one ends (self-terminates via guards above)
sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,PREV_STEP="$STEP" \
       "$REPO/scripts/train/stage1/train_stage1_llava_4gpu.sbatch" "$CONFIG" "$@" >/dev/null
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}" "$@"
