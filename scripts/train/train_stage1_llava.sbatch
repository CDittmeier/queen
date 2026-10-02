#!/usr/bin/env bash
#SBATCH --job-name=train_stage1_llava
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#
# Stage-1 LLaVA full-FT training (POV, 30k steps, global batch 256 on 2 GPUs).
#   sbatch scripts/train/train_stage1_llava.sbatch
# Auto-resumes from the newest checkpoint and self-chains across 8h pli-cp slots
# until n_steps; the chain stops at completion or if a slot makes no progress.
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="${1:-$REPO/configs/train/stage1/llava.yaml}"
[ $# -gt 0 ] && shift   # pass any remaining args (e.g. --resume-from <ckpt>) through
VENV="$REPO/.venv"
source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# wandb: offline (compute nodes have no internet); sync later with `wandb sync`
export WANDB_MODE=offline
export WANDB_PROJECT=chesslm-stage1
export WANDB_DIR="$REPO/runs"

# --- auto-resume + self-chaining across 8h pli-cp slots ---
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
       "$REPO/scripts/train/train_stage1_llava.sbatch" "$CONFIG" "$@" >/dev/null
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}" "$@"
