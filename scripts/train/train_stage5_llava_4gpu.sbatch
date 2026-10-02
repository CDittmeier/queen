#!/usr/bin/env bash
#SBATCH --job-name=train_stage5_llava_4gpu
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mem=256G   # 128G OOM-killed during a prior checkpoint's rank-0 gather
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#
# Four-H100 Stage-5 LLaVA training. Cold start warm-starts from the stage-4
# checkpoint in `init_from:`; later slots resume the newest stage-5 checkpoint.
#
# Bounded seven-slot chain by default (7 x 8h = 56 GPU-active hours). The next
# dependency exits immediately once n_steps is complete, and the chain halts if
# a slot makes no checkpoint progress. Override with CHAIN_REMAINING=N.
#
#   sbatch scripts/train/train_stage5_llava_4gpu.sbatch \
#       configs/train/stage5/llava_stage5_sim_fp32_b64.yaml
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="${1:-$REPO/configs/train/stage5/llava_stage5_sim_fp32_b64.yaml}"
[ $# -gt 0 ] && shift
VENV="$REPO/.venv"
source "$VENV/bin/activate"

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
CHAIN_REMAINING="${CHAIN_REMAINING:-6}"
if [ "$STEP" -ge "$NSTEPS" ]; then echo "done: step $STEP >= $NSTEPS"; exit 0; fi
if [ -n "${PREV_STEP+x}" ] && [ "$STEP" -le "$PREV_STEP" ]; then
  echo "no checkpoint progress since step $PREV_STEP; halting chain"; exit 1; fi
if [ "$CHAIN_REMAINING" -gt 0 ]; then
  sbatch --dependency=afterany:"$SLURM_JOB_ID" \
         --export=ALL,PREV_STEP="$STEP",CHAIN_REMAINING=$((CHAIN_REMAINING - 1)) \
         "$REPO/scripts/train/train_stage5_llava_4gpu.sbatch" "$CONFIG" "$@" >/dev/null
  echo "queued next slot (chain_remaining $CHAIN_REMAINING -> $((CHAIN_REMAINING - 1)))"
else
  echo "chain budget exhausted; final slot"
fi
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NGPU" -ne 4 ]; then
  echo "expected 4 GPUs, found $NGPU" >&2
  exit 1
fi
echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}" "$@"
