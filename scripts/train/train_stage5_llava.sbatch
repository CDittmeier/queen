#!/usr/bin/env bash
#SBATCH --job-name=train_stage5_llava
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --constraint=gpu80
#SBATCH --mem=256G   # 128G OOM-killed during the first checkpoint save's rank-0 gather (node has 1TB)
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#
# Stage-5 LLaVA training (POV) on 8xH100. Cold start warm-starts from the stage-4 checkpoint named
# by `init_from:` (weights only, fresh optimizer/step); every later slot auto-resumes its own newest
# stage-5 checkpoint via --resume-from (which makes train.py skip the warm start).
#
# BOUNDED chain: the launch runs one slot and queues CHAIN_REMAINING more as afterany deps
# (default 2 -> main + 2 deps = 3 x 8h = 24h total), each decrementing the counter. The chain
# also stops early at n_steps or if a slot makes no checkpoint progress. Override the budget with
# CHAIN_REMAINING=N at submit time.
#   sbatch scripts/train/train_stage5_llava.sbatch \
#          configs/train/stage5/llava_stage5_unfrozen_const.yaml
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="${1:-$REPO/configs/train/stage5/llava_stage5_unfrozen_const.yaml}"
[ $# -gt 0 ] && shift
VENV="$REPO/.venv"
source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_PROJECT=chesslm-stage5
export WANDB_DIR="$REPO/runs"

# --- auto-resume + self-chaining across 8h slots ---
EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN="$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP"
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(ls -d "$RUN"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
STEP=0; [ -n "$LATEST" ] && STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))
CHAIN_REMAINING="${CHAIN_REMAINING:-2}"   # deps still to queue after this slot (main default: 2)
if [ "$STEP" -ge "$NSTEPS" ]; then echo "done: step $STEP >= $NSTEPS"; exit 0; fi
if [ "${PREV_STEP:-0}" -gt 0 ] && [ "$STEP" -le "$PREV_STEP" ]; then
  echo "no checkpoint progress since step $PREV_STEP; halting chain"; exit 1; fi
if [ "$CHAIN_REMAINING" -gt 0 ]; then
  sbatch --dependency=afterany:"$SLURM_JOB_ID" \
         --export=ALL,PREV_STEP="$STEP",CHAIN_REMAINING=$((CHAIN_REMAINING - 1)) \
         "$REPO/scripts/train/train_stage5_llava.sbatch" "$CONFIG" "$@" >/dev/null
  echo "queued next slot (chain_remaining $CHAIN_REMAINING -> $((CHAIN_REMAINING - 1)))"
else
  echo "chain budget exhausted; final slot"
fi
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}" "$@"
