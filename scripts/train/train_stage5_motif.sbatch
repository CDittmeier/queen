#!/usr/bin/env bash
#SBATCH --job-name=stage5_motif_sft
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%j-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="$REPO/configs/train/stage5/flamingo_stage5_motif_frozen.yaml"
EXPECTED_GPUS=${EXPECTED_GPUS:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-}
RACE_JOB_IDS=${RACE_JOB_IDS:-}
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_PROJECT=chesslm-stage5
export WANDB_DIR="$REPO/runs"

# A smaller-GPU race job can make useful progress while the full-node job is
# queued. Stop the race job as soon as any designated full-node arm starts;
# the latter will resume from the latest complete checkpoint below.
RACE_WATCHER_PID=""
cleanup_race_watcher() {
  if [ -n "$RACE_WATCHER_PID" ]; then
    kill "$RACE_WATCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup_race_watcher EXIT
if [ -n "$RACE_JOB_IDS" ]; then
  (
    while true; do
      if squeue -h -j "$RACE_JOB_IDS" -t RUNNING | grep -q .; then
        echo "race winner is running ($RACE_JOB_IDS); cancelling $SLURM_JOB_ID"
        scancel "$SLURM_JOB_ID"
        exit 0
      fi
      sleep 5
    done
  ) &
  RACE_WATCHER_PID=$!
fi

EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN=$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(find "$RUN" -maxdepth 1 -type d -name 'step_*' 2>/dev/null \
  | sort -t_ -k2 -n | tail -1 || true)
STEP=0
if [ -n "$LATEST" ]; then
  STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))
fi
if [ "$STEP" -ge "$NSTEPS" ]; then
  echo "done: step $STEP >= $NSTEPS"
  exit 0
fi

RESUME=()
if [ -n "$LATEST" ]; then
  RESUME=(--resume-from "$LATEST")
fi
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [ "$NGPU" -ne "$EXPECTED_GPUS" ]; then
  echo "expected $EXPECTED_GPUS GPUs, found $NGPU" >&2
  exit 1
fi

TRAIN_OVERRIDES=()
if [ -n "$GRAD_ACCUM_STEPS" ]; then
  TRAIN_OVERRIDES+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
fi

echo "launching: config=$CONFIG ngpu=$NGPU grad_accum=${GRAD_ACCUM_STEPS:-config} resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" \
  --config "$CONFIG" "${TRAIN_OVERRIDES[@]}" "${RESUME[@]}"
