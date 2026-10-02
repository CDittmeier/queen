#!/usr/bin/env bash
#SBATCH --job-name=stage5sim_grpo
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c

set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
CONFIG="${1:?usage: sbatch $0 CONFIG.yaml}"
source "$REPO/lab/vllm_nightly_venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN=$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(find "$RUN" -mindepth 1 -maxdepth 1 -type d -name 'step_*' 2>/dev/null \
  | sort -t_ -k2n | tail -1 || true)
STEP=0
if [[ -n "$LATEST" ]]; then
  STEP=$((10#$(basename "$LATEST" | tr -dc '0-9')))
fi

if [[ "$STEP" -ge "$NSTEPS" ]]; then
  echo "done: step $STEP >= $NSTEPS"
  exit 0
fi

RESUME=()
if [[ -n "$LATEST" ]]; then
  RESUME=(--resume-from "$LATEST")
fi

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
if [[ "$NGPU" -ne 4 ]]; then
  echo "expected 4 GPUs, found $NGPU" >&2
  exit 1
fi

echo "launching: config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
cd "$REPO"
torchrun --standalone --nproc_per_node="$NGPU" \
  "$REPO/rl.py" --config "$CONFIG" "${RESUME[@]}"
