#!/usr/bin/env bash
#SBATCH --job-name=stage3_exp2
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=23:59:00
#SBATCH --partition=pli-c
#SBATCH --array=0-13
#
# Stage-3 sweep, exp2 — the 4 non-SmolLM combos (Gemma/Qwen × Flamingo/LLaVA).
# Flamingo gets a 4-LR grid {2.5e-5, 5e-5, 7.5e-5, 1e-4}; LLaVA a 3-LR grid
# {2e-4, 1e-4, 5e-5}, so 4+3 per model = 14 configs as a SLURM array (0-13).
# 2 GPUs/run on plain pli-c (no qos). Each run warm-starts from its Stage-2 test
# winner (init_from in the config) and trains on Stage-3 data. 8h chunks with
# auto-resume + self-chaining until n_steps; the chain preserves the index via
# --array=$SLURM_ARRAY_TASK_ID.
#
#   bash scripts/train_sweeps/launch_exp.sh 3 2
# Or submit a subset directly:
#   sbatch --array=0,7 scripts/train_sweeps/stage3_exp2.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
CONFIGS=(
  "$REPO/configs/train_sweeps/s3e2/gemma_flamingo_2.5e-5.yaml"  # 0  — gemma flamingo, bridge lr 2.5e-5
  "$REPO/configs/train_sweeps/s3e2/gemma_flamingo_5e-5.yaml"    # 1  — gemma flamingo, bridge lr 5e-5
  "$REPO/configs/train_sweeps/s3e2/gemma_flamingo_7.5e-5.yaml"  # 2  — gemma flamingo, bridge lr 7.5e-5
  "$REPO/configs/train_sweeps/s3e2/gemma_flamingo_1e-4.yaml"    # 3  — gemma flamingo, bridge lr 1e-4
  "$REPO/configs/train_sweeps/s3e2/gemma_llava_2e-4.yaml"       # 4  — gemma llava, bridge lr 2e-4
  "$REPO/configs/train_sweeps/s3e2/gemma_llava_1e-4.yaml"       # 5  — gemma llava, bridge lr 1e-4
  "$REPO/configs/train_sweeps/s3e2/gemma_llava_5e-5.yaml"       # 6  — gemma llava, bridge lr 5e-5
  "$REPO/configs/train_sweeps/s3e2/qwen_flamingo_2.5e-5.yaml"   # 7  — qwen flamingo, bridge lr 2.5e-5
  "$REPO/configs/train_sweeps/s3e2/qwen_flamingo_5e-5.yaml"     # 8  — qwen flamingo, bridge lr 5e-5
  "$REPO/configs/train_sweeps/s3e2/qwen_flamingo_7.5e-5.yaml"   # 9  — qwen flamingo, bridge lr 7.5e-5
  "$REPO/configs/train_sweeps/s3e2/qwen_flamingo_1e-4.yaml"     # 10 — qwen flamingo, bridge lr 1e-4
  "$REPO/configs/train_sweeps/s3e2/qwen_llava_2e-4.yaml"        # 11 — qwen llava, bridge lr 2e-4
  "$REPO/configs/train_sweeps/s3e2/qwen_llava_1e-4.yaml"        # 12 — qwen llava, bridge lr 1e-4
  "$REPO/configs/train_sweeps/s3e2/qwen_llava_5e-5.yaml"        # 13 — qwen llava, bridge lr 5e-5
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

VENV="$REPO/.venv"
source "$VENV/bin/activate"

# Enable outbound HTTPS on compute nodes (needed for online wandb).
module load proxy/default

export WANDB_MODE=offline   # compute-node proxy is unreliable for streaming/init; sync from login node with: wandb sync runs/wandb/run-*
export WANDB_PROJECT=chesslm-sweep
export WANDB_DIR="$REPO/runs"

# expandable_segments allocator; default it here so a bare `sbatch` gets it too.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- auto-resume + self-chaining across pli-c slots ---
EXP=$(awk '/^exp_name:/{print $2}' "$CONFIG")
RUN="$(awk '/^output_dir:/{print $2}' "$CONFIG")/$EXP"
NSTEPS=$(awk '/^n_steps:/{print $2}' "$CONFIG")
LATEST=$(ls -d "$RUN"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
STEP=0; [ -n "$LATEST" ] && STEP=$((10#$(basename "$LATEST" | tr -dc 0-9)))
if [ "$STEP" -ge "$NSTEPS" ]; then echo "done: step $STEP >= $NSTEPS"; exit 0; fi
if [ "${PREV_STEP:-0}" -gt 0 ] && [ "$STEP" -le "$PREV_STEP" ]; then
  echo "no checkpoint progress since step $PREV_STEP; halting chain"; exit 1; fi
# Queue the next slot for THIS task only — single-task array preserves the index.
sbatch --dependency=afterany:"$SLURM_JOB_ID" \
       --array="$SLURM_ARRAY_TASK_ID" \
       --export=ALL,PREV_STEP="$STEP" \
       "$REPO/scripts/train_sweeps/stage3_exp2.sbatch" >/dev/null
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: idx=$SLURM_ARRAY_TASK_ID config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}"
