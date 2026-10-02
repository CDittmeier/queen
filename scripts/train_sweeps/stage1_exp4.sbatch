#!/usr/bin/env bash
#SBATCH --job-name=stage1_exp4
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=96G
#SBATCH --time=23:59:00
#SBATCH --partition=pli-c
#SBATCH --array=0-5
#
# Stage-1 exp4 sweep — Gemma3-4B, 6 configs as a SLURM array, 24h chunks with
# auto-resume and self-chaining across pli-c slots until n_steps is hit.
# Each array index routes to one config (see CONFIGS below); the chain
# preserves the index by resubmitting with --array=$SLURM_ARRAY_TASK_ID.
#
# Launch via the wrapper (all 6 equal priority):
#   bash scripts/train_sweeps/launch_exp.sh 1 4
# Or submit subsets manually:
#   sbatch --array=0,1,2 scripts/train_sweeps/stage1_exp4.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
CONFIGS=(
  "$REPO/configs/train_sweeps/s1e4/llava_2e-4.yaml"      # 0  — llava (full FT), bridge lr 2e-4   (baseline)
  "$REPO/configs/train_sweeps/s1e4/llava_1e-4.yaml"      # 1  — llava (full FT), bridge lr 1e-4
  "$REPO/configs/train_sweeps/s1e4/flamingo_1e-4.yaml"   # 2  — flamingo (frozen), bridge lr 1e-4 (baseline)
  "$REPO/configs/train_sweeps/s1e4/flamingo_7.5e-5.yaml" # 3  — flamingo (frozen), bridge lr 7.5e-5
  "$REPO/configs/train_sweeps/s1e4/llava_4e-4.yaml"      # 4  — llava (full FT), bridge lr 4e-4
  "$REPO/configs/train_sweeps/s1e4/flamingo_5e-5.yaml"   # 5  — flamingo (frozen), bridge lr 5e-5  [LOW PRIO]
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

NICE_OPT=""

VENV="$REPO/.venv"
source "$VENV/bin/activate"

# Enable outbound HTTPS on compute nodes (needed for online wandb).
module load proxy/default

export WANDB_MODE=offline   # compute-node proxy is unreliable for streaming/init; sync from login node with: wandb sync runs/wandb/run-*
export WANDB_PROJECT=chesslm-sweep
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
# Queue the next slot for THIS task only — single-task array preserves the index.
sbatch $NICE_OPT \
       --dependency=afterany:"$SLURM_JOB_ID" \
       --array="$SLURM_ARRAY_TASK_ID" \
       --export=ALL,PREV_STEP="$STEP" \
       "$REPO/scripts/train_sweeps/stage1_exp4.sbatch" >/dev/null
RESUME=(); [ -n "$LATEST" ] && RESUME=(--resume-from "$LATEST")

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "launching: idx=$SLURM_ARRAY_TASK_ID config=$CONFIG ngpu=$NGPU resume=${LATEST:-none} step=$STEP/$NSTEPS"
torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG" "${RESUME[@]}"
