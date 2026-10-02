#!/usr/bin/env bash
#SBATCH --job-name=stage4_eval
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#SBATCH --array=0-14
#
# Stage-4 TEST eval of the 15 best-val checkpoints on the stage-1, stage-2,
# stage-3 AND stage-4 full test sets. Each array index evals one checkpoint on
# 1 GPU, running eval.py four times (once per test set) to capture forward-task
# performance plus earlier-stage retention.
#
# Mirrors scripts/train_sweeps/stage3_eval.sbatch: eval.py rebuilds the model
# from the sweep config and overlays <ckpt>/trainable.pt.
#
# The best-val step for each run is the argmax of overall/correct in that run's
# metrics.jsonl (ties -> earliest step); baked into CKPT below.
#
#   sbatch scripts/train_sweeps/stage4_eval.sbatch            # all 15
#   sbatch --array=3,7 scripts/train_sweeps/stage4_eval.sbatch  # a subset
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
cd "$REPO"
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Shared full test sets (no --max-examples).
STAGE1_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage1/test_dataset.arrow
STAGE2_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage2/test_dataset.arrow
STAGE3_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage3/test_dataset.arrow
STAGE4_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage4/test_dataset.arrow
OUTDIR="$REPO/runs/stage4_eval"
mkdir -p "$OUTDIR"

# Per-index: "<name> <config> <ckpt step dir>". NAME is the output key ->
# $OUTDIR/<name>__stage{1,2,3,4}.json. CONFIG points at the canonical sweep yaml
# (1:1 with the run). Best-val step is baked into CKPT.
JOBS=(
  # --- SmolLM3-3B, Flamingo (s4e1) ---
  "stage4_smollm_flamingo_5e-5    configs/train_sweeps/s4e1/flamingo_5e-5.yaml           runs/stage4_exp1/stage4_smollm_flamingo_5e-5/step_0050000"
  "stage4_smollm_flamingo_1e-4    configs/train_sweeps/s4e1/flamingo_1e-4.yaml           runs/stage4_exp1/stage4_smollm_flamingo_1e-4/step_0058000"
  "stage4_smollm_flamingo_2e-4    configs/train_sweeps/s4e1/flamingo_2e-4.yaml           runs/stage4_exp1/stage4_smollm_flamingo_2e-4/step_0048000"
  # --- Gemma3-4B, Flamingo (s4e2) ---
  "stage4_gemma_flamingo_5e-5     configs/train_sweeps/s4e2/gemma_flamingo_5e-5.yaml     runs/stage4_exp2/stage4_gemma_flamingo_5e-5/step_0056000"
  "stage4_gemma_flamingo_1e-4     configs/train_sweeps/s4e2/gemma_flamingo_1e-4.yaml     runs/stage4_exp2/stage4_gemma_flamingo_1e-4/step_0042000"
  "stage4_gemma_flamingo_2e-4     configs/train_sweeps/s4e2/gemma_flamingo_2e-4.yaml     runs/stage4_exp2/stage4_gemma_flamingo_2e-4/step_0052000"
  # --- Gemma3-4B, LLaVA (s4e2) ---
  "stage4_gemma_llava_5e-5        configs/train_sweeps/s4e2/gemma_llava_5e-5.yaml        runs/stage4_exp2/stage4_gemma_llava_5e-5/step_0046000"
  "stage4_gemma_llava_1e-4        configs/train_sweeps/s4e2/gemma_llava_1e-4.yaml        runs/stage4_exp2/stage4_gemma_llava_1e-4/step_0048000"
  "stage4_gemma_llava_2e-4        configs/train_sweeps/s4e2/gemma_llava_2e-4.yaml        runs/stage4_exp2/stage4_gemma_llava_2e-4/step_0046000"
  # --- Qwen3-4B, Flamingo (s4e2) ---
  "stage4_qwen_flamingo_5e-5      configs/train_sweeps/s4e2/qwen_flamingo_5e-5.yaml      runs/stage4_exp2/stage4_qwen_flamingo_5e-5/step_0046000"
  "stage4_qwen_flamingo_1e-4      configs/train_sweeps/s4e2/qwen_flamingo_1e-4.yaml      runs/stage4_exp2/stage4_qwen_flamingo_1e-4/step_0062000"
  "stage4_qwen_flamingo_2e-4      configs/train_sweeps/s4e2/qwen_flamingo_2e-4.yaml      runs/stage4_exp2/stage4_qwen_flamingo_2e-4/step_0056000"
  # --- Qwen3-4B, LLaVA (s4e2) ---
  "stage4_qwen_llava_5e-5         configs/train_sweeps/s4e2/qwen_llava_5e-5.yaml         runs/stage4_exp2/stage4_qwen_llava_5e-5/step_0026000"
  "stage4_qwen_llava_1e-4         configs/train_sweeps/s4e2/qwen_llava_1e-4.yaml         runs/stage4_exp2/stage4_qwen_llava_1e-4/step_0044000"
  "stage4_qwen_llava_2e-4         configs/train_sweeps/s4e2/qwen_llava_2e-4.yaml         runs/stage4_exp2/stage4_qwen_llava_2e-4/step_0048000"
)

read -r NAME CONFIG CKPT <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[stage4_eval] idx=$SLURM_ARRAY_TASK_ID name=$NAME"
echo "[stage4_eval] config=$CONFIG"
echo "[stage4_eval] ckpt=$CKPT"

for STAGE in 1 2 3 4; do
  case "$STAGE" in
    1) TEST="$STAGE1_TEST" ;;
    2) TEST="$STAGE2_TEST" ;;
    3) TEST="$STAGE3_TEST" ;;
    4) TEST="$STAGE4_TEST" ;;
  esac
  OUT="$OUTDIR/${NAME}__stage${STAGE}.json"
  echo "[stage4_eval] -> stage${STAGE} test=$TEST out=$OUT (all examples)"
  python eval.py \
    --config "$REPO/$CONFIG" \
    --ckpt   "$REPO/$CKPT" \
    --test-dataset "$TEST" \
    --out    "$OUT"
done
