#!/usr/bin/env bash
#SBATCH --job-name=stage2_eval
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
# Stage-2 TEST eval of the 15 best-val checkpoints (5 model/arch combos x 3 LR)
# on BOTH the stage-1 and stage-2 full test sets. Each array index evals one
# checkpoint on 1 GPU, running eval.py twice (once per test set).
#
# Mirrors scripts/train_sweeps/stage1_eval.sbatch: eval.py rebuilds the model
# from the sweep config and overlays <ckpt>/trainable.pt.
#
# The best-val step for each run is the argmax of overall/correct in that run's
# metrics.jsonl (ties -> earliest step); baked into CKPT below.
#
#   sbatch scripts/train_sweeps/stage2_eval.sbatch            # all 15
#   sbatch --array=3,7 scripts/train_sweeps/stage2_eval.sbatch  # a subset
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
cd "$REPO"
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Shared full test sets (no --max-examples).
STAGE1_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage1/test_dataset.arrow
STAGE2_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage2/test_dataset.arrow
OUTDIR="$REPO/runs/stage2_eval"
mkdir -p "$OUTDIR"

# Per-index: "<name> <config> <ckpt step dir>". NAME is the output key ->
# $OUTDIR/<name>__stage{1,2}.json. CONFIG points at the canonical sweep yaml
# (1:1 with the run). Best-val step is baked into CKPT.
JOBS=(
  # --- SmolLM3-3B, Flamingo (s2e1) ---
  "stage2_smollm_flamingo_8e-5  configs/train_sweeps/s2e1/flamingo_8e-5.yaml         runs/stage2_exp1/stage2_smollm_flamingo_8e-5/step_0032000"
  "stage2_smollm_flamingo_4e-5  configs/train_sweeps/s2e1/flamingo_4e-5.yaml         runs/stage2_exp1/stage2_smollm_flamingo_4e-5/step_0034000"
  "stage2_smollm_flamingo_2e-5  configs/train_sweeps/s2e1/flamingo_2e-5.yaml         runs/stage2_exp1/stage2_smollm_flamingo_2e-5/step_0030000"
  # --- Gemma3-4B, Flamingo (s2e2) ---
  "stage2_gemma_flamingo_8e-5   configs/train_sweeps/s2e2/gemma_flamingo_8e-5.yaml   runs/stage2_exp2/stage2_gemma_flamingo_8e-5/step_0032000"
  "stage2_gemma_flamingo_4e-5   configs/train_sweeps/s2e2/gemma_flamingo_4e-5.yaml   runs/stage2_exp2/stage2_gemma_flamingo_4e-5/step_0032000"
  "stage2_gemma_flamingo_2e-5   configs/train_sweeps/s2e2/gemma_flamingo_2e-5.yaml   runs/stage2_exp2/stage2_gemma_flamingo_2e-5/step_0060000"
  # --- Gemma3-4B, LLaVA (s2e2) ---
  "stage2_gemma_llava_1e-4      configs/train_sweeps/s2e2/gemma_llava_1e-4.yaml      runs/stage2_exp2/stage2_gemma_llava_1e-4/step_0018000"
  "stage2_gemma_llava_5e-5      configs/train_sweeps/s2e2/gemma_llava_5e-5.yaml      runs/stage2_exp2/stage2_gemma_llava_5e-5/step_0024000"
  "stage2_gemma_llava_2.5e-5    configs/train_sweeps/s2e2/gemma_llava_2.5e-5.yaml    runs/stage2_exp2/stage2_gemma_llava_2.5e-5/step_0024000"
  # --- Qwen3-4B, Flamingo (s2e2) ---
  "stage2_qwen_flamingo_8e-5    configs/train_sweeps/s2e2/qwen_flamingo_8e-5.yaml    runs/stage2_exp2/stage2_qwen_flamingo_8e-5/step_0024000"
  "stage2_qwen_flamingo_4e-5    configs/train_sweeps/s2e2/qwen_flamingo_4e-5.yaml    runs/stage2_exp2/stage2_qwen_flamingo_4e-5/step_0036000"
  "stage2_qwen_flamingo_2e-5    configs/train_sweeps/s2e2/qwen_flamingo_2e-5.yaml    runs/stage2_exp2/stage2_qwen_flamingo_2e-5/step_0028000"
  # --- Qwen3-4B, LLaVA (s2e2) ---
  "stage2_qwen_llava_1e-4       configs/train_sweeps/s2e2/qwen_llava_1e-4.yaml       runs/stage2_exp2/stage2_qwen_llava_1e-4/step_0016000"
  "stage2_qwen_llava_5e-5       configs/train_sweeps/s2e2/qwen_llava_5e-5.yaml       runs/stage2_exp2/stage2_qwen_llava_5e-5/step_0030000"
  "stage2_qwen_llava_2.5e-5     configs/train_sweeps/s2e2/qwen_llava_2.5e-5.yaml     runs/stage2_exp2/stage2_qwen_llava_2.5e-5/step_0030000"
)

read -r NAME CONFIG CKPT <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[stage2_eval] idx=$SLURM_ARRAY_TASK_ID name=$NAME"
echo "[stage2_eval] config=$CONFIG"
echo "[stage2_eval] ckpt=$CKPT"

for STAGE in 1 2; do
  case "$STAGE" in
    1) TEST="$STAGE1_TEST" ;;
    2) TEST="$STAGE2_TEST" ;;
  esac
  OUT="$OUTDIR/${NAME}__stage${STAGE}.json"
  echo "[stage2_eval] -> stage${STAGE} test=$TEST out=$OUT (all examples)"
  python eval.py \
    --config "$REPO/$CONFIG" \
    --ckpt   "$REPO/$CKPT" \
    --test-dataset "$TEST" \
    --out    "$OUT"
done
