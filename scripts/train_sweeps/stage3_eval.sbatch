#!/usr/bin/env bash
#SBATCH --job-name=stage3_eval
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/p-chess-lm/joblog/%A_%a-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=pli-c
#SBATCH --array=0-16
#
# Stage-3 TEST eval of the 17 best-val checkpoints (the dead gemma_llava_2e-4 run
# is dropped) on the stage-1, stage-2 AND stage-3 full test sets. Each array
# index evals one checkpoint on 1 GPU, running eval.py three times (once per
# test set) to capture forward-task performance plus earlier-stage retention.
#
# Mirrors scripts/train_sweeps/stage2_eval.sbatch: eval.py rebuilds the model
# from the sweep config and overlays <ckpt>/trainable.pt.
#
# The best-val step for each run is the argmax of overall/correct in that run's
# metrics.jsonl (ties -> earliest step); baked into CKPT below.
#
#   sbatch scripts/train_sweeps/stage3_eval.sbatch            # all 17
#   sbatch --array=3,7 scripts/train_sweeps/stage3_eval.sbatch  # a subset
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
cd "$REPO"
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Shared full test sets (no --max-examples).
STAGE1_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage1/test_dataset.arrow
STAGE2_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage2/test_dataset.arrow
STAGE3_TEST=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage3/test_dataset.arrow
OUTDIR="$REPO/runs/stage3_eval"
mkdir -p "$OUTDIR"

# Per-index: "<name> <config> <ckpt step dir>". NAME is the output key ->
# $OUTDIR/<name>__stage{1,2,3}.json. CONFIG points at the canonical sweep yaml
# (1:1 with the run). Best-val step is baked into CKPT.
JOBS=(
  # --- SmolLM3-3B, Flamingo (s3e1) ---
  "stage3_smollm_flamingo_2.5e-5  configs/train_sweeps/s3e1/flamingo_2.5e-5.yaml        runs/stage3_exp1/stage3_smollm_flamingo_2.5e-5/step_0022000"
  "stage3_smollm_flamingo_5e-5    configs/train_sweeps/s3e1/flamingo_5e-5.yaml          runs/stage3_exp1/stage3_smollm_flamingo_5e-5/step_0010000"
  "stage3_smollm_flamingo_7.5e-5  configs/train_sweeps/s3e1/flamingo_7.5e-5.yaml        runs/stage3_exp1/stage3_smollm_flamingo_7.5e-5/step_0012000"
  "stage3_smollm_flamingo_1e-4    configs/train_sweeps/s3e1/flamingo_1e-4.yaml          runs/stage3_exp1/stage3_smollm_flamingo_1e-4/step_0012000"
  # --- Gemma3-4B, Flamingo (s3e2) ---
  "stage3_gemma_flamingo_2.5e-5   configs/train_sweeps/s3e2/gemma_flamingo_2.5e-5.yaml  runs/stage3_exp2/stage3_gemma_flamingo_2.5e-5/step_0010000"
  "stage3_gemma_flamingo_5e-5     configs/train_sweeps/s3e2/gemma_flamingo_5e-5.yaml    runs/stage3_exp2/stage3_gemma_flamingo_5e-5/step_0012000"
  "stage3_gemma_flamingo_7.5e-5   configs/train_sweeps/s3e2/gemma_flamingo_7.5e-5.yaml  runs/stage3_exp2/stage3_gemma_flamingo_7.5e-5/step_0012000"
  "stage3_gemma_flamingo_1e-4     configs/train_sweeps/s3e2/gemma_flamingo_1e-4.yaml    runs/stage3_exp2/stage3_gemma_flamingo_1e-4/step_0012000"
  # --- Gemma3-4B, LLaVA (s3e2) --- (2e-4 dead, dropped)
  "stage3_gemma_llava_5e-5        configs/train_sweeps/s3e2/gemma_llava_5e-5.yaml       runs/stage3_exp2/stage3_gemma_llava_5e-5/step_0014000"
  "stage3_gemma_llava_1e-4        configs/train_sweeps/s3e2/gemma_llava_1e-4.yaml       runs/stage3_exp2/stage3_gemma_llava_1e-4/step_0012000"
  # --- Qwen3-4B, Flamingo (s3e2) ---
  "stage3_qwen_flamingo_2.5e-5    configs/train_sweeps/s3e2/qwen_flamingo_2.5e-5.yaml   runs/stage3_exp2/stage3_qwen_flamingo_2.5e-5/step_0016000"
  "stage3_qwen_flamingo_5e-5      configs/train_sweeps/s3e2/qwen_flamingo_5e-5.yaml     runs/stage3_exp2/stage3_qwen_flamingo_5e-5/step_0014000"
  "stage3_qwen_flamingo_7.5e-5    configs/train_sweeps/s3e2/qwen_flamingo_7.5e-5.yaml   runs/stage3_exp2/stage3_qwen_flamingo_7.5e-5/step_0012000"
  "stage3_qwen_flamingo_1e-4      configs/train_sweeps/s3e2/qwen_flamingo_1e-4.yaml     runs/stage3_exp2/stage3_qwen_flamingo_1e-4/step_0012000"
  # --- Qwen3-4B, LLaVA (s3e2) ---
  "stage3_qwen_llava_5e-5         configs/train_sweeps/s3e2/qwen_llava_5e-5.yaml        runs/stage3_exp2/stage3_qwen_llava_5e-5/step_0010000"
  "stage3_qwen_llava_1e-4         configs/train_sweeps/s3e2/qwen_llava_1e-4.yaml        runs/stage3_exp2/stage3_qwen_llava_1e-4/step_0008000"
  "stage3_qwen_llava_2e-4         configs/train_sweeps/s3e2/qwen_llava_2e-4.yaml        runs/stage3_exp2/stage3_qwen_llava_2e-4/step_0010000"
)

read -r NAME CONFIG CKPT <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

echo "[stage3_eval] idx=$SLURM_ARRAY_TASK_ID name=$NAME"
echo "[stage3_eval] config=$CONFIG"
echo "[stage3_eval] ckpt=$CKPT"

for STAGE in 1 2 3; do
  case "$STAGE" in
    1) TEST="$STAGE1_TEST" ;;
    2) TEST="$STAGE2_TEST" ;;
    3) TEST="$STAGE3_TEST" ;;
  esac
  OUT="$OUTDIR/${NAME}__stage${STAGE}.json"
  echo "[stage3_eval] -> stage${STAGE} test=$TEST out=$OUT (all examples)"
  python eval.py \
    --config "$REPO/$CONFIG" \
    --ckpt   "$REPO/$CKPT" \
    --test-dataset "$TEST" \
    --out    "$OUT"
done
