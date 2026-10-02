#!/usr/bin/env bash
#SBATCH --job-name=stage1_eval
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
# Stage-1 TEST eval of the 15 best-val checkpoints (5 model/arch combos x 3 LR)
# on the full stage-1 test set. Each array index evals one checkpoint on 1 GPU.
#
# Adapted from scripts/train/eval_testset.sbatch (ab4197): REPO repointed to the
# jeff tree, args hard-coded per index instead of --export. Runs eval.py, which
# rebuilds the model from the sweep config and overlays <ckpt>/trainable.pt.
#
# The best-val step for each run is the argmax of overall/correct in that run's
# metrics.jsonl (ties -> earliest step). Verified 2026-07-02; see plans/experiment_plan.md.
#
#   sbatch scripts/train_sweeps/stage1_eval.sbatch            # all 15
#   sbatch --array=3,7 scripts/train_sweeps/stage1_eval.sbatch  # a subset
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
cd "$REPO"
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Single, shared stage-1 test set (full — no --max-examples).
TEST_DATASET=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/data/stage1/test_dataset.arrow
OUTDIR="$REPO/runs/stage1_eval"
mkdir -p "$OUTDIR"

# Per-index: "<name> <config> <ckpt step dir>". NAME is the run dir name and the
# output key -> $OUTDIR/<name>.json. Runs hold no config, so CONFIG points at the
# canonical sweep yaml (1:1 with the run name). Best-val step is baked into CKPT.
JOBS=(
  # --- SmolLM3-3B, Flamingo (exp3) ---
  "stage1_flamingo_1e-4         configs/train_sweeps/s1e3/flamingo_1e-4.yaml         runs/stage1_exp3/stage1_flamingo_1e-4/step_0020000"
  "stage1_flamingo_7.5e-5       configs/train_sweeps/s1e3/flamingo_7.5e-5.yaml       runs/stage1_exp3/stage1_flamingo_7.5e-5/step_0022000"
  "stage1_flamingo_5e-5         configs/train_sweeps/s1e3/flamingo_5e-5.yaml         runs/stage1_exp3/stage1_flamingo_5e-5/step_0024000"
  # --- Gemma3-4B, Flamingo (exp4) ---
  "stage1_gemma_flamingo_1e-4   configs/train_sweeps/s1e4/flamingo_1e-4.yaml         runs/stage1_exp4/stage1_gemma_flamingo_1e-4/step_0022000"
  "stage1_gemma_flamingo_7.5e-5 configs/train_sweeps/s1e4/flamingo_7.5e-5.yaml       runs/stage1_exp4/stage1_gemma_flamingo_7.5e-5/step_0026000"
  "stage1_gemma_flamingo_5e-5   configs/train_sweeps/s1e4/flamingo_5e-5.yaml         runs/stage1_exp4/stage1_gemma_flamingo_5e-5/step_0024000"
  # --- Gemma3-4B, LLaVA (exp4) ---
  "stage1_gemma_llava_4e-4      configs/train_sweeps/s1e4/llava_4e-4.yaml            runs/stage1_exp4/stage1_gemma_llava_4e-4/step_0010000"
  "stage1_gemma_llava_2e-4      configs/train_sweeps/s1e4/llava_2e-4.yaml            runs/stage1_exp4/stage1_gemma_llava_2e-4/step_0020000"
  "stage1_gemma_llava_1e-4      configs/train_sweeps/s1e4/llava_1e-4.yaml            runs/stage1_exp4/stage1_gemma_llava_1e-4/step_0018000"
  # --- Qwen3-4B, Flamingo (exp5) ---
  "stage1_qwen_flamingo_1e-4    configs/train_sweeps/s1e5/flamingo_1e-4.yaml         runs/stage1_exp5/stage1_qwen_flamingo_1e-4/step_0016000"
  "stage1_qwen_flamingo_7.5e-5  configs/train_sweeps/s1e5/flamingo_7.5e-5.yaml       runs/stage1_exp5/stage1_qwen_flamingo_7.5e-5/step_0022000"
  "stage1_qwen_flamingo_5e-5    configs/train_sweeps/s1e5/flamingo_5e-5.yaml         runs/stage1_exp5/stage1_qwen_flamingo_5e-5/step_0026000"
  # --- Qwen3-4B, LLaVA (exp5) ---
  "stage1_qwen_llava_4e-4       configs/train_sweeps/s1e5/llava_4e-4.yaml            runs/stage1_exp5/stage1_qwen_llava_4e-4/step_0014000"
  "stage1_qwen_llava_2e-4       configs/train_sweeps/s1e5/llava_2e-4.yaml            runs/stage1_exp5/stage1_qwen_llava_2e-4/step_0014000"
  "stage1_qwen_llava_1e-4       configs/train_sweeps/s1e5/llava_1e-4.yaml            runs/stage1_exp5/stage1_qwen_llava_1e-4/step_0016000"
)

read -r NAME CONFIG CKPT <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
OUT="$OUTDIR/$NAME.json"

echo "[stage1_eval] idx=$SLURM_ARRAY_TASK_ID name=$NAME"
echo "[stage1_eval] config=$CONFIG"
echo "[stage1_eval] ckpt=$CKPT"
echo "[stage1_eval] test=$TEST_DATASET out=$OUT (all examples)"

python eval.py \
  --config "$REPO/$CONFIG" \
  --ckpt   "$REPO/$CKPT" \
  --test-dataset "$TEST_DATASET" \
  --out    "$OUT"
