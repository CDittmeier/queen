#!/usr/bin/env bash
#SBATCH --job-name=eval_testset
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#
# Test-set eval of a trained checkpoint via eval.py.
#   sbatch --export=ALL,CONFIG=<train.yaml>,CKPT=<run/step_XXXXXXX>,\
#TEST_DATASET=<test.arrow>,OUT=<out.json>[,MAX_EXAMPLES=N] \
#          scripts/eval/native/evaluate_testset.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
source "$REPO/scripts/environment.sh"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ARGS=(--config "$CONFIG" --ckpt "$CKPT" --test-dataset "$TEST_DATASET" --out "$OUT")
[ -n "${MAX_EXAMPLES:-}" ] && ARGS+=(--max-examples "$MAX_EXAMPLES")

echo "[eval] config=$CONFIG ckpt=$CKPT test=$TEST_DATASET out=$OUT max=${MAX_EXAMPLES:-all}"
"$PYTHON" eval.py "${ARGS[@]}"
