#!/usr/bin/env bash
#SBATCH --job-name=eval_testset
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#
# Test-set eval of a trained checkpoint via eval.py.
#   sbatch --export=ALL,CONFIG=<train.yaml>,CKPT=<run/step_XXXXXXX>,\
#TEST_DATASET=<test.arrow>,OUT=<out.json>[,MAX_EXAMPLES=N] \
#          scripts/train/eval_testset.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
source "$REPO/.venv/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ARGS=(--config "$CONFIG" --ckpt "$CKPT" --test-dataset "$TEST_DATASET" --out "$OUT")
[ -n "${MAX_EXAMPLES:-}" ] && ARGS+=(--max-examples "$MAX_EXAMPLES")

echo "[eval] config=$CONFIG ckpt=$CKPT test=$TEST_DATASET out=$OUT max=${MAX_EXAMPLES:-all}"
python eval.py "${ARGS[@]}"
