#!/usr/bin/env bash
#SBATCH --job-name=eval_regret
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#
# Stage-5 regret eval of a checkpoint: generate narratives on the val split (HF),
# then score legal-rate/regret with the Stockfish oracle (utils/stage5_eval.py).
#   sbatch --export=ALL,CONFIG=<train.yaml>,CKPT=<run/step_XXXXXXX>,\
# VAL_DATASET=<val.arrow>,OUT=<out.json>[,MAX_EXAMPLES=N][,MAX_NEW=4096][,TEMP=0.0] \
#          scripts/eval/native/evaluate_regret.sbatch
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
source "$REPO/scripts/environment.sh"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ARGS=(--config "$CONFIG" --ckpt "$CKPT" --test-dataset "$VAL_DATASET" --out "$OUT"
      --regret --max-new-tokens "${MAX_NEW:-4096}" --temperature "${TEMP:-0.0}")
[ -n "${MAX_EXAMPLES:-}" ] && ARGS+=(--max-examples "$MAX_EXAMPLES")

echo "[eval_regret] ckpt=$CKPT val=$VAL_DATASET out=$OUT max=${MAX_EXAMPLES:-all} temp=${TEMP:-0.0}"
"$PYTHON" eval.py "${ARGS[@]}"
echo "EXIT=$?"
