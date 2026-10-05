#!/usr/bin/env bash
#SBATCH --job-name=evaluate_analysis
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

# Native Arrow evaluation/generation only. Not the historical PV benchmark or LM Elo.
# sbatch scripts/eval/native/evaluate_analysis.sbatch [configs/eval/analysis.yaml]
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

CONFIG="${1:-configs/eval/analysis.yaml}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
# Read configuration only; all generation/evaluation remains in the existing eval.py.
VALUES=$("$PYTHON" -c '
import sys,yaml
c=yaml.safe_load(open(sys.argv[1])); d=c["datasets"][int(sys.argv[2])]
for v in [c["train_config"],c["checkpoint"],d["path"],d["output"],
          c["temperature"],c["max_new_tokens"],c["max_examples"]]:
    print(v)
' "$CONFIG" "$INDEX")
mapfile -t ARGS <<< "$VALUES"
exec "$PYTHON" -u eval.py --config "${ARGS[0]}" --ckpt "${ARGS[1]}" \
  --test-dataset "${ARGS[2]}" --out "${ARGS[3]}" \
  --temperature "${ARGS[4]}" --max-new-tokens "${ARGS[5]}" --max-examples "${ARGS[6]}"
