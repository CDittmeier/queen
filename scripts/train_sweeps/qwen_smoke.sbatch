#!/usr/bin/env bash
#SBATCH --job-name=qwen_smoke
#SBATCH --output=/scratch/gpfs/DANQIC/jeff/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#
# Qwen3-4B plumbing smoke — runs BOTH arches (flamingo frozen + llava full-FT)
# back-to-back in a single slot, no chaining. Validates the Qwen3 plumbing end
# to end on real data: adapter_for(qwen3) build, FSDP wrap on Qwen3DecoderLayer,
# the Flamingo manual walk under Qwen's single shared RoPE + full-attention
# masks, the LLaVA full forward, the frozen_vocab slice of Qwen's 267 padding
# rows + the training_utils vocab assertion, identity new-token embed scaling,
# ckpt save/reload, and eval generation stopping on <|im_end|>.
#
# Run it:
#   sbatch scripts/train_sweeps/qwen_smoke.sbatch
# Or interactively on an allocation:
#   salloc --nodes=1 --gres=gpu:2 --constraint=gpu80 --mem=96G --cpus-per-task=16 --time=01:00:00 --partition=pli-c
#   bash scripts/train_sweeps/qwen_smoke.sbatch
set -uo pipefail   # NOT -e: we want both smokes to run even if the first fails

REPO=/scratch/gpfs/DANQIC/jeff/p-chess-lm
CONFIGS=(
  "$REPO/configs/train_sweeps/expintro/qwen_flamingo_smoke.yaml"
  "$REPO/configs/train_sweeps/expintro/qwen_llava_smoke.yaml"
)

source "$REPO/.venv/bin/activate"
module load proxy/default 2>/dev/null || true

# Smoke runs: keep wandb out of the way and dump checkpoints in a throwaway dir.
export WANDB_MODE=disabled
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"

rc=0
for CONFIG in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "[qwen_smoke] launching $CONFIG  (ngpu=$NGPU)"
  echo "============================================================"
  if torchrun --standalone --nproc_per_node="$NGPU" "$REPO/train.py" --config "$CONFIG"; then
    echo "[qwen_smoke] PASS: $CONFIG"
  else
    echo "[qwen_smoke] FAIL (exit $?): $CONFIG"
    rc=1
  fi
done

echo "============================================================"
if [ "$rc" -eq 0 ]; then
  echo "[qwen_smoke] ALL SMOKES PASSED"
else
  echo "[qwen_smoke] ONE OR MORE SMOKES FAILED"
fi
exit "$rc"
