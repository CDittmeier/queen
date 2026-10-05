#!/usr/bin/env bash
#SBATCH --job-name=train_sft
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Train any curriculum, distillation, or HCE config, with Flamingo or LLaVA.
# sbatch scripts/train/common/train_sft.sbatch configs/train/stage1/flamingo.yaml
# Override Slurm resources at submission; match GPU count/accumulation to the config.
# Automatically resumes the newest checkpoint. AUTO_RESUME=0 starts from the config.
# Does not submit dependent jobs: submit any continuation slots explicitly.
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"

CONFIG="${1:?usage: sbatch scripts/train/common/train_sft.sbatch CONFIG.yaml [train.py args ...]}"
shift
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-queen}"
export WANDB_DIR="${WANDB_DIR:-$REPO/runs}"
mkdir -p "$WANDB_DIR"

# Resume the latest checkpoint of this experiment, without changing its warm start.
RESUME=()
if [[ "${AUTO_RESUME:-1}" == 1 ]]; then
  RUN=$("$PYTHON" -c 'import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(pathlib.Path(c["output_dir"])/c["exp_name"])' "$CONFIG")
  shopt -s nullglob
  CHECKPOINTS=("$RUN"/step_*)
  LATEST=""
  for checkpoint in "${CHECKPOINTS[@]}"; do
    [[ -d "$checkpoint" ]] && LATEST="$checkpoint"
  done
  [[ -z "$LATEST" ]] || RESUME=(--resume-from "$LATEST")
fi
NGPU="${SLURM_GPUS_ON_NODE:-${NGPU:-4}}"
echo "config=$CONFIG gpus=$NGPU resume=${LATEST:-none}"
exec "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
  train.py --config "$CONFIG" "${RESUME[@]}" "$@"
