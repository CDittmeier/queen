#!/usr/bin/env bash
# Submit one representative round through Arrow packing; optionally queue SFT.
# bash scripts/datagen/si_iteration/prepare_iteration.sh [configs/self_distill/iteration] [--train]
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/environment.sh"
ROUND_CONFIG="${1:-configs/self_distill/iteration}"
[[ $# -le 2 && ( -z "${2:-}" || "${2:-}" == --train ) ]] || { echo "Usage: $0 [config-dir] [--train]" >&2; exit 1; }

# Refuse a parent mismatch between generation and subsequent training.
TRAIN_CONFIG="$(round_value training_config)"
[[ "$(yaml_value "$TRAIN_CONFIG" init_from)" == "$(round_value source_checkpoint)" ]] || {
  echo "recipe.yaml source_checkpoint must match training config init_from" >&2; exit 1;
}
submit() {
  local job
  job=$(sbatch --parsable "$@")
  printf '%s\n' "${job%%;*}"
}
export REPO
MINE_TASKS=$(task_count "$ROUND_CONFIG/mine.yaml")
CONSOLIDATE_TASKS=$(task_count "$ROUND_CONFIG/consolidate.yaml")
EXPORT=$(submit scripts/datagen/si_iteration/export_iteration.sbatch "$ROUND_CONFIG")
SAMPLE=$(submit scripts/datagen/si_iteration/sample_iteration.sbatch "$ROUND_CONFIG")
PLAY=$(submit --dependency=afterok:"$EXPORT" scripts/datagen/si_iteration/play_iteration.sbatch "$ROUND_CONFIG")
SCORE=$(submit --dependency=afterok:"$PLAY" scripts/datagen/si_iteration/score_iteration_play.sbatch "$ROUND_CONFIG")
MINE=$(submit --dependency=afterok:"$EXPORT:$SAMPLE:$SCORE" --array="0-$((MINE_TASKS - 1))" \
  scripts/datagen/si_iteration/mine_iteration.sbatch "$ROUND_CONFIG")
REBALANCE=$(submit --dependency=afterok:"$MINE" scripts/datagen/si_iteration/rebalance_iteration.sbatch "$ROUND_CONFIG")
CONSOLIDATE=$(submit --dependency=afterok:"$REBALANCE" --array="0-$((CONSOLIDATE_TASKS - 1))" \
  scripts/datagen/si_iteration/consolidate_iteration.sbatch "$ROUND_CONFIG")
echo "Export=$EXPORT sampling=$SAMPLE play=$PLAY scoring=$SCORE mining=$MINE rebalance=$REBALANCE consolidation=$CONSOLIDATE" >&2
PACK=$(submit --dependency=afterok:"$CONSOLIDATE" scripts/datagen/si_iteration/pack_iteration.sbatch "$ROUND_CONFIG")
echo "Packing=$PACK" >&2
if [[ "${2:-}" == --train ]]; then
  TRAIN=$(submit --dependency=afterok:"$PACK" scripts/train/si_iteration/train_iteration.sbatch "$TRAIN_CONFIG")
  echo "Training=$TRAIN" >&2
  printf '%s\n' "$TRAIN"
else
  printf '%s\n' "$PACK"
fi
