#!/usr/bin/env bash
# Submit the exact stage-4 preparation chain; --train also queues training.
# Run from the repository root. Prerequisite: earlier curriculum stages are prepared.
# bash scripts/datagen/stage4/prepare_stage4.sh [--train]
set -euo pipefail
REPO="${REPO:-$PWD}"
cd "$REPO"
export REPO
[[ $# == 0 || ( $# == 1 && "$1" == --train ) ]] || { echo "usage: $0 [--train]" >&2; exit 1; }
submit() {
  local job
  job=$(sbatch --parsable "$@")
  printf '%s\n' "${job%%;*}"
}
SAMPLE=$(submit scripts/datagen/stage4/sample_stage4.sbatch)
BUILD=$(submit --dependency=afterok:"$SAMPLE" scripts/datagen/stage4/build_stage4.sbatch)
READY=$(submit --dependency=afterok:"$BUILD" scripts/datagen/stage4/mix_stage4.sbatch)
echo "stage 4 data-ready job: $READY" >&2
if [[ "${1:-}" == --train ]]; then
  READY=$(submit --dependency=afterok:"$READY" scripts/train/stage4/train_stage4.sbatch)
  echo "stage 4 training job: $READY" >&2
fi
printf '%s\n' "$READY"
