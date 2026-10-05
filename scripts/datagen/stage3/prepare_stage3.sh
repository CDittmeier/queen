#!/usr/bin/env bash
# Submit the exact stage-3 preparation chain; --train also queues training.
# Run from the repository root. Prerequisite: earlier curriculum stages are prepared.
# bash scripts/datagen/stage3/prepare_stage3.sh [--train]
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
SAMPLE=$(submit scripts/datagen/stage3/sample_stage3.sbatch)
BUILD=$(submit --dependency=afterok:"$SAMPLE" scripts/datagen/stage3/build_stage3.sbatch)
READY=$(submit --dependency=afterok:"$BUILD" scripts/datagen/stage3/mix_stage3.sbatch)
echo "stage 3 data-ready job: $READY" >&2
if [[ "${1:-}" == --train ]]; then
  READY=$(submit --dependency=afterok:"$READY" scripts/train/stage3/train_stage3.sbatch)
  echo "stage 3 training job: $READY" >&2
fi
printf '%s\n' "$READY"
