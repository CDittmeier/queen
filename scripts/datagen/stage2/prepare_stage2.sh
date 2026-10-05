#!/usr/bin/env bash
# Submit the exact stage-2 preparation chain; --train also queues training.
# Run from the repository root. Prerequisite: earlier curriculum stages are prepared.
# bash scripts/datagen/stage2/prepare_stage2.sh [--train]
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
SAMPLE=$(submit scripts/datagen/stage2/sample_stage2.sbatch)
BUILD=$(submit --dependency=afterok:"$SAMPLE" scripts/datagen/stage2/build_stage2.sbatch)
READY=$(submit --dependency=afterok:"$BUILD" scripts/datagen/stage2/mix_stage2.sbatch)
echo "stage 2 data-ready job: $READY" >&2
if [[ "${1:-}" == --train ]]; then
  READY=$(submit --dependency=afterok:"$READY" scripts/train/stage2/train_stage2.sbatch)
  echo "stage 2 training job: $READY" >&2
fi
printf '%s\n' "$READY"
