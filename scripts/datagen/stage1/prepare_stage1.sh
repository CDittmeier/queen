#!/usr/bin/env bash
# Submit the exact stage-1 preparation chain; --train also queues training.
# Run from the repository root. Prerequisite: earlier curriculum stages are prepared.
# bash scripts/datagen/stage1/prepare_stage1.sh [--train]
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
READY=$(submit scripts/datagen/stage1/build_stage1.sbatch)
echo "stage 1 data-ready job: $READY" >&2
if [[ "${1:-}" == --train ]]; then
  READY=$(submit --dependency=afterok:"$READY" scripts/train/stage1/train_stage1.sbatch)
  echo "stage 1 training job: $READY" >&2
fi
printf '%s\n' "$READY"
