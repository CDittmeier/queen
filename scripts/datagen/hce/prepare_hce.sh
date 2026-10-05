#!/usr/bin/env bash
# Exact 0/0/40 HCE recipe: game pool + 300k puzzle roots -> game train + 100k puzzles.
# bash scripts/datagen/hce/prepare_hce.sh [--train]
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
GAME_POOL=$(submit scripts/datagen/hce/sample_hce.sbatch)
PUZZLE_POOL=$(submit scripts/datagen/hce/sample_hce_puzzles.sbatch)
GAMES=$(submit --dependency=afterok:"$GAME_POOL" scripts/datagen/hce/build_hce_games.sbatch)
PUZZLES=$(submit --dependency=afterok:"$PUZZLE_POOL" scripts/datagen/hce/build_hce_puzzles.sbatch)
READY=$(submit --dependency=afterok:"$GAMES:$PUZZLES" scripts/datagen/hce/pack_hce_mixture.sbatch)
echo "HCE data-ready job: $READY" >&2
if [[ "${1:-}" == --train ]]; then
  READY=$(submit --dependency=afterok:"$READY" scripts/train/hce/train_hce.sbatch)
  echo "HCE training job: $READY" >&2
fi
printf '%s\n' "$READY"
