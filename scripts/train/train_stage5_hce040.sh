#!/usr/bin/env bash
# Launch a stage-5 hce040 puzzle-mix training arm (bounded 3x8h chain via train_stage5_llava.sbatch).
#   scripts/train/train_stage5_hce040.sh nopuzzles     # game positions only
#   scripts/train/train_stage5_hce040.sh puzzles100k   # game + 100k puzzles
# Extra args pass through to train.py; CHAIN_REMAINING=N overrides the chain budget.
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
VARIANT="${1:?usage: $0 nopuzzles|puzzles100k [extra train.py args...]}"
shift
case "$VARIANT" in
  nopuzzles)   CONFIG="$REPO/configs/train/stage5/llava_stage5_040_nopuzzles.yaml" ;;
  puzzles100k) CONFIG="$REPO/configs/train/stage5/llava_stage5_040_puzzles100k.yaml" ;;
  *) echo "unknown variant: $VARIANT (want nopuzzles|puzzles100k)" >&2; exit 1 ;;
esac
sbatch "$REPO/scripts/train/train_stage5_llava.sbatch" "$CONFIG" "$@"
