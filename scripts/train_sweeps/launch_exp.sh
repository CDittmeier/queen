#!/usr/bin/env bash
# Stage/experiment dispatcher — takes (stage, exp-number) and routes to the
# matching stage${STAGE}_exp${EXP}.sbatch with the correct --array range.
#
#   bash scripts/train_sweeps/launch_exp.sh 1 3
#
# Each case handles the array partitioning for that experiment (some exps split
# high/low priority submissions; others submit all indices in one call).
set -euo pipefail

STAGE="${1:-}"
EXP="${2:-}"
if [ -z "$STAGE" ] || [ -z "$EXP" ]; then
    echo "usage: $(basename "$0") <stage> <exp-number>" >&2
    echo "  e.g. bash $(basename "$0") 1 3" >&2
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$DIR/stage${STAGE}_exp${EXP}.sbatch"
if [ ! -f "$SCRIPT" ]; then
    echo "error: $SCRIPT not found" >&2
    exit 1
fi

# Env carried into every submission (and forward through the sbatch self-chain
# via its --export=ALL). expandable_segments reduces allocator fragmentation —
# the fix that recovered the exp4 LLaVA full-FT OOMs.
EXPORTS="ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

case "${STAGE}_${EXP}" in
    1_3)
        # 6 configs total — high-prio: 0..3 (llava 1e-5, 7.5e-6 / flamingo 1e-4, 7.5e-5);
        # low-prio: 4,5 (llava 5e-6 / flamingo 5e-5).
        HIGH=$(sbatch --parsable --export="$EXPORTS" --array=0,1,2,3 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} high-prio array (0-3) submitted: $HIGH"
        LOW=$(sbatch --parsable --export="$EXPORTS" --array=4,5 --nice=10000 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} low-prio  array (4,5) submitted: $LOW  (--nice=10000)"
        ;;
    1_4)
        # Gemma3-4B — 6 configs total, all equal priority.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-5 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-5) submitted: $JOB"
        ;;
    1_5)
        # Qwen3-4B — 6 configs total, all equal priority.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-5 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-5) submitted: $JOB"
        ;;
    2_1)
        # SmolLM3 Flamingo Stage-2 sweep {8e-5,4e-5,2e-5} — 3 configs, 2 GPUs/run.
        # qos=pli-cp lives in the sbatch #SBATCH header (honored on self-chain too).
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-2 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-2) submitted: $JOB  [2 GPU/run, qos=pli-cp]"
        ;;
    2_2)
        # Gemma/Qwen x Flamingo/LLaVA Stage-2 sweep — 12 configs, 2 GPUs/run, pli-c.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-11 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-11) submitted: $JOB  [2 GPU/run, pli-c]"
        ;;
    3_1)
        # SmolLM3 Flamingo Stage-3 sweep {2.5e-5,5e-5,7.5e-5,1e-4} — 4 configs, 2 GPUs/run.
        # Plain pli-c (no qos) — moved off pli-cp to avoid the GPU-hour cap.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-3 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-3) submitted: $JOB  [2 GPU/run, pli-c]"
        ;;
    3_2)
        # Gemma/Qwen x Flamingo/LLaVA Stage-3 sweep — 14 configs (flam 4-LR, llava
        # 3-LR per model), 2 GPUs/run, pli-c.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-13 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-13) submitted: $JOB  [2 GPU/run, pli-c]"
        ;;
    4_1)
        # SmolLM3 Flamingo Stage-4 sweep {5e-5,1e-4,2e-4} — 3 configs, 2 GPUs/run, pli-c.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-2 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-2) submitted: $JOB  [2 GPU/run, pli-c]"
        ;;
    4_2)
        # Gemma/Qwen x Flamingo/LLaVA Stage-4 sweep — 12 configs (3-LR grid
        # {5e-5,1e-4,2e-4} per combo), 2 GPUs/run, pli-c.
        JOB=$(sbatch --parsable --export="$EXPORTS" --array=0-11 "$SCRIPT")
        echo "stage${STAGE}_exp${EXP} array (0-11) submitted: $JOB  [2 GPU/run, pli-c]"
        ;;
    *)
        echo "error: stage=$STAGE exp=$EXP has no dispatch rule in $(basename "$0")." >&2
        echo "add a case to handle stage${STAGE}_exp${EXP}.sbatch's array/nice partitioning." >&2
        exit 1
        ;;
esac
