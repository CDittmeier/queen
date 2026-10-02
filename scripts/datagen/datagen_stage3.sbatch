#!/bin/bash -l
#SBATCH --job-name=datagen_stage3
#SBATCH --output=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm/joblog/%A-%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=danqic
#SBATCH --time=10:00:00
# Wait for the forward position sampling (job 10104230 -> data/lichess/forward/)
# to finish OK. Update the id if that job is resubmitted; drop this line once
# data/lichess/forward/ is already on disk.
#SBATCH --dependency=afterok:10104230
#
# Stage-3 forward-task datagen (build only; positions come from
# scripts/datagen/sample_forward.sbatch -> data/lichess/forward/):
#   1. build the QA datasets via configs/sample_instances/stage3.yaml -> data/stage3/
#      (7 per-task train arrows + val + test; every record is a forward
#       "After the moves <seq>, <question>" example answered on the final board)
#   2. concatenate the 7 per-task train arrows into data/stage3/train_all.arrow
#      (equal sizes => uniform 1/7; DataLoader shuffles)
#
# Usage:  sbatch scripts/datagen/datagen_stage3.sbatch
set -euo pipefail

REPO=/scratch/gpfs/DANQIC/ab4197/mydata/p-chess-lm
cd "$REPO"
export PYTHONPATH="$REPO"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/joblog"

CONFIG="${1:-$REPO/configs/sample_instances/stage3.yaml}"

# Fresh start for the build output (positions in data/lichess/forward/ are kept).
OUT="$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_path'])")"
rm -rf "$REPO/$OUT"

# --- 1. build QA datasets --------------------------------------------------
"$PY" -m datagen.build_qa_dataset --config "$CONFIG"

# --- 2. concatenate per-task train arrows into one (uniform 1/7) ------------
"$PY" - "$REPO/$OUT" <<'PYEOF'
import glob, json, os, sys
from datasets import load_from_disk, concatenate_datasets
root = sys.argv[1]
parts = sorted(glob.glob(f"{root}/training_datasets/*.arrow"))
dss = [load_from_disk(p) for p in parts]
print("[concat] per-task sizes:", {os.path.basename(p): len(d) for p, d in zip(parts, dss)})
combined = concatenate_datasets(dss)
out = f"{root}/train_all.arrow"
combined.save_to_disk(out)
pov = json.load(open(f"{parts[0]}/dataset_config.json"))["pov"]
json.dump({"pov": pov, "split": "train",
           "note": "concatenated stage-3 forward per-task arrows",
           "records": len(combined),
           "tasks": [os.path.basename(p).replace('.arrow','') for p in parts]},
          open(f"{out}/dataset_config.json", "w"), indent=2)
print(f"[concat] -> {out}  ({len(combined):,} records, pov={pov})")
PYEOF
echo "[datagen] done -> $OUT (train_all.arrow, val_dataset.arrow, test_dataset.arrow)"
