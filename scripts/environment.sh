#!/usr/bin/env bash
# Shared environment; submit jobs from the repository root or export REPO.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
[[ -f train.py && -d datagen ]] || { echo "REPO must point to the queen repository" >&2; exit 1; }
export REPO
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
export PYTHON
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$REPO/data/engines/cudalibs:${LD_LIBRARY_PATH:-}"

# Shared YAML access for iteration launchers; no data-generation logic lives here.
yaml_value() {
  "$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
round_value() { yaml_value "${ROUND_CONFIG:-configs/self_distill/iteration}/recipe.yaml" "$1"; }
task_count() {
  "$PYTHON" -c 'import sys,yaml; print(len(yaml.safe_load(open(sys.argv[1]))["tasks"]))' "$1"
}
