# Sol-full-1 → sol-full-2: first self-distillation round

Editable production-entrypoint examples, not a submission script. All paths are
relative to the repository root; run the commands there. Nothing here launches
training or changes the original lab lineage.

## Prerequisites

Use `.venv-inference` and the runtime setup in
`docs/inference-environment.md`. Run GPU stages on allocated compute nodes,
not a login node. Supply these assets (or edit the YAML paths):

- `runs/self_distill/sol_full_1/merged`: the **merged inference export** of sol-full-1,
  not an unmerged training checkpoint.
- `data/engines/lc0_hf_bt5` and `data/engines/stockfish_25080907_x64_avx2`.
- `models/Qwen3-32B`: the editor checkpoint matching the working Qwen export.
- Explicit human PGN and puzzle JSONL sources in the commands below. These example
  input names are not a claim that the assets already exist.

Do not substitute an unrelated model simply because it exists at another path.
The recipe writes only under `data/self_distill/sol_full_2` and its named
scratch directory. It does not export/copy models automatically.

## Position pool

Create the output parent before running the samplers:

```bash
mkdir -p data/self_distill/sol_full_2/positions
.venv-inference/bin/python -m datagen.sample_positions data/lichess_dbs/self_distill/human_000.pgn.zst --config configs/self_distill/sol_full_1_to_2/human.yaml
.venv-inference/bin/python -m datagen.sample_positions --puzzles data/puzzles/self_distill/puzzles.jsonl --config configs/self_distill/sol_full_1_to_2/puzzles.yaml
.venv-inference/bin/python -m datagen.sample_positions --play-games 2000 --config configs/self_distill/sol_full_1_to_2/play.yaml
```

Human sampling selects five positions per game, without a total-position cap.
Puzzle sampling requests 35k with the documented rating buckets; ensure the source
has enough eligible unique puzzles. Exclude held-out/training-overlap positions
when preparing sources; these configs do not implement global exclusion.

After gameplay completes, run the CPU-only phase separately:

```bash
.venv-inference/bin/python -m datagen.sample_positions --score-play data/self_distill/sol_full_2/positions/play_000.games.jsonl --config configs/self_distill/sol_full_1_to_2/play.yaml
```

Successful scoring publishes `play_000.scored.jsonl` and deletes the raw games
file. Keep a separate copy first if raw games are needed for another purpose.
Gameplay scratch is cleaned after durable raw publication.

For another gameplay shard, copy/edit the YAML: change output basename, job name,
and game_start together. With 2,000 games per shard, use starts 0, 2000, 4000, ...
with the same seed. Human workers must receive disjoint PGN sources; puzzle
workers must receive disjoint source pools. Add their explicit outputs to mine.yaml
and their accepted files to rebalance.yaml. No exact accepted yield is promised.

## Mine → rebalance → consolidate

The hybrid defaults to V2. Set `parameters.use_v1_vllm: true` in mine.yaml or
`play.use_v1_vllm: true` in play.yaml for V1. Alternatively append
`--use-v1-vllm` to the mining/gameplay command (including config mode); the CLI
overrides YAML. `--no-use-v1-vllm` explicitly selects V2. Existing V1 mining and
gameplay checkpoints require V1 on resume: the same seed produces different
sampled text between runners. Qwen consolidation does not take this hybrid flag.

```bash
.venv-inference/bin/python -m datagen.self_distill_stages mine --config configs/self_distill/sol_full_1_to_2/mine.yaml
.venv-inference/bin/python -m datagen.self_distill_stages rebalance --config configs/self_distill/sol_full_1_to_2/rebalance.yaml
.venv-inference/bin/python -m datagen.self_distill_stages consolidate --config configs/self_distill/sol_full_1_to_2/consolidate.yaml
```

Wait for **all** mining tasks before rebalancing, and for rebalance before
consolidation. Rebalance deduplicates its supplied accepted records and publishes
four shards. If changing its shard count, update the consolidation task list too.
Final files are `consolidated/shard_0000N/consolidated.jsonl`; subsequent target
filtering, translation, Arrow packing, and SFT are separate workflows.

Mining and consolidation can be split between collaborators using half-open
task-index ranges. For the three mining tasks, one person can append
`--start-index 0 --end-index 1`, the other `--start-index 1 --end-index 3`.
For four consolidation tasks use [0,2) and [2,4). CLI bounds override YAML bounds;
null end means all tasks. Rebalance is one global task, not one per collaborator.
Keep the same ordered task list and parameters when resuming. Do not edit an
already-started experiment in place; use a new output namespace for changed inputs
or semantic settings.

## Provenance and resources

Mining mirrors `lab/two_networks/scripts/sample_sol_full_6_redo_self_play_recursive_wave1.sbatch`:
6 Stockfish workers, 100k nodes, batch/checkpoint 1024, 256 sequences, 4096 output
tokens, five recursions. Consolidation mirrors
`lab/two_networks/scripts/consolidate_sol_iteration06_redo_qwen38.sbatch`:
unhinted editing, chunks 64, context 16384, output 8192, 16 sequences, GPU fraction
0.92. Only model identity, namespace, and task layout represent the first round.
Gameplay uses the approved main-repo selection policy, including loguniform
Stockfish budgets from 100 to 100k.

GPU stages need an 80GB GPU; the bounded integration test used one A100 and 64GB
CPU RAM. Gameplay/mining use six engine workers; provision CPUs accordingly.
Scoring/rebalance need no GPU. These are not full-shard memory or runtime
guarantees, and the YAMLs do not request Slurm resources.
