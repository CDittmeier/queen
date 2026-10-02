# `datagen/` — QA dataset pipeline

Three stages: raw positions → per-task QA arrows → optional cross-stage mix.

```
lichess PGN / PGN.zst
        │
        ▼  sample_positions.py   (config: configs/sample_positions/*.yaml)
positions.jsonl   ([start_fen, moves, end_fen] per line, optionally feature-gated)
        │
        ▼  build_qa_dataset.py   (config: configs/sample_instances/*.yaml)
<output_path>/
├── test_dataset.arrow/         HF dir + dataset_config.json
├── test_dataset_modeA.arrow/   (optional — shared-positions exhaustive)
├── val_dataset.arrow/
└── training_datasets/
    └── <task>.arrow/           one per registered task
        │
        ▼  build_train_all.py
<output_path>/train_all.arrow   per-task arrows concatenated for the trainer
        │
        ▼  mix_arrows.py        (optional — cross-stage replay mixture)
data/<stage>/train_with_mix.arrow
```

Per-task QA logic lives in [`tasks/`](tasks/README.md). Shared rendering
helpers live in `prose.py`; the per-position feature extractor that backs
the stage-2 dynamic tasks lives in `position_features.py`.

---

## `sample_positions.py`

Streams a Lichess PGN and writes sampled positions to JSONL. Each line is
`[start_fen, move_list, end_fen]`:

- `start_fen` — FEN up to 7 plies before the sampled position
- `move_list` — UCI moves taking `start_fen → end_fen`
- `end_fen`   — the sampled position

Format matches what `lc0` expects (8 history slots = start board + ≤7
moves). Storing more than 7 moves is wasteful.

**Usage:**

```bash
# Sanity check: load one random game, show a random ply
python -m datagen.sample_positions data/lichess/games.pgn.zst --interactive

# Bulk: sample positions across all games in the file
python -m datagen.sample_positions data/lichess/games.pgn.zst \
    --config configs/sample_positions/random.yaml \
    --output data/lichess/train-001.jsonl \
    --seed 42

# Rating-stratified self-distillation puzzles
python -m datagen.sample_positions \
    --puzzles 'data/lichess/puzzles/shard-*.jsonl' \
    --config configs/sample_positions/self_distill_puzzles.yaml \
    --output data/self_distill/puzzles.jsonl \
    --seed 42

# Gameplay only (GPU job; caller assigns a unique job and output name)
python -m datagen.sample_positions \
    --play-games 512 \
    --job stage7_001 \
    --output data/self_distill/stage7_001.games.jsonl \
    --model runs/self-iterate/sol-full-4 \
    --encoder data/engines/lc0_hf_bt5 \
    --stockfish data/engines/stockfish_25080907_x64_avx2 \
    --config configs/sample_positions/self_distill_play.yaml \
    --seed 42

# Scoring + selection only (separate CPU job; no model/encoder needed)
python -m datagen.sample_positions \
    --score-play data/self_distill/stage7_001.games.jsonl \
    --stockfish data/engines/stockfish_25080907_x64_avx2 \
    --config configs/sample_positions/self_distill_play.yaml
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `pgn`           | —                 | Path to `.pgn` or `.pgn.zst` file (positional). |
| `--config`      | —                 | Sampling config YAML (gate features + `per_game`). |
| `--output`      | `positions.jsonl` | Output JSONL path. |
| `--max-games`   | all               | Stop after this many games (1000 in interactive). |
| `--seed`        | `42`              | RNG seed (random in interactive mode). |
| `--interactive` | off               | Show one position + board; don't write JSONL. |

Exactly one source is required: positional `pgn`, `--puzzles`, or
`--play-games`, or `--score-play PATH`. Puzzle mode writes named records containing the Lichess puzzle
metadata and correct line. Play mode alternates the model's color, batches all
live model turns, samples a log-uniform Stockfish opponent strength per game,
and records positions from both movers. A 100k-node Stockfish oracle scores the
actual move at each position. The CPU-side sampler then deduplicates positions,
force-keeps moves with at least a five-point win-rate loss, downsamples the
`|eval| >= 2.5` bucket fourfold, and randomly selects up to
`ceil(played plies / 10)` ordinary positions per game. These are additional to
mistakes, drawn across the whole game rather than one per consecutive window.
Quotas count played plies before filtering; insufficient eligible positions
produce a recorded shortfall. FEN deduplication keeps the largest observed error. These defaults
are explicit in the two `self_distill_*.yaml` configs.

Puzzle and scored gameplay outputs use the common recursive-seed shape
`{record_id, fen, history, extra, cached_root?}` and receive a neighboring
`.manifest.json`. Model-turn play records retain their generated root analysis
in `cached_root`; Stockfish-turn records intentionally do not.

Gameplay checkpoints live under `play.scratch_dir/<job>/`. The caller supplies
`--job` (or `play.job`) and a unique `--output` (or top-level `output`) ending
in `.games.jsonl`, for example `stage7_001.games.jsonl`. Each scratch directory
stores live game move stacks, committed JSONL offsets, and a specification
fingerprint. A changed configuration cannot silently resume a different run.
Every batched cycle is checkpointed; SIGTERM/SIGINT stop after that cycle.
Hard interruptions discard only the uncommitted tail.

`get_play_positions(config, output)` atomically publishes a self-contained raw
shard and returns its path. The first JSONL record holds provenance and counts
under `metadata`; subsequent records contain `game` summaries or `position`
records. The launcher verifies the durable shard and deletes precisely its
own `scratch_dir/<job>/` directory. No receipt or external hash handoff is used.

The separate CPU command reads that raw shard, scores and selects positions,
and publishes the matching `stage7_001.scored.jsonl` plus its manifest.
Only after fsync and verification does it delete `stage7_001.games.jsonl`.
The scored file retains the established recursive-seed record format.
The CPU stage needs neither scratch files nor model/encoder paths. Its settings
live under `score:`; selection uses the gameplay seed stored in the raw shard.

CPU scoring restarts from the raw input after an interruption; it no longer
maintains an intermediate scoring cache. An interrupted final publication can
reuse identical scored data; different existing data is never overwritten.
A verified completed scored shard makes repeated CPU invocations a no-op.
A persistent per-shard lock serializes publication/scoring, while separate
job directories protect gameplay resumes. Tiny lock files remain outside job
directories to avoid lock-inode races. Assign distinct game ranges, job names,
and output basenames when splitting work among collaborators.

## Recursive self-distillation mining

### Config-backed task lists

Mining, rebalance preparation, and consolidation also accept durable task YAMLs:
`configs/self_distill/{mine,rebalance,consolidate}.yaml` are example templates,
not submissions. Replace their input lists and model paths for your experiment.

```bash
python -m datagen.self_distill_stages mine --config configs/self_distill/mine.yaml
python -m datagen.self_distill_stages mine --config configs/self_distill/mine.yaml \
    --start-index 20 --end-index 40
python -m datagen.self_distill_stages rebalance --config configs/self_distill/rebalance.yaml
python -m datagen.self_distill_stages consolidate --config configs/self_distill/consolidate.yaml
```

All relative config/model/directory paths resolve against the **repository root**,
never the YAML directory. Task filenames are joined to the configured `input_dir`;
output names are joined to `output_dir`. A string task derives its output directory
as `input.stem + output_suffix` (default suffix: `_<stage>`). A mapping specifies
`input` and `output` explicitly; mine/rebalance may group an explicit list of inputs.
Globs and directory discovery are not accepted in config tasks.

The YAML's `start_index` and `end_index` select a half-open range, defaulting to all.
CLI bounds override the corresponding YAML bounds. Indices always refer to the
full ordered task list, including completed tasks. Workers process selected tasks
sequentially using the existing stage entrypoints in isolated subprocesses.
Only index bounds are CLI overrides in config mode; experiment parameters stay
under `parameters:`. Legacy direct CLI invocation remains available.

Each task saves its resolved parameters (including defaults), input digests, model
fingerprints, index, and originating config path under `output_dir/.tasks/`.
Concurrent ownership of the same output is rejected; the child inherits the lock
so killing only its launcher does not release ownership prematurely.
Completed tasks are skipped only after checking their output digests.
Incomplete tasks resume using the existing stage checkpoint mechanism.
Mining persists its batch-derived seed counter and handles SIGTERM/SIGINT by
checkpointing after the current batch. Schema-3 mining checkpoints cannot be
resumed by the schema-4 miner; they lack the required continuation state.
Mining and consolidation bind resumes to weight contents, not just paths.
Checkpoint cadence and Stockfish worker count can change on mining resume;
batch sizes and other generation settings remain strict.
Changed inputs/settings/models, duplicate or nested task outputs, and unowned
existing outputs are rejected. Use new output directories for new experiments.
Rebalance publishes its own durable shard manifest for the next stage; enumerate
those filenames in the consolidation YAML. This interface does not silently
discover tasks, merge independent source pools, or change the mining algorithm.



`self_distill_stages.py` turns one or more position JSONLs into records ready
for consolidation. Independent processes use stable, disjoint input partitions:

```bash
python -m datagen.self_distill_stages mine \
    --input data/self_distill/general.jsonl data/self_distill/puzzles.jsonl \
    --output data/self_distill/mined/task-000 \
    --model runs/self-iterate/sol-full-5 \
    --encoder data/engines/lc0_hf_bt5 \
    --stockfish data/engines/stockfish_25080907_x64_avx2 \
    --part 0 --parts 30
```

Each run writes `accepted.jsonl`, `discarded.jsonl`, `state.json`,
`summary.json`, and `input_manifest.json`. The state file contains the FIFO
recursion queue and committed output byte offsets. Repeating the same command
resumes safely; changing an input or policy setting is rejected.

The canonical policy repairs illegal root candidates, injects Stockfish's move
when the model misses it by at least five win-rate points, samples and audits
the three children, and either accepts the improved root/child record or adds
the offending child position to the end of the queue. It uses 100k-node
Stockfish searches, a ten-point mistake/evaluation boundary, and at most five
recursions by default. Model generation is eager and position-conditioned
prefix caching remains disabled.

Completed runs can be repartitioned for the next stage without inference:

```bash
python -m datagen.self_distill_stages rebalance \
    --input 'data/self_distill/mined/task-*' \
    --output data/self_distill/consolidation-inputs \
    --shards 20
```

Rebalancing selects accepted records, removes exact duplicates, preserves the
original JSONL bytes, assigns identities deterministically, and atomically
publishes the shards plus a count/hash manifest. An existing destination is
never replaced; `--overwrite` is rejected. Inputs must remain unchanged throughout
both rebalance passes; changed hashes, changed records, or missing records abort publication.

Each rebalanced shard is consolidated independently with Qwen3.8-27B:

```bash
python -m datagen.self_distill_stages consolidate \
    --input data/self_distill/consolidation-inputs/shard_00000.jsonl \
    --output data/self_distill/consolidated/task-000 \
    --model /path/to/Qwen3.8-27B
```

The default is the exact Sol-full-4 faithful-editor recipe: low reasoning
effort, temperature 1.0, top-p 0.95, top-k 20, a 16,384-token context, and up
to 8,192 output tokens. It supplies Qwen with the three opponent-POV child
analyses, the mandatory selected root move, clean critical-line boundaries,
and mate-distance notes. The original root explanation is retained in output
metadata but deliberately omitted from the prompt. `--material-hints` adds the
root's exact material inventory and asks only for minimal correction of
conflicting material claims; it does not add evaluation hints.

Every generated batch becomes an immutable atomic chunk. Repeating the same
command skips completed identities, while a changed input, prompt, model, or
generation setting is rejected. Once every identity is present, chunks are
assembled in original input order into `consolidated.jsonl`. `manifest.json`
records hashes, settings, throughput, truncations, and audit statistics.
Audits flag malformed fields, POV-token leakage, illegal or unsupplied
promising moves, and critical lines that are not prefixes of the selected
child's mechanically verified line; they record issues without editing output.

### Feature gates (`configs/sample_positions/`)

Each pre-canned config sets a position filter and a `per_game` budget; only
plies matching the gate are emitted. Feature definitions live in
[`position_features.py:FEATURES`](position_features.py).

| Config | Gate features | `per_game` | Notes |
|---|---|---|---|
| `random.yaml`          | (none)                  | 5 | Unfiltered. The default stage-1 source. |
| `check.yaml`           | `in_check`              | 4 | Side-to-move is in check. |
| `have_check.yaml`      | `has_check`             | 4 | At least one legal checking move is available to side-to-move. |
| `checkmate.yaml`       | `is_checkmate`          | 1 | Terminal mate (≤1 per game). |
| `check_near_mate.yaml` | `in_check`, `near_mate` | 1 | Check positions within `near_plies: 6` of a game-ending mate (not the mate ply itself). |
| `stalemate.yaml`       | `is_stalemate`          | 1 | Terminal stalemate (≤1 per game). |

The gated outputs feed the stage-2 task `mix:` sources (see below).

---

## `build_qa_dataset.py`

YAML-driven multi-task multi-split builder. One command produces
test / val / train datasets across every registered task.

**Usage:**

```bash
# Run the config verbatim
python -m datagen.build_qa_dataset --config configs/sample_instances/stage1.yaml

# Tweak any nested field from the CLI without editing the YAML:
python -m datagen.build_qa_dataset --config configs/sample_instances/stage1.yaml \
    --override tasks.piece_on_square.num_positions=50000 \
               splits.val.num_positions=2000 \
               output_path=data/stage1_seed7
```

Flat scalar flags (`--seed`, `--pov` / `--no-pov`, `--output-path`) carry
sentinel defaults — anything not passed falls back to the YAML value.

### Config format

Top-level keys:

| Key | Type | Notes |
|---|---|---|
| `seed`        | int  | Reproducibility. |
| `pov`         | bool | `true` → POV-relative tokens; `false` → board-absolute. |
| `output_path` | str  | Root directory for the generated datasets. |
| `tasks`       | dict | `{task_name: task_spec}` — see below. |
| `splits`      | dict | `{train, val, test, test_modeA?}` — each names its shards + sampling mode. |

`tasks` only knows about the task names registered in
[`tasks/__init__.py:TASKS`](tasks/__init__.py).

### Task spec

Each task carries `num_positions`, `num_questions`, and optionally `mix`:

```yaml
tasks:
  piece_on_square:
    num_positions: 200000
    num_questions: 4
  checking_moves:
    num_positions: 200000
    num_questions: 1
    mix:
      - {source: data/lichess/have_check/{split}-*.jsonl, frac: 0.7}
      - {source: shards,                                  frac: 0.3}
```

When `mix:` is absent the task draws from the split's top-level `shards`.
When `mix:` is present, it lists one or more sources with `frac:` weights:

- `{split}` in a source path is replaced at runtime by `train` / `val` / `test`.
- The literal string `source: shards` refers back to the split's top-level
  shard list (the "random" / unfiltered pool).
- Streams are deterministically weight-interleaved via a smooth weighted
  round-robin. One stream at weight `1.0` == a plain pool pass.

### Concatenating per-task arrows → `train_all.arrow`

Per-task arrows live under `<output_path>/training_datasets/<task>.arrow/`.
The trainer reads one `train_all.arrow` per stage:

```bash
python -m datagen.build_train_all --in-dir data/stage1 \
    --note "concatenated 7 per-task stage1 arrows"
```

Reads `<in-dir>/training_datasets/*.arrow`, concatenates them in sorted
filename order, writes `<in-dir>/train_all.arrow` with a top-level
`dataset_config.json` carrying `pov` / `records` / `tasks`. Asserts every
per-task arrow agrees on `pov`. See [`build_train_all.py`](build_train_all.py).

### Cross-stage dataset mixing → `mix_arrows.py`

For replay across curriculum stages (e.g. 90% stage-2 + 10% stage-1):

```bash
python -m datagen.mix_arrows \
    --inputs data/stage2/train_all.arrow data/stage1/train_all.arrow \
    --dist   9 1 \
    --out    data/stage2/train_with_stage1_mix.arrow
```

`--dist` weights are post-normalized to a probability distribution. The
largest-weight input is the "anchor" and is used whole; every other input
is uniformly sub-sampled to `N_anchor * p_i / p_anchor` rows. Generalizes
to N inputs; the 90/10 case is just N=2. Asserts column-name and `pov`
agreement across inputs. See [`mix_arrows.py`](mix_arrows.py).

### Per-arrow `dataset_config.json`

Each `.arrow/` directory carries its own self-describing config. The
trainer reads only `pov` from the top-level (which is also the only key
in the concatenated `train_all.arrow`'s config).

Per-task train arrow:

```json
{
  "pov":           true,
  "seed":          0,
  "split":         "train",
  "task":          "piece_on_square",
  "task_spec":     { "num_positions": 200000, "num_questions": 4 },
  "train_shards":  ["data/lichess/train-*.jsonl"],
  "summary_stats": {
    "positions":            200000,
    "records":              800000,
    "capped_num_questions": false,
    "answer_class_freq":    { "<SQUARE_A1>": ..., "<SQUARE_A2>": ..., ... }
  }
}
```

Eval-split arrows write `"tasks": [...]` (the full task list) and
`"split_spec": {...}` in place of `"task"` / `"task_spec"`. The full input
YAML for repro lives at the user's `--config` path — not duplicated into
every arrow.

---

## Eval splits

Each of `val`, `test`, and (optional) `test_modeA` picks exactly one mode:

- **Per-task balanced — `per_task: N` + `num_questions: K`**: each task
  pulls `N` positions from the shared stream; each position emits `K`
  distinct queries (via `sample_n`). `K` is silently capped at the task's
  `MAX_UNIQUE_QUERIES` (one warning per cap). The cheap default for both
  `val` and `test`.

- **Shared-positions exhaustive ("mode A") — `num_positions: N`**: load
  `N` unique FENs; for each FEN every task emits `sample_all(board)`
  records (every distinct entity). Total records ≈ `N × Σ MAX_UNIQUE_QUERIES`.
  Lands at `test_dataset_modeA.arrow` (a separate slot from `test`). Tasks
  that carry a `mix:` spec can't run here — mode A runs every task on the
  same positions, but `mix` tasks pull from their own source pool; the
  validator rejects the combination.
