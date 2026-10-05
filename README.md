Download our trained models on [Huggingface](https://huggingface.co/collections/princeton-nlp/queen-chess-models)

Extended README and clean code release coming soon!

## Reproducing the curriculum and HCE recipes

Run commands from this repository root. These are fixed historical recipes, not
examples requiring the caller to reconstruct proportions or source selections.
Slurm launchers use generic `gpu` and `cpu` partitions, without account, QoS,
or site-specific GPU constraints. Override partition/resource requests for your
installation. GPU memory requirements have not changed: the inference recipes
target 80GB-class GPUs. No launcher downloads data.

### Template curriculum: stages 1–4

For stage N, run `bash scripts/datagen/stageN/prepare_stageN.sh --train`
(replace both occurrences of N with 1, 2, 3, or 4).
Omit `--train` to prepare data only. Each command prints the final job ID and
queues the required sampling, building, packing/mixing, and (optionally) training
dependencies. Run successive training stages only once the preceding selected
checkpoint exists; these commands do not wait for another stage's continuation chain.

| Stage | Exact preparation chain | Training script |
|---|---|---|
| 1 | `build_stage1.sbatch`: sample allocated PGNs, build task datasets, concatenate | `scripts/train/stage1/train_stage1.sbatch` |
| 2 | `sample_stage2.sbatch` → `build_stage2.sbatch` → `mix_stage2.sbatch` | `scripts/train/stage2/train_stage2.sbatch` |
| 3 | `sample_stage3.sbatch` → `build_stage3.sbatch` → `mix_stage3.sbatch` | `scripts/train/stage3/train_stage3.sbatch` |
| 4 | `sample_stage4.sbatch` → `build_stage4.sbatch` → `mix_stage4.sbatch` | `scripts/train/stage4/train_stage4.sbatch` |

Preparation scripts in that table live under `scripts/datagen/stageN/`.
Every builder includes concatenation to `train_all.arrow`.
All four stages reuse the disjoint game-level train/held-out PGN allocation in
`data/lichess_dbs/allocated/stage1/{train,val_test}_shardrated_*.pgn.zst`.
The sampling seed is 1, and the mixture seed is 0.

- Stage 1: `random.yaml` → `data/lichess/{train-NNN,val_test-NNN}.jsonl`.
- Stage 2: check, checkmate, stalemate, check_near_mate, and have_check pools;
  ordinary positions come from stage 1. Special pool filenames are
  `data/lichess/TYPE/{train,val-test}_NNNN.jsonl`.
- Stage 3: `forward.yaml` → `data/lichess/forward/`, with the ordinary pool naming.
- Stage 4: the five `forward_TYPE.yaml` configs → `data/lichess/forward_TYPE/`,
  with the special pool naming. It also reuses stage 3's ordinary forward pool.

Task counts and task-specific sampling proportions remain in
`configs/sample_instances/stageN.yaml`. Stage 1 trains on its concatenation;
stage 2 mixes stage2:stage1 at 90:10; stage 3 mixes stage3:stage2:stage1 at
86:8:6; stage 4 mixes stage4:stage3:stage2:stage1 at 80:8:6:6.
Final training paths are `data/stage1/train_all.arrow`,
`data/stage2/train_with_stage1_mix.arrow`, and
`data/stage{3,4}/train_with_mix.arrow`.

The selected Flamingo configs are `configs/train/stageN/flamingo.yaml`.
They use two GPUs, batch 8 per GPU, and accumulation 16 (global batch 256).
The original sweep budgets are preserved, not silently shortened to selected steps:

| Stage | Training budget | Checkpoint used by the next stage | Output directory |
|---|---:|---:|---|
| 1 | 50,000 | 20,000 | `runs/stage1_exp3/stage1_flamingo_1e-4` |
| 2 | 60,000 | 32,000 | `runs/stage2_exp1/stage2_smollm_flamingo_8e-5` |
| 3 | 30,000 | 22,000 | `runs/stage3_exp1/stage3_smollm_flamingo_2.5e-5` |
| 4 | 100,000 | 48,000 | `runs/stage4_exp1/stage4_smollm_flamingo_2e-4` |

Training resumes its latest checkpoint and queues continuation slots automatically,
stopping when the configured budget is met or a continuation made no progress.
Stages 1/2 request 8-hour slots; stages 3/4 request 23:59-hour slots.
The LLaVA ablation retains separate `train_stageN_llava.sbatch` launchers and
the original explicitly named four-GPU variants where present.

### HCE branch

Run `bash scripts/datagen/hce/prepare_hce.sh --train` after the stage-4 checkpoint
exists. The exact chain is:

1. `sample_hce.sbatch`: sample the separate
   `data/lichess_dbs/allocated/stage5/train_shardrated_*.pgn.zst` allocation
   (historically shards 00050–00074) into `data/lichess/stage5/`.
2. `sample_hce_puzzles.sbatch`: convert puzzle shards
   `data/lichess/puzzles/shard-{001,002,003}.jsonl` into three pool JSONLs.
3. `build_hce_games.sbatch`: 72 tasks, target 17,362/task, 0/0/40 trees,
   normal de-skew; `build_hce_puzzles.sbatch`: 12 tasks, target 25,000/task,
   same trees, no de-skew.
4. `pack_hce_mixture.sbatch`: hold out 500 game test examples and 200 game
   validation examples, hold out 10,000 puzzles, and mix 100,000 other puzzles
   with game training examples (seed 0). This also retains a game-only dataset.
5. `scripts/train/hce/train_hce.sbatch`: four-GPU Flamingo training using
   `configs/train/stage5/flamingo_stage5_040_puzzles100k_unfrozen.yaml`,
   warm-starting from stage 4 step 48,000. This job resumes when resubmitted;
   unlike the curriculum scripts, it does not queue its own continuations.

The packer writes `data/stage5/0-0-40/{train_all,val_dataset,test_dataset}.arrow`,
`data/stage5/0-0-40-puzzles100k/train_all.arrow`, and
`data/stage5/puzzles-0-0-40/heldout_eval.{arrow,jsonl}`.
The standalone game/puzzle pack scripts are optional; do not run the game packer
concurrently with the mixture packer, which writes the same game dataset paths.
Packing expects complete valid JSONL shards and fails on malformed rows.

### Scope and safety

Existing sampled position files are reused; the launchers no longer delete entire
dataset directories before building. Dataset builders/packers retain their own
output semantics: use fresh output locations or inspect existing outputs before a rerun.
The old hardcoded Slurm dependency IDs are replaced by IDs returned at submission.
No downloads, jobs, or training are triggered by checking out these scripts.

## Sol initialization and one general self-improvement round

These entry points use one representative round, not the original experiment's
job splits or a config for every historical iteration. All paths are repository-relative.
No Python source modifications are required by these launchers.

### Sol distillation SFT

Put the supplied 8,402-example dataset at `data/sol_distill/train.arrow` and
its held-out dataset at `data/sol_distill/validation.arrow`. Run:

```bash
sbatch scripts/train/sol_distill/train_sol_distill.sbatch
```

`configs/train/sol_distill.yaml` sets full decoder fine-tuning, four GPUs,
global batch 16, learning rate 1e-5, constant schedule with 5% warmup,
max length 2048, 4,000 training steps, and validation/checkpoints every 100 steps.
The representative round starts from step 2,100, the selected first-round checkpoint,
not necessarily the last checkpoint of that training budget.

The optimizer/data settings come from the historical 8.4k Sol recipe.
Its lab YAML used a stage-5.sim initialization; **this production config instead
uses stage 4 step 48,000**, following the intended stage-4 → Sol-distillation lineage.
This initialization change is intentional, not an assertion that the lab YAML
originally used stage 4.

### Sampling, recursive mining, and consolidation

The general recipe is `configs/self_distill/iteration/`:

| Config | Role |
|---|---|
| `recipe.yaml` | Allocated human/puzzle sources, parent checkpoint/export, source dataset, game count, training config |
| `human.yaml` | Five positions per game from the allocated PGN source |
| `puzzles.yaml` | 35,000 puzzles with the established eight rating buckets |
| `play.yaml` | 2,000 batched games against log-uniform Stockfish 100–100k; scoring/subsampling parameters |
| `mine.yaml` | 100k-node oracle, five recursions, one-clean-ply gate, temperature 0.6, 4096 output tokens |
| `rebalance.yaml` | Deduplicate accepted records and repartition into four consolidation shards |
| `consolidate.yaml` | Qwen3.8-27B; 8192 output tokens, 16384 context, material hints off; existing boundary instructions unchanged |
| `packing.yaml` | Filter/deduplicate, rewrite numerical evaluations, translate to POV, hold out 100, pack Arrow |

The gameplay policy keeps >=5pp mistakes and samples ordinary positions according
to game length, with 4x downsampling of high-evaluation positions.
The human source has no total-position cap: choose its size when allocating sources.
Sources must be disjoint from benchmark/previous-round data; the launcher does not
invent or register that allocation for you.

```bash
bash scripts/datagen/si_iteration/prepare_iteration.sh
```

This queues the following dependency graph:

```text
export parent → gameplay → score/subsample play ─┐
human/puzzle sampling ──────────────────────────┼→ mine → rebalance → consolidate → pack [→ train]
export parent ─────────────────────────────────┘
```

The individual launchers are `export_iteration.sbatch`, `sample_iteration.sbatch`,
`play_iteration.sbatch`, `score_iteration_play.sbatch`, `mine_iteration.sbatch`,
`rebalance_iteration.sbatch`, `consolidate_iteration.sbatch`, and `pack_iteration.sbatch`, all under
`scripts/datagen/si_iteration/`. They take the recipe directory as their optional first argument.
The submitter derives mining/consolidation array sizes from the YAML task lists;
it does not hardcode a historical distribution between collaborators.
Direct submissions of those two array scripts use the shipped three/four-task defaults.

The merged parent is `runs/self_distill/input_merged`; data is under
`data/self_distill/iteration/{positions,mined,rebalanced,consolidated}`.
Scoring deletes raw gameplay JSONL after durable publication, as implemented by
the existing sampler. Copy raw traces first if they must be retained.
The hybrid uses V2 (`use_v1_vllm: false`). Do not switch V1/V2 mid-resume;
their generated text differs even at the same seed. Qwen uses its own backend.
All launchers use `.venv/bin/python` for data preparation, training, and inference;
set `PYTHON` to override that single interpreter. The one shared
`scripts/environment.sh` supplies repository setup and YAML-reading helpers.
The environment must include compatible training and vLLM inference dependencies.
Representative time limits are resource defaults, not measured
completion guarantees; resubmit the resumable mining/consolidation tasks if needed.

### Training the resulting round

The submitter queues CPU packing after consolidation. Add `--train` as the
second argument to also queue SFT after packing:

```bash
bash scripts/datagen/si_iteration/prepare_iteration.sh configs/self_distill/iteration --train
```

For already consolidated data, pack separately with
`python -m utils.pack_self_distill --config configs/self_distill/iteration/packing.yaml`.
The packer reads published chunks and their accepted-root metadata. It excludes
terminal/malformed outputs, illegal or unsupplied structured moves, duplicates,
validation overlaps, and consolidated evaluations with >=7.5 percentage points
of error. It then applies the stored Stockfish root evaluation: first by numerical
replacement, otherwise by replacing the evaluation sentence, before POV translation.
It does not rescore positions or downsample sources again. Audit JSONLs and an
input snapshot accompany the Arrow directories; an existing output is never overwritten.
Set `validation_arrow` to reuse a prior round's validation split; otherwise 100
filtered positions are held out deterministically. To train an existing pack:

```bash
sbatch scripts/train/si_iteration/train_iteration.sbatch
```

`configs/train/self_distill.yaml` uses the shared representative SFT recipe:
full decoder fine-tuning; four GPUs; batch 4/GPU × accumulation 2 = global 32;
learning rate 1e-4; weight decay 0.1; constant schedule with 5% warmup;
10,000 steps; max length 2048; validation/checkpoints every 250 steps;
100 validation examples. The 2048 limit follows the HCE iteration recipe;
some historical Sol iterations used 1536.

Input datasets are `data/self_distill/iteration/training/{train,validation}.arrow`;
output is `runs/self_distill/iteration`. The launcher checks those dataset
directories exist and resumes the latest checkpoint when resubmitted.

The same round works with an HCE parent: update `source_checkpoint` and
`source_train_dataset` in `recipe.yaml` and `init_from` in the SFT YAML.
The submitter rejects mismatched parent paths. For a subsequent round, copy the
config directory and change **all** input/output namespaces and selected parent
paths; copying the directory alone does not change paths inside YAMLs.
Do not change a started round's semantic settings and resume into its old outputs.

### Evaluation

`eval/benchmark.py` implements the historical tactical/general critical-line
benchmark; `eval/model_ladder.py` implements the batched LM Elo matches.
Export the selected checkpoint first (substitute its path and training dataset):

```bash
.venv/bin/python -m models.vllm.merge_weights \
  --ckpt runs/self_distill/iteration/step_0010000 \
  --out runs/self_distill/iteration/merged \
  --train-dataset data/self_distill/iteration/training/train.arrow \
  --base local/SmolLM3-3B --arch flamingo --save-dtype bf16
sbatch scripts/eval/benchmarks/evaluate_benchmark.sbatch configs/eval/benchmark.yaml
sbatch scripts/eval/ladder/evaluate_ladder.sbatch configs/eval/model_ladder.yaml
```

The benchmark expects `sample.json` in each configured source directory: a list
of rows with `sample_id`, `fen`, `prompt`, `correct_move_uci`, `solution_uci`, and
optional `history`. Both 100- and 1000-position sets use this format. It reports
first-move accuracy, first-move no-mistake rate, STM no-mistake accuracy, exact
prefix/suffix accuracy, and legal-line rate. Stockfish uses 10k nodes; >=10pp
win-rate drop is a mistake. STM accuracy checks only the initial player's moves,
but requires the entire line to be legal. The parser preserves the historical
compact-token move-geometry convention; this is not a prose hallucination audit.
Generation uses temperature 0.6, top-k 20, top-p .95, and 8192 output tokens.
`prompt_mode: sample` preserves neutral benchmark prompts; `training` selects
the structured analysis prompt. Results include generations, per-position scores,
`metrics.json`, and `results.md`. Resume by rerunning the same command/config.
CPU-only rescoring: `python -m eval.benchmark --config configs/eval/benchmark.yaml
--dataset-index 0 --phase score` (omit the line break in your shell).

The ladder plays exactly four games against each of the eight fixed opponents:
two openings (`e4 e5 Nf3 Nc6`, `d4 d5 c4 e6`), each with both colors. It uses the
training prompt, temperature 0.6, 2048 output tokens, history, the historical
legal-move fallback policy, claimable draws, and a 400-ply cap. Ratings fit the
fixed internal ladder anchors in `model_ladder.yaml`, not a separate Lichess
offset. Supply the configured Stockfish/Lc0 binaries and weights (use an Lc0
build compatible with the allocated GPU). All live LM turns are batched.
`state.json`, `model_moves.jsonl`, `games.pgn`, and `results.json` are saved after
each cycle. Resubmitting resumes unfinished games; time-limited stops do not
adjudicate unfinished games as draws. One hour is a resumable leg, not a runtime
guarantee. Both runners explicitly use the historical V1 Flamingo backend,
eager execution, and the generator's disabled prefix cache.

The older native Arrow evaluator is also retained:

`configs/eval/analysis.yaml` specifies a checkpoint, its training config,
tactical/general Arrow inputs, temperature 0.6, and 8192 generated tokens.

```bash
sbatch scripts/eval/native/evaluate_analysis.sbatch
```

The two tasks invoke the existing `eval.py`, saving generations and its native
task-grading results. Prompts are read from the Arrow rows, never changed by the
launcher. Native token-match scores are **not** first-move, no-mistake, exact-prefix,
or line-legality benchmark scores for narrative analyses.

The existing `evaluate_testset.sbatch` and `evaluate_regret.sbatch` remain for
their original formats. In particular, regret parses the legacy “I should play”
recommendation and must not be applied to a different conclusion format.

`eval/ladder.py` remains the separate UCI-only engine-calibration runner.

## Repository layout

```text
├── models/
│   ├── encoder.py                  # LC0 encoder loading
│   ├── flamingo.py, llava.py       # Model architectures
│   └── vllm/                      # Batched inference and checkpoint exports
├── datagen/
│   ├── sample_positions.py        # Human, puzzle, and gameplay sampling
│   ├── build_qa_dataset.py        # Template data construction
│   ├── build_train_all.py         # Task-dataset concatenation
│   ├── mix_arrows.py              # Curriculum mixtures
│   ├── tasks/                     # Task templates
│   ├── tree/                      # Stockfish/HCE data construction
│   ├── sim/                       # Plan/verdict pipeline
│   ├── self_distill/              # Gameplay, recursion, and consolidation logic
│   └── self_distill_stages.py      # Mining/rebalancing/consolidation CLI
├── eval/
│   ├── benchmark.py               # Tactical/general PV benchmarks
│   ├── critical_line.py           # Benchmark line parsing
│   ├── model_ladder.py            # Batched 32-game model Elo evaluation
│   └── ladder.py                  # UCI engine calibration
├── utils/
│   ├── pack_self_distill.py       # Filtering, eval rewriting, POV translation, Arrow
│   └── ...                        # Shared training, token, and board utilities
├── scripts/
│   ├── environment.sh             # Shared repository/venv setup
│   ├── datagen/
│   │   ├── common/                # Generic sampling/building/packing/mixing
│   │   ├── stage1/                # Stage 1 preparation and builder
│   │   ├── stage2/                # Stage 2 preparation, sampling, building, mixing
│   │   ├── stage3/                # Stage 3 preparation, sampling, building, mixing
│   │   ├── stage4/                # Stage 4 preparation, sampling, building, mixing
│   │   ├── hce/                   # HCE preparation, games/puzzles, and packing
│   │   └── si_iteration/          # One self-improvement round through Arrow packing
│   ├── train/
│   │   ├── common/                # Generic SFT launcher
│   │   ├── stage1/                # Flamingo and LLaVA stage 1
│   │   ├── stage2/                # Flamingo and LLaVA stage 2
│   │   ├── stage3/                # Flamingo and LLaVA stage 3
│   │   ├── stage4/                # Flamingo and LLaVA stage 4
│   │   ├── hce/                   # HCE SFT and LLaVA ablations
│   │   ├── sol_distill/           # Initial Sol distillation SFT
│   │   └── si_iteration/          # Iterative SFT
│   └── eval/
│       ├── benchmarks/            # Tactical/general benchmark array
│       ├── ladder/                # Resumable model Elo evaluation
│       └── native/                # Native Arrow/task and regret evaluations
├── configs/                       # Data, training, iteration, and evaluation YAMLs
├── train.py                       # Training entry point
└── eval.py                        # Native Arrow evaluation entry point
```

Sol distillation consumes the prepared dataset described above, so it has a
training launcher but no separate data-generation shell wrapper.
