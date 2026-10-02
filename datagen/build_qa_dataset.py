#!/usr/bin/env python3
"""YAML-driven multi-split builder for QA datasets (stage-1 and stage-2).

Output layout:

    config.output_path/
    ├── val_dataset.arrow/        # HF arrow shards + dataset_config.json
    ├── test_dataset.arrow/
    └── training_datasets/
        ├── piece_on_square.arrow/
        └── ...

Generation order: test → [test_modeA] → val → train. Each split threads a shared
`seen` FEN set so they stay position-disjoint.

Two position-sampling models, selected automatically:

  * Shared-positions ("mode A"): an eval split with `num_positions: N` reads N
    held-out positions and runs `sample_all` for EVERY task on each one — the same
    positions across all tasks. Used by stage-1's exhaustive test set. `mix` tasks
    cannot run here and are rejected by config validation.
  * Per-task: train, and any eval split with `per_task: K`, build each task's
    positions independently via `_collect_positions`. A task may carry a `mix`:
        mix: [{source, frac}, ...]  deterministically interleave sources by their
                                    fractions (smooth weighted round-robin). A
                                    source is a jsonl path/glob or 'shards' (the
                                    split's pool); when a source exhausts it drops
                                    out and the survivors keep their relative
                                    shares. All streams share one dedup set.
    Otherwise it falls back to a plain file-order pass of the pool (deduped across
    tasks, so each task gets a disjoint run).

Usage:

    python -m datagen.build_qa_dataset \\
        --config configs/sample_instances/stage1.yaml \\
        [--seed N] [--pov | --no-pov] [--output-path PATH] \\
        [--override KEY.PATH=VALUE ...]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

import chess
import yaml
from datasets import Dataset
from tqdm import tqdm

from datagen.tasks import TASKS
from utils.board_representation import BoardRepr


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _coerce(s: str):
    """Coerce a string to int → float → bool → str (in that order)."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _apply_override(cfg: dict, token: str) -> None:
    """Walk a dot-path into `cfg` and assign the (type-coerced) value."""
    key, sep, raw = token.partition("=")
    if not sep:
        raise ValueError(f"--override token missing '=': {token!r}")
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = _coerce(raw)


def _resolve_config(args) -> dict:
    """Load YAML, apply overrides, then layer in explicit CLI scalars."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    for token in args.override or []:
        _apply_override(cfg, token)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.pov is not None:
        cfg["pov"] = args.pov
    if args.output_path is not None:
        cfg["output_path"] = args.output_path

    return cfg


def _validate_config(cfg: dict) -> None:
    for key in ("seed", "pov", "output_path", "tasks", "splits"):
        if key not in cfg:
            raise ValueError(f"config missing required field: {key!r}")
    for task, spec in cfg["tasks"].items():
        if task not in TASKS:
            raise ValueError(f"unknown task in config: {task!r}; known: {sorted(TASKS)}")
        for k in ("num_positions", "num_questions"):
            if k not in spec:
                raise ValueError(f"tasks.{task} missing required field: {k!r}")
    for split in ("train", "val", "test"):
        if split not in cfg["splits"]:
            raise ValueError(f"splits missing required entry: {split!r}")
        if "shards" not in cfg["splits"][split]:
            raise ValueError(f"splits.{split} missing required field: 'shards'")
    # eval splits: exactly one of num_positions (mode A) / per_task, and mode A
    # (shared positions across all tasks) is incompatible with `mix` tasks.
    shaped = [t for t, sp in cfg["tasks"].items() if sp.get("mix")]
    for t in shaped:
        for m in cfg["tasks"][t]["mix"]:
            if "source" not in m or not isinstance(m.get("frac"), (int, float)):
                raise ValueError(
                    f"tasks.{t}.mix entries need 'source' and a numeric 'frac' (no 'max'): {m!r}")
    for split in [s for s in ("val", "test", "test_modeA") if s in cfg["splits"]]:
        s = cfg["splits"][split]
        has_n = "num_positions" in s
        has_pt = "per_task" in s
        if has_n == has_pt:
            raise ValueError(
                f"splits.{split} must specify exactly one of "
                f"'num_positions' or 'per_task' (got num_positions={has_n}, per_task={has_pt})")
        if has_pt and "num_questions" not in s:
            raise ValueError(f"splits.{split} with 'per_task' must also specify 'num_questions'")
        if has_n and shaped:
            raise ValueError(
                f"splits.{split} uses num_positions (mode-A shared positions), which runs every "
                f"task on the same positions; `mix` tasks can't run there: {shaped}. "
                f"Use 'per_task' for those.")
    # PARALLEL_ANALYZE tasks (stage-5: one get_tree analysis per position via the search
    # pipeline) have no sample_all, so they can't run in a mode-A (num_positions) split.
    specialized_tasks = [
        t for t in cfg["tasks"]
        if (getattr(TASKS[t], "PARALLEL_ANALYZE", False)
            or getattr(TASKS[t], "MINED_RECORDS", False))
    ]
    if specialized_tasks:
        for split in [s for s in ("val", "test", "test_modeA") if s in cfg["splits"]]:
            if "num_positions" in cfg["splits"][split]:
                raise ValueError(f"specialized tasks {specialized_tasks} can't run in mode-A "
                                 f"(num_positions) split {split!r}; use per_task")
    mined_tasks = [t for t in cfg["tasks"]
                   if getattr(TASKS[t], "MINED_RECORDS", False)]
    if mined_tasks:
        if not cfg["pov"]:
            raise ValueError(f"mined motif tasks require pov=true: {mined_tasks}")
        if cfg.get("lookahead"):
            raise ValueError(
                "mined motif records choose their own 0--8-ply lookahead; "
                "do not set the global lookahead flag"
            )
        for task in mined_tasks:
            if cfg["tasks"][task].get("mix"):
                raise ValueError(f"mined motif task {task!r} does not support mix sources")
            fixed_lookahead = cfg["tasks"][task].get("lookahead_plies")
            if (fixed_lookahead is not None
                    and (type(fixed_lookahead) is not int
                         or not 0 <= fixed_lookahead <= 8)):
                raise ValueError(
                    f"mined motif task {task!r} lookahead_plies must be an "
                    "integer in [0, 8]"
                )
            groups = cfg["tasks"][task].get("source_groups")
            if groups:
                total = 0
                for group in groups:
                    if "shards" not in group or "num_positions" not in group:
                        raise ValueError(
                            f"tasks.{task}.source_groups entries need shards and "
                            f"num_positions: {group!r}"
                        )
                    count = int(group["num_positions"])
                    if count <= 0:
                        raise ValueError(
                            f"tasks.{task}.source_groups num_positions must be positive"
                        )
                    total += count
                if total != int(cfg["tasks"][task]["num_positions"]):
                    raise ValueError(
                        f"tasks.{task}.source_groups request {total} positions but "
                        f"num_positions={cfg['tasks'][task]['num_positions']}"
                    )
    for split_name, split in cfg["splits"].items():
        if split.get("stratify_motifs") and len(mined_tasks) != len(cfg["tasks"]):
            raise ValueError(
                f"splits.{split_name}.stratify_motifs requires exclusively mined motif tasks"
            )
    # lookahead (stage-3/4 forward) builds: every task must implement the forward
    # sampler, and mode-A (shared-positions) splits are unsupported.
    if cfg.get("lookahead"):
        missing = [t for t in cfg["tasks"] if not hasattr(TASKS[t], "sample_n_forward")]
        if missing:
            raise ValueError(f"lookahead build needs sample_n_forward on every task; missing: {missing}")
        bad = [s for s in ("val", "test", "test_modeA")
               if s in cfg["splits"] and "num_positions" in cfg["splits"][s]]
        if bad:
            raise ValueError(f"lookahead is incompatible with mode-A (num_positions) splits: {bad}; use per_task")


def _resolve_shards(patterns) -> list[str]:
    """Expand a list of file paths / globs into a sorted unique list of paths."""
    if isinstance(patterns, str):
        patterns = [patterns]
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob(str(pattern)))
        if not matched:
            raise ValueError(f"no files matched shard pattern: {pattern!r}")
        paths.extend(matched)
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Shard streaming
# ---------------------------------------------------------------------------

def _load_positions(paths, lookahead: bool = False):
    """Stream position records as a uniform 4-tuple
    (present_fen, history_moves, dedup_fen, move_sequence) — so the rest of the
    pipeline is identical for normal and lookahead data:

      * normal (3-field line [start_fen, moves, end_fen]): the model is shown
        end_fen        -> (start_fen, moves, end_fen, None).
      * lookahead (5-field line [start_fen, history, q_fen, seq, final_fen]): the
        model is shown Q and must play `seq`
                       -> (start_fen, history, q_fen, seq).
        final_fen is dropped (tasks recompute the final board while verbalising the
        sequence). move_sequence is a UCI list iff lookahead — which is the only
        thing that switches which task sampler is called downstream.
      * simulate pool (4-field line [start_fen, moves, end_fen, metadata dict],
        stage 5.b): the metadata rides slot [3]; only analysis tasks with their
        own `analyze` read it (a dict is never mistaken for a move sequence)."""
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if lookahead:
                    start_fen, history, present_fen, seq, _final_fen = rec
                    yield start_fen, history, present_fen, seq
                else:
                    start_fen, moves, end_fen = rec[:3]
                    yield (start_fen, moves, end_fen,
                           rec[3] if len(rec) > 3 else None)


def _stream_filtered(paths, seen_fens, lookahead: bool = False):
    """Yield records from paths, skipping those whose presented FEN (index [2]) is
    in seen_fens. seen_fens is read-only here. (Mode-A only.)"""
    for rec in _load_positions(paths, lookahead):
        if rec[2] in seen_fens:
            continue
        yield rec


# ---------------------------------------------------------------------------
# Per-task position collection (mix sources, else a plain pool pass)
# ---------------------------------------------------------------------------

def _source_iter(source: str, pool_shards: list[str], split_token: str,
                 lookahead: bool = False):
    """Iterator of position records for a `mix` source spec: the literal 'shards' =
    the split's pool; otherwise a jsonl path/glob, where a `{split}` placeholder is
    filled with 'train' (train split) or 'val-test' (eval splits) so the source
    reads the shard set matching the split being built."""
    if source == "shards":
        return _load_positions(pool_shards, lookahead)
    return _load_positions(_resolve_shards(source.replace("{split}", split_token)), lookahead)


def _weighted_interleave(streams, seen_fens: set, local_seen: set):
    """Deterministic smooth weighted round-robin over (iterator, weight) pairs, and
    the single dedup point for collection. Each step credits every active stream by
    its weight, draws the next end-FEN-fresh triple from the highest-credit one
    (skipping end-FENs already in seen_fens/local_seen, which is shared across all
    streams so dedup spans them), marks it seen, and debits the stream by the active
    total. A stream with no fresh triples left is dropped, so survivors keep their
    relative proportions. One stream at weight 1.0 == a plain deduped pool pass."""
    pool = [{"it": it, "w": float(w), "credit": 0.0} for it, w in streams]
    while pool:
        total = sum(s["w"] for s in pool)
        for s in pool:
            s["credit"] += s["w"]
        s = max(pool, key=lambda d: d["credit"])
        s["credit"] -= total
        fresh = next((t for t in s["it"]
                      if t[2] not in seen_fens and t[2] not in local_seen), None)
        if fresh is None:
            pool.remove(s)
            continue
        local_seen.add(fresh[2])
        yield fresh


def _collect_positions(spec: dict, pool_shards: list[str], rng: random.Random,
                       seen_fens: set, n: int, label: str,
                       split_token: str = "train", lookahead: bool = False) -> tuple[list, set]:
    """Return (records, new_fens). Both the multi-source `mix` case and the plain
    single-pool case feed the same interleaver (the lone dedup point): a `mix` task
    interleaves its explicit sources by their fractions, while a non-mix task is just
    one stream at weight 1.0 (a deduped file-order pass, then shuffled to match the
    historical stage-1 order). `split_token` fills the `{split}` placeholder in mix
    source globs. Single pass per source (no cycling); shortfalls warn. Records are
    the uniform 4-tuples from `_load_positions` (lookahead changes only their read)."""
    local_seen: set = set()
    is_mix = bool(spec.get("mix"))
    if is_mix:
        streams = [(_source_iter(m["source"], pool_shards, split_token, lookahead), m["frac"])
                   for m in spec["mix"]]
    else:
        streams = [(_load_positions(pool_shards, lookahead), 1.0)]

    result: list = []
    for rec in _weighted_interleave(streams, seen_fens, local_seen):
        result.append(rec)
        if len(result) >= n:
            break
    if len(result) < n:
        print(f"  WARN: {label}: {'mix' if is_mix else 'base'} yielded {len(result)}/{n}")
    if not is_mix:
        rng.shuffle(result)   # preserve historical stage-1 file-order-then-shuffle
    return result, local_seen


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def _build_history(start_fen: str, moves: list[str]) -> list[str]:
    if not moves:
        return []
    board = chess.Board(start_fen)
    out = [board.fen()]
    for uci in moves[:-1]:
        board.push_uci(uci)
        out.append(board.fen())
    return out


def _record(s: dict, end_fen: str, history: list[str]) -> dict:
    return {
        "fen":      end_fen,
        "history":  history,
        "prompt":   s["question"],
        "response": s["answer"],
        "extra": {
            "task":         s["question_type"],
            "answer_class": s["answer_class"],
        },
    }


def _bump_frequency(frequency: dict, s: dict) -> None:
    """Bump the per-token counter that shapes the next `_choose_entity` call.

    `answer_class` is None for tasks whose `_choose_entity` doesn't read the
    counter (1-question-per-position tasks and the line tasks that pick
    length-proportionally or uniformly); skip cleanly in that case.
    """
    ac = s["answer_class"]
    if ac is None:
        return
    for cls in ac:
        frequency[cls] += 1


def _records_for_analysis_task(module, spec: dict, pool_shards: list[str], rng: random.Random,
                               seen_fens: set, n: int, label: str, split_token: str,
                               lookahead: bool) -> tuple[list, set, dict]:
    """Stage-5: the answer is computed per position rather than sampled per-board — the
    machine-readable HCE analysis (get_tree(...).string("token"), via datagen.tree.parallelize)
    unless the module carries its own `analyze(triples)` (stage-5.a verdict_eval, whose
    classification needs the setup move from the triple). One record per position; `prompt`
    comes from the task module."""
    from datagen.tree import parallelize
    triples, new_fens = _collect_positions(spec, pool_shards, rng, seen_fens, n,
                                           f"{label}:{module.NAME}", split_token, lookahead)
    print(f"  {label}:{module.NAME}: analyzing {len(triples)} positions "
          f"across {parallelize.cpu_budget()} workers...")
    if hasattr(module, "analyze"):
        tokens = module.analyze(triples)
    else:
        tokens = parallelize.analyze([present_fen for (_s, _m, present_fen, _seq) in triples])
    records: list = []
    for (start_fen, moves, present_fen, _seq), token in zip(triples, tokens):
        if token is None:
            continue                                # analysis failed for this position — skip
        records.append({
            "fen":      present_fen,
            "history":  _build_history(start_fen, moves),
            "prompt":   module.PROMPT,
            "response": token,
            "extra":    {"task": module.NAME, "answer_class": None},
        })
    summary = {"positions": len(triples), "records": len(records),
               "failed": len(triples) - len(records)}
    return records, new_fens, summary


def _records_for_mined_task(module, spec: dict, pool_shards: list[str],
                            rng: random.Random, seen_fens: set, n: int,
                            n_q: int, label: str,
                            stratify_motifs: bool = False) -> tuple[list, set, dict]:
    """Render pre-mined motif dictionaries with a fresh 0--8-ply lookahead.

    Ordinary game rows require the full 15-ply context. For sampled N, the last
    N+7 moves are replayed: seven become LC0 history, and the remaining N are
    verbalized in the prompt. Puzzle rows retain only their genuine setup ply;
    those use N=0 and never synthesize unavailable game history.

    Training may define exact ``source_groups`` in the task spec. This preserves
    the established one-million-row game sample while appending every qualifying
    puzzle row instead of letting the larger game glob consume the whole budget.
    Eval splits may request deterministic multi-label motif stratification. A
    task-level ``lookahead_plies`` override fixes N instead of sampling it.
    """
    from datagen.sample_positions import motif_prompt_window

    if n_q != 1:
        raise ValueError(f"{module.NAME} requires num_questions=1")
    def motif_names(record: dict) -> tuple[str, ...]:
        return tuple(dict.fromkeys(motif["motif"] for motif in record["motifs"]))

    def is_puzzle(record: dict) -> bool:
        return record.get("source", {}).get("kind") == "lichess_puzzle"

    def eligible(record: dict, required_source: str | None = None) -> bool:
        if not is_puzzle(record) and len(record["moves"]) < 15:
            return False
        if required_source is not None and not any(
            motif.get("metadata", {}).get("source") == required_source
            for motif in record["motifs"]
        ):
            return False
        return True

    def stream(paths: list[str], required_source: str | None = None):
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    if eligible(record, required_source):
                        yield record

    selected: list[dict] = []
    local_seen = set()
    skipped_short = 0
    source_group_counts = []

    if stratify_motifs:
        # Keep a bounded deterministic reservoir for each label, then greedily
        # draw examples containing the currently least-represented motif. A row
        # contributes to every motif it contains, so this is true multi-label
        # balancing rather than one arbitrarily assigned class per position.
        reservoirs: dict[str, list[dict]] = defaultdict(list)
        observed: dict[str, int] = defaultdict(int)
        reservoir_size = max(256, n)
        for record in stream(pool_shards):
            target_fen = record["fen"]
            if target_fen in seen_fens:
                continue
            for motif in motif_names(record):
                observed[motif] += 1
                bucket = reservoirs[motif]
                if len(bucket) < reservoir_size:
                    bucket.append(record)
                else:
                    replacement = rng.randrange(observed[motif])
                    if replacement < reservoir_size:
                        bucket[replacement] = record

        candidates = {
            record["fen"]: record
            for bucket in reservoirs.values()
            for record in bucket
        }
        candidate_names = {
            fen: motif_names(record) for fen, record in candidates.items()
        }
        coverage: dict[str, int] = defaultdict(int)
        motif_order = {
            motif: index for index, motif in enumerate(module.MOTIF_ORDER)
        }
        while len(selected) < n:
            available = [
                motif for motif, bucket in reservoirs.items()
                if any(record["fen"] not in local_seen for record in bucket)
            ]
            if not available:
                break
            motif = min(
                available,
                key=lambda name: (coverage[name], motif_order.get(name, 10**9), name),
            )
            choices = {
                record["fen"]: record for record in reservoirs[motif]
                if record["fen"] not in local_seen
            }
            chosen = min(
                choices.values(),
                key=lambda record: (
                    sum(coverage[name] for name in candidate_names[record["fen"]]),
                    -len(candidate_names[record["fen"]]),
                    record["fen"],
                ),
            )
            selected.append(chosen)
            local_seen.add(chosen["fen"])
            for name in candidate_names[chosen["fen"]]:
                coverage[name] += 1
        source_group_counts.append({
            "source": "split_shards",
            "requested": n,
            "selected": len(selected),
            "available_per_motif": dict(sorted(observed.items())),
        })
    elif label == "train" and spec.get("source_groups"):
        for group in spec["source_groups"]:
            quota = int(group["num_positions"])
            paths = _resolve_shards(group["shards"])
            required_source = group.get("require_motif_source")
            group_selected = 0
            for record in stream(paths, required_source):
                target_fen = record["fen"]
                if target_fen in seen_fens or target_fen in local_seen:
                    continue
                selected.append(record)
                local_seen.add(target_fen)
                group_selected += 1
                if group_selected >= quota:
                    break
            source_group_counts.append({
                "shards": group["shards"],
                "require_motif_source": required_source,
                "requested": quota,
                "selected": group_selected,
            })
    else:
        for path in pool_shards:
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    target_fen = record["fen"]
                    if target_fen in seen_fens or target_fen in local_seen:
                        continue
                    if not eligible(record):
                        skipped_short += 1
                        continue
                    selected.append(record)
                    local_seen.add(target_fen)
                    if len(selected) >= n:
                        break
            if len(selected) >= n:
                break
    if len(selected) < n:
        print(f"  WARN: {label}:{module.NAME} yielded {len(selected)}/{n}")
    rng.shuffle(selected)

    records = []
    prompt_fens = set()
    skipped_prompt_collisions = 0
    lookahead_counts = defaultdict(int)
    motif_counts = defaultdict(int)
    for record in tqdm(selected, desc=f"{label}:{module.NAME}", unit="pos"):
        # The mined target FEN and the earlier FEN presented to the model are
        # different whenever N > 0. Reserve both across splits. Start from a
        # uniformly sampled N, but try the other lookaheads if that prompt FEN
        # is already reserved; this keeps the requested split size without
        # allowing prompt-position leakage.
        fixed_lookahead = spec.get("lookahead_plies")
        prompt_plies_choices = (
            [0] if is_puzzle(record)
            else [fixed_lookahead] if fixed_lookahead is not None
            else rng.sample(range(9), 9)
        )
        window = None
        for candidate_plies in prompt_plies_choices:
            candidate_window = motif_prompt_window(record, candidate_plies)
            candidate_prompt_fen = candidate_window[2]
            if (candidate_prompt_fen not in seen_fens
                    and candidate_prompt_fen not in prompt_fens):
                prompt_plies = candidate_plies
                window = candidate_window
                break
        if window is None:
            skipped_prompt_collisions += 1
            continue
        history_start_fen, history_moves, prompt_fen, prompt_moves, target_fen = window
        expected_history = len(record["moves"]) if is_puzzle(record) else 7
        if len(history_moves) != expected_history or len(prompt_moves) != prompt_plies:
            raise ValueError("motif context did not yield the expected history/prompt split")
        sample = module.render_mined_record(
            record, prompt_fen, prompt_moves, rng
        )
        history = _build_history(history_start_fen, history_moves)
        if len(history) != expected_history:
            raise ValueError("motif Arrow row carries the wrong LC0 history length")
        records.append(_record(sample, prompt_fen, history))
        prompt_fens.add(prompt_fen)
        lookahead_counts[prompt_plies] += 1
        for motif in motif_names(record):
            motif_counts[motif] += 1

        replay = chess.Board(prompt_fen)
        for move in prompt_moves:
            replay.push_uci(move)
        if replay.fen() != target_fen:
            raise ValueError("motif prompt continuation does not reach target FEN")

    return records, local_seen | prompt_fens, {
        "positions": len(records),
        "selected_positions": len(selected),
        "records": len(records),
        "skipped_short_context": skipped_short,
        "skipped_prompt_collisions": skipped_prompt_collisions,
        "lookahead_counts": dict(sorted(lookahead_counts.items())),
        "motif_counts": dict(sorted(motif_counts.items())),
        "source_groups": source_group_counts,
    }


def _records_for_task(task: str, spec: dict, pool_shards: list[str], pov: bool,
                      rng: random.Random, seen_fens: set, n: int, n_q: int,
                      label: str, split_token: str = "train",
                      lookahead: bool = False,
                      stratify_motifs: bool = False) -> tuple[list, set, dict]:
    """Collect a task's positions (shaping-aware) and render `n_q` questions each.
    With lookahead, the presented board is Q (the record's index-[2] FEN) and the
    task answers on the board reached after the record's move sequence."""
    module = TASKS[task]
    if getattr(module, "MINED_RECORDS", False):
        return _records_for_mined_task(
            module, spec, pool_shards, rng, seen_fens, n, n_q, label,
            stratify_motifs=stratify_motifs,
        )
    if getattr(module, "PARALLEL_ANALYZE", False):  # stage-5: answer via the search pipeline
        return _records_for_analysis_task(module, spec, pool_shards, rng, seen_fens, n,
                                          label, split_token, lookahead)
    max_q = module.MAX_UNIQUE_QUERIES
    capped = n_q > max_q
    if capped:
        print(f"  WARN: {label}: task {task} num_questions={n_q} > MAX_UNIQUE_QUERIES={max_q}; capping to {max_q}")
    n_q = min(n_q, max_q)
    triples, new_fens = _collect_positions(spec, pool_shards, rng, seen_fens, n,
                                           f"{label}:{task}", split_token, lookahead)
    freq: dict = defaultdict(int)
    records: list = []
    for start_fen, moves, present_fen, seq in tqdm(triples, desc=f"{label}:{task}", unit="pos"):
        board = BoardRepr.from_fen(present_fen, pov=pov)
        history = _build_history(start_fen, moves)
        samples = (module.sample_n_forward(board, [chess.Move.from_uci(u) for u in seq],
                                           freq, rng, n_q)
                   if isinstance(seq, list) else module.sample_n(board, freq, rng, n_q))
        for s in samples:
            _bump_frequency(freq, s)
            records.append(_record(s, present_fen, history))
    summary = {
        "positions": len(triples),
        "records": len(records),
        "capped_num_questions": capped,
        "answer_class_freq": dict(sorted(freq.items(), key=lambda kv: -kv[1])[:40]),
    }
    return records, new_fens, summary


# ---------------------------------------------------------------------------
# Eval split builds (test / val / test_modeA)
# ---------------------------------------------------------------------------

def _build_eval_split(label: str, split_cfg: dict, tasks_cfg: dict,
                      pov: bool, rng: random.Random, seen_fens: set,
                      lookahead: bool = False) -> tuple[list, set, dict]:
    """Build records for one eval split. `num_positions` -> shared-positions mode A
    (sample_all per task on each of N positions); `per_task` -> per-task shaped
    collection. Returns (records, new_fens, summary). (Mode A is incompatible with
    lookahead — config validation rejects that combination.)"""
    shards = _resolve_shards(split_cfg["shards"])
    task_names = list(tasks_cfg)
    records: list = []
    new_fens: set = set()
    per_task_counts: dict = defaultdict(int)

    if "num_positions" in split_cfg:
        # Mode A: read N unique positions; for each, run sample_all per task.
        n = split_cfg["num_positions"]
        freq = {t: defaultdict(int) for t in task_names}
        pbar = tqdm(total=n, desc=f"{label} (mode=num_positions)", unit="pos")
        for start_fen, moves, end_fen, _seq in _stream_filtered(shards, seen_fens | new_fens, lookahead):
            if len(new_fens) >= n:
                break
            board = BoardRepr.from_fen(end_fen, pov=pov)
            history = _build_history(start_fen, moves)
            for task in task_names:
                recs = TASKS[task].sample_all(board, freq[task], rng)
                for s in recs:
                    _bump_frequency(freq[task], s)
                    records.append(_record(s, end_fen, history))
                per_task_counts[task] += len(recs)
            new_fens.add(end_fen)
            pbar.update(1)
        pbar.close()
        summary = {
            "mode":             "num_positions",
            "positions":        len(new_fens),
            "records":          len(records),
            "per_task_records": dict(per_task_counts),
        }
    else:
        # Per-task (shaping-aware): each task collects its own positions, threading
        # a cumulative seen-FEN set so tasks stay disjoint within the split.
        k = split_cfg["per_task"]
        q = split_cfg["num_questions"]
        per_task: dict = {}
        for task, spec in tasks_cfg.items():
            recs, fens, summ = _records_for_task(
                task, spec, shards, pov, rng, seen_fens | new_fens, k, q, label,
                split_token="val-test", lookahead=lookahead,
                stratify_motifs=bool(split_cfg.get("stratify_motifs", False)))
            records.extend(recs)
            new_fens |= fens
            per_task[task] = summ
        summary = {
            "mode":            "per_task",
            "per_task":        k,
            "num_questions":   q,
            "records":         len(records),
            "per_task":        per_task,
        }
    return records, new_fens, summary


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_dataset(records: list[dict], path: Path) -> None:
    if records:
        Dataset.from_list(records).save_to_disk(str(path))
    else:
        Dataset.from_dict({"fen": [], "history": [], "prompt": [], "response": [], "extra": []}).save_to_disk(str(path))


def _save_arrow_config(arrow_dir: Path, payload: dict) -> None:
    with open(arrow_dir / "dataset_config.json", "w") as f:
        json.dump(payload, f, indent=2)


def _emit_eval_split(split_name: str, arrow_name: str, split_cfg: dict, tasks_cfg: dict,
                     output_path: Path, pov: bool, seed: int, rng: random.Random,
                     seen_fens: set, lookahead: bool = False) -> set:
    """Build one eval split, write its arrow + dataset_config, return its FENs."""
    records, fens, summary = _build_eval_split(split_name, split_cfg, tasks_cfg, pov, rng,
                                               seen_fens, lookahead)
    out_dir = output_path / arrow_name
    _save_dataset(records, out_dir)
    _save_arrow_config(out_dir, {
        "pov":           pov,
        "seed":          seed,
        "split":         split_name,
        "tasks":         list(tasks_cfg),
        "split_spec":    split_cfg,
        "summary_stats": summary,
    })
    print(f"      → {len(records)} records across {len(fens)} positions  [{arrow_name}]")
    return fens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _pack_stage5(shards_dir: str, output_path: Path, cfg: dict) -> None:
    """Stage-5 alternative to computing: pack the shard jsonls produced by the sharded runner
    (datagen/tree/parallelize.py run_shard -> shard_*.jsonl) into the hce_rationale arrow, split by
    global idx into test/val/train. Works on an IN-PROGRESS array — prints a warning and packs the
    current partial set (re-run when complete to overwrite)."""
    from datagen.tree import parallelize
    shard_glob = str(Path(shards_dir) / "shard_*.jsonl")
    files = sorted(glob(shard_glob))
    if not files:
        raise ValueError(f"no stage-5 shard jsonls under {shards_dir!r}")
    counts = [sum(1 for _ in open(f)) for f in files]
    total, target, nshards = sum(counts), 17362, 72
    at_target = sum(1 for c in counts if c >= target)
    if len(files) < nshards or at_target < nshards:
        print(f"  WARNING: stage-5 shards INCOMPLETE — {len(files)}/{nshards} shard files, "
              f"{at_target}/{nshards} at target ({target}); {total:,} examples so far. Packing the "
              f"current PARTIAL dataset (re-run this once the array completes to overwrite with the full set).")
    else:
        print(f"  stage-5 shards complete: {total:,} examples across {len(files)} shards.")
    test_n = cfg["splits"].get("test", {}).get("per_task", 500)
    val_n  = cfg["splits"].get("val", {}).get("per_task", 200)
    output_path.mkdir(parents=True, exist_ok=True)
    parallelize.pack(shard_glob, str(output_path), test_n, val_n, bool(cfg["pov"]))


def parse_args():
    p = argparse.ArgumentParser(description="YAML-driven QA dataset builder (stage-1 + stage-2).")
    p.add_argument("--config",      required=True, type=Path)
    p.add_argument("--seed",        type=int,        default=None)
    p.add_argument("--pov",         dest="pov", action="store_true",  default=None)
    p.add_argument("--no-pov",      dest="pov", action="store_false", default=None)
    p.add_argument("--output-path", type=str,        default=None)
    p.add_argument("--override",    nargs="*",       default=None,
                   help="Dotted KEY.PATH=VALUE overrides for nested YAML fields.")
    p.add_argument("--stage5-pack", action="store_true",
                   help="Stage 5: pack existing shard jsonls into arrow (no compute); tolerates in-progress shards.")
    p.add_argument("--stage5sim-pack", action="store_true",
                   help="Stage 5.sim: pack datagen.sim.parallelize shard jsonls into arrow (no compute).")
    p.add_argument("--shards-dir",  type=str, default=None,
                   help="Stage-5 shard jsonl dir for --stage5-pack (default: <output_path>/shards).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = _resolve_config(args)
    _validate_config(cfg)

    pov         = bool(cfg["pov"])
    seed        = int(cfg["seed"])
    output_path = Path(cfg["output_path"])
    tasks_cfg   = cfg["tasks"]
    lookahead   = bool(cfg.get("lookahead", False))

    if args.stage5_pack:                    # jsonl -> arrow instead of the (fold-in) compute path
        _pack_stage5(args.shards_dir or str(output_path / "shards"), output_path, cfg)
        return

    if args.stage5sim_pack:                 # stage-5.sim: sharded GPU runner's jsonls -> arrow
        from datagen.sim import parallelize as sim_par
        sim_par.pack(args.shards_dir or str(output_path / "shards"), str(output_path),
                     cfg["splits"].get("test", {}).get("per_task", 500),
                     cfg["splits"].get("val", {}).get("per_task", 200))
        return

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_datasets").mkdir(parents=True, exist_ok=True)

    # Independent RNGs per split so re-running with a different val budget
    # doesn't shift the train sequence.
    rng_test   = random.Random(seed + 1)
    rng_val    = random.Random(seed + 2)
    rng_train  = random.Random(seed + 3)
    rng_test_a = random.Random(seed + 4)

    seen_fens: set = set()
    build_started = time.time()

    print("[1/3] Building test split...")
    seen_fens |= _emit_eval_split("test", "test_dataset.arrow", cfg["splits"]["test"],
                                  tasks_cfg, output_path, pov, seed, rng_test, seen_fens, lookahead)

    if "test_modeA" in cfg["splits"]:
        print("[1b/3] Building Mode A (exhaustive) test split...")
        seen_fens |= _emit_eval_split("test", "test_dataset_modeA.arrow", cfg["splits"]["test_modeA"],
                                      tasks_cfg, output_path, pov, seed, rng_test_a, seen_fens, lookahead)

    print("[2/3] Building val split...")
    seen_fens |= _emit_eval_split("val", "val_dataset.arrow", cfg["splits"]["val"],
                                  tasks_cfg, output_path, pov, seed, rng_val, seen_fens, lookahead)

    print("[3/3] Building train split (per-task)...")
    train_shards = _resolve_shards(cfg["splits"]["train"]["shards"])
    train_seen: set = set()
    total_records = 0
    for task, spec in tasks_cfg.items():
        recs, fens, summ = _records_for_task(
            task, spec, train_shards, pov, rng_train, seen_fens | train_seen,
            spec["num_positions"], spec["num_questions"], "train", lookahead=lookahead)
        train_seen |= fens
        total_records += len(recs)
        task_dir = output_path / "training_datasets" / f"{task}.arrow"
        _save_dataset(recs, task_dir)
        _save_arrow_config(task_dir, {
            "pov":           pov,
            "seed":          seed,
            "split":         "train",
            "task":          task,
            "task_spec":     spec,
            "train_shards":  cfg["splits"]["train"]["shards"],
            "summary_stats": summ,
        })
        print(f"      → {task}: {len(recs)} records across {summ['positions']} positions")
    print(f"      total train records: {total_records}")

    build_finished = time.time()
    print(f"Done in {build_finished - build_started:.1f}s. Output: {output_path}")


if __name__ == "__main__":
    main()
