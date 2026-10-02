#!/usr/bin/env python3
"""Mine recursive self-distillation records or rebalance them for consolidation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datagen.self_distill.consolidation import ConsolidationConfig, Consolidator
from datagen.self_distill.io import atomic_json, load_seeds, rebalance, resolve_jsonls
from datagen.self_distill.recursion import MiningConfig, RecursiveMiner


def _mine_parser(subparsers) -> None:
    parser = subparsers.add_parser("mine", help="recursively mine accepted root/child records")
    parser.add_argument("--input", nargs="+", required=True, metavar="JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--sf-workers", type=int, default=8)
    parser.add_argument("--sf-nodes", type=int, default=100_000)
    parser.add_argument("--max-recursions", type=int, default=5)
    parser.add_argument("--max-root-retries", type=int, default=3)
    parser.add_argument("--max-child-retries", type=int, default=0)
    parser.add_argument("--min-clean-ply", type=int, default=1)
    parser.add_argument("--work-batch-size", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--use-v1-vllm", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the legacy hybrid runner (default: V2)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--inject-drop", type=float, default=0.05)
    parser.add_argument("--mistake-drop", type=float, default=0.10)
    parser.add_argument("--eval-truth-limit", type=float, default=0.10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)


def _rebalance_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rebalance",
        aliases=["--rebalance"],
        help="deterministically repartition accepted mining records",
    )
    parser.add_argument("--input", nargs="+", required=True, metavar="JSONL_OR_RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    sizes = parser.add_mutually_exclusive_group(required=True)
    sizes.add_argument("--shards", type=int)
    sizes.add_argument("--rows-per-shard", type=int)
    parser.add_argument("--overwrite", action="store_true")


def _consolidate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "consolidate",
        help="consolidate accepted child analyses into root supervision",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--material-hints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--max-output-tokens", type=int, default=8_192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    _mine_parser(subparsers)
    _rebalance_parser(subparsers)
    _consolidate_parser(subparsers)
    for child in set(subparsers.choices.values()):
        child.add_argument("--config", type=Path, help="Repository-root-relative task YAML")
        child.add_argument("--start-index", type=int, help="Override YAML start_index (inclusive)")
        child.add_argument("--end-index", type=int, help="Override YAML end_index (exclusive)")
    return result


def run_mine(args: argparse.Namespace) -> None:
    if args.parts <= 0 or not 0 <= args.part < args.parts:
        raise ValueError("part must be in [0, parts)")
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    inputs = resolve_jsonls(args.input)

    # input preparation: normalize legacy rows and select this process's stable partition.
    seeds, input_manifest = load_seeds(
        inputs,
        args.part,
        args.parts,
        args.limit,
        args.seed,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "input_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != input_manifest["identity"]:
            raise RuntimeError("existing input manifest does not match current inputs")
    else:
        atomic_json(manifest_path, input_manifest)

    # mining: process the FIFO until every lineage is accepted or reaches its depth limit.
    # the queue is FIFO, because whenever we recurse into a child, we add it to the end of the queue.
    # this way we can ensure that all logic is batched (processing a position fully would be sequential)
    config = MiningConfig(
        output=output,
        model=args.model.resolve(),
        encoder=args.encoder.resolve(),
        stockfish=args.stockfish.resolve(),
        input_identity=input_manifest["identity"],
        seed=args.seed,
        sf_workers=args.sf_workers,
        sf_nodes=args.sf_nodes,
        max_recursions=args.max_recursions,
        max_root_retries=args.max_root_retries,
        max_child_retries=args.max_child_retries,
        min_clean_ply=args.min_clean_ply,
        work_batch_size=args.work_batch_size,
        checkpoint_every=args.checkpoint_every,
        max_output_tokens=args.max_output_tokens,
        max_num_seqs=args.max_num_seqs,
        use_v1_vllm=args.use_v1_vllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        inject_drop=args.inject_drop,
        mistake_drop=args.mistake_drop,
        eval_truth_limit=args.eval_truth_limit,
        resume=args.resume,
    )
    miner = RecursiveMiner(config)
    try:
        miner.run(seeds)
    finally:
        miner.close()


def run_rebalance(args: argparse.Namespace) -> None:
    inputs = resolve_jsonls(args.input, accepted_name="accepted.jsonl")
    manifest = rebalance(
        inputs,
        args.output.resolve(),
        args.shards,
        args.rows_per_shard,
        args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


def run_consolidate(args: argparse.Namespace) -> None:
    # consolidation: synthesize each accepted root from its three audited children.
    config = ConsolidationConfig(
        input=args.input.resolve(),
        output=args.output.resolve(),
        model=args.model.resolve(),
        material_hints=args.material_hints,
        chunk_size=args.chunk_size,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    Consolidator(config).run()


def main() -> None:
    # Config mode supplies each task's arguments to the same stage parsers.
    dispatch = argparse.ArgumentParser(add_help=False)
    dispatch.add_argument("--config", type=Path)
    dispatch.add_argument("--start-index", type=int)
    dispatch.add_argument("--end-index", type=int)
    dispatch.add_argument("--use-v1-vllm", action=argparse.BooleanOptionalAction, default=None)
    options, remaining = dispatch.parse_known_args()
    if options.config is not None:
        if len(remaining) != 1 or remaining[0] not in {"mine", "rebalance", "consolidate"}:
            dispatch.error("config mode takes a stage, --config, and optional index overrides only")
        from datagen.self_distill.tasks import run_config
        if options.use_v1_vllm is not None and remaining[0] != "mine":
            dispatch.error("--use-v1-vllm applies only to hybrid mining, not Qwen consolidation")
        run_config(options.config, remaining[0], options.start_index, options.end_index,
                   use_v1_vllm=options.use_v1_vllm)
        return
    if options.start_index is not None or options.end_index is not None:
        dispatch.error("index overrides require --config")
    arguments = parser().parse_args()
    if arguments.command == "mine":
        run_mine(arguments)
    elif arguments.command == "consolidate":
        run_consolidate(arguments)
    else:
        run_rebalance(arguments)


if __name__ == "__main__":
    main()
