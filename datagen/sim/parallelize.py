"""Stage-5.sim datagen at scale: one shard of positions per GPU job.

Positions come from a triples pool (sample_positions.py output,
[start_fen, moves, end_fen] per line). The pool has already sampled a ply from
each source game and retained at most seven history moves. We reject terminal
positions, give every accepted position a global index in file order, and job
i takes those with `index % num_shards == i` — disjoint and reproducible with
no coordination between jobs. Positions in check and positions with fewer
than 16 pieces remain eligible.

Per position: the plan-model alpha-beta (datagen.sim.search.Builder — union
deepening, forcing, verdict leaves) and the token-mode narrative
(datagen.sim.flatten). Generation is batched across positions: K positions
run in their own threads and every model call parks on the Batcher, whose
single server thread drains same-kind batches into the two vLLM engines
(datagen.sim.sampler); Stockfish goes through the shared pools
(datagen.sim.sfpool). Each finished record is appended to the shard's jsonl
and flushed, and startup reads back the indices already present, so a killed
job resumes. `pack()` assembles the shard jsonls into the arrow datasets
(test/val/train split by global index, position-disjoint by construction).

    python -m datagen.sim.parallelize run --shard 0 --num-shards 24 \\
        --pool "data/lichess/stage5/train-*.jsonl" --out data/stage5/sim/shards
    python -m datagen.sim.parallelize pack --shards data/stage5/sim/shards \\
        --out data/stage5/sim
"""
import argparse
import json
import threading
import time
import traceback
from glob import glob
from pathlib import Path

import chess

PROMPT = "Analyze the position carefully and find the best move for player."
TASK = "sim_rationale"


def stream_pool(pool_glob, total=0):
    """Accepted ``(global_idx, triple)`` records in fixed file order.

    ``data/lichess/stage5`` is already a sampled-position pool: its move list
    is the <=7-ply LC0 history window, not the remainder of the source game.
    Consequently the lab generator's old attempt to draw another ply from
    that list would reject every current record. Use the stored end FEN
    directly, rejecting only positions in which the game is already over.
    """
    idx = 0
    for path in sorted(glob(str(pool_glob))):
        with open(path) as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                    triple = rec[:3]
                    board = chess.Board(triple[2])
                except Exception:
                    continue
                if board.is_game_over():
                    continue
                if total and idx >= total:
                    return
                yield idx, triple
                idx += 1


def _history(start_fen, moves):
    if not moves:
        return []
    b = chess.Board(start_fen)
    out = [b.fen()]
    for u in moves[:-1]:
        b.push_uci(u)
        out.append(b.fen())
    return out


def run_shard(pool_glob, out_dir, shard, num_shards, *, concurrency=32,
              max_batch=32, max_num_seqs=64, sf_judges=4, limit=0,
              total=0, depth=10, max_nodes=15, cap=4, cache_path=None,
              prefetch=True, indices_path=None, errors_path=None, use_v1_vllm=False):
    from datagen.sim import sfpool
    sfpool.install(judges=sf_judges)
    from datagen.sim import search as SS
    from datagen.sim.flatten import flatten_model_tree
    from datagen.sim.sampler import Batcher, BatchedModels, VLLMBackend

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard:02d}.jsonl"
    if errors_path is None:
        errors_path = out_dir / f"shard_{shard:02d}.errors.jsonl"
    else:
        errors_path = Path(errors_path)

    requested = None
    if indices_path:
        requested = {
            int(ln.strip()) for ln in Path(indices_path).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        }
        bad = [idx for idx in requested if idx % num_shards != shard]
        if bad:
            raise ValueError(
                f"{len(bad)} requested indices do not belong to shard {shard}"
            )
        print(f"[shard {shard}] {len(requested)} explicit indices", flush=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["idx"])
                except Exception:
                    pass            # a line torn by a kill mid-write
    print(f"[shard {shard}] {len(done)} already done", flush=True)

    backend = VLLMBackend(max_num_seqs=max_num_seqs, use_v1_vllm=use_v1_vllm)
    batcher = Batcher(backend, max_batch=max_batch)
    models = BatchedModels(batcher, cache_path=cache_path, prefetch=prefetch)

    from collections import deque
    q, lock = deque(), threading.Lock()
    fh = open(out_path, "a")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    err_fh = open(errors_path, "a")
    stats = {"ok": 0, "err": 0, "nodes": 0}
    t0 = time.perf_counter()

    def work(idx, triple):
        start_fen, moves, fen = triple
        b = SS.Builder(models, depth=depth, max_nodes=max_nodes, cap=cap,
                       union=True, force_key=True)
        root = b.build(fen)
        text = flatten_model_tree(root, b, mode="token", move_desc=True,
                                  extend_refuted=True)
        forced, bb = [], chess.Board(fen)
        for u in b.forced.get(fen, []):
            mv = chess.Move.from_uci(u)
            forced.append(bb.san(mv))
            bb.push(mv)
        return {"idx": idx, "fen": fen, "history": _history(start_fen, moves),
                "nodes": b.n_nodes, "depth": b.built_depth,
                "forced_root": forced, "response": text}

    def runner():
        while True:
            with lock:
                if not q:
                    return
                idx, triple = q.popleft()
            try:
                rec = work(idx, triple)
            except Exception as exc:
                with lock:
                    stats["err"] += 1
                    err_fh.write(json.dumps({
                        "idx": idx,
                        "type": type(exc).__name__,
                        "error": str(exc),
                    }) + "\n")
                    err_fh.flush()
                    if stats["err"] <= 3:
                        traceback.print_exc()
                continue
            with lock:                     # one writer at a time, flushed, so
                fh.write(json.dumps(rec) + "\n")   # a kill costs one position
                fh.flush()
                stats["ok"] += 1
                stats["nodes"] += rec["nodes"]
                n = stats["ok"]
                if n % 10 == 0:
                    el = time.perf_counter() - t0
                    print(f"[shard {shard}] {n} done, {stats['err']} err, "
                          f"mean_nodes {stats['nodes'] / n:.1f}, "
                          f"{3600 * n / el:.1f} pos/hour", flush=True)

    taken = 0
    for idx, triple in stream_pool(pool_glob, total=total):
        if (idx % num_shards != shard or idx in done
                or (requested is not None and idx not in requested)):
            continue
        q.append((idx, triple))
        taken += 1
        if limit and taken >= limit:
            break
    print(f"[shard {shard}] {taken} positions queued", flush=True)

    ts = [threading.Thread(target=runner) for _ in range(concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    batcher.shutdown()
    fh.close()
    err_fh.close()
    el = time.perf_counter() - t0
    print(f"[batcher] {json.dumps(batcher.stats(el), sort_keys=True)}", flush=True)
    print(f"[shard {shard}] FINAL ok={stats['ok']} err={stats['err']} "
          f"mean_nodes={stats['nodes'] / max(1, stats['ok']):.1f} "
          f"pos_per_hour={3600 * stats['ok'] / max(1e-9, el):.1f}", flush=True)


def pack(shards_dir, arrow_out, test_n=500, val_n=200):
    """Shard jsonls -> test/val/train arrows, split by global idx (records are
    position-disjoint by construction). Tolerates an in-progress array."""
    from datasets import Dataset
    rows = {}
    for f in sorted(Path(shards_dir).glob("shard_*.jsonl")):
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            rows[r["idx"]] = r
    print(f"[pack] {len(rows)} records")
    ordered = [rows[i] for i in sorted(rows)]

    def to_row(r):
        return {"fen": r["fen"], "history": r["history"], "prompt": PROMPT,
                "response": r["response"],
                "extra": {"task": TASK, "answer_class": None}}

    out = Path(arrow_out)
    out.mkdir(parents=True, exist_ok=True)
    # The source pool can contain the same FEN from different games/histories.
    # Assign every occurrence to the split of its first global index so exact
    # positions cannot leak across evaluation and training. This retains the
    # old lowest-index boundaries; a split can grow slightly when a later
    # duplicate follows its earlier occurrence into that split.
    split_of_fen, test, val, train = {}, [], [], []
    targets = {"test": test, "val": val, "train": train}
    for ordinal, r in enumerate(ordered):
        split = split_of_fen.get(r["fen"])
        if split is None:
            split = ("test" if ordinal < test_n else
                     "val" if ordinal < test_n + val_n else "train")
            split_of_fen[r["fen"]] = split
        targets[split].append(r)
    assert not ({r["fen"] for r in test} & {r["fen"] for r in val})
    assert not ({r["fen"] for r in test} & {r["fen"] for r in train})
    assert not ({r["fen"] for r in val} & {r["fen"] for r in train})

    splits = {"test_dataset.arrow": test,
              "val_dataset.arrow": val}
    (out / "training_datasets").mkdir(exist_ok=True)
    splits[str(Path("training_datasets") / f"{TASK}.arrow")] = train
    for name, data in splits.items():
        ds = Dataset.from_list([to_row(r) for r in data])
        ds.save_to_disk(str(out / name))
        print(f"[pack] {name}: {ds.num_rows} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # everything here is for run, which runs the search process and verbalization
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--num-shards", type=int, default=24)
    r.add_argument("--pool", required=True) # pool is the glob with jsonls, e.g. data/lichess/stage5/train-*.jsonl
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=32) # how many positions to run in parallel
    r.add_argument("--max-batch", type=int, default=32) # how many positions to batch into a single model call
    r.add_argument("--max-num-seqs", type=int, default=64,
                   help="vLLM sequence ceiling per model engine")
    r.add_argument("--use-v1-vllm", action="store_true", help="Use the legacy hybrid runner")
    r.add_argument("--sf-judges", type=int, default=4) # number of stockfish instances for judge (shared)
    r.add_argument("--limit", type=int, default=0) # limit number of processed positions
    r.add_argument("--total", type=int, default=0,
                   help="global accepted-position cap before sharding (0 = all)")
    r.add_argument("--cache", default=None,
                   help="optional read-only generation cache jsonl")
    r.add_argument("--no-prefetch", action="store_true",
                   help="disable speculative child generation (benchmark/debug)") # prefetch: speculatively run child of node for batching
    r.add_argument("--indices", default=None,
                   help="optional newline-delimited global indices to process")
    r.add_argument("--errors", default=None,
                   help="optional JSONL path for failed indices and exceptions")

    # pack's job is to assemble the already-run shards into the test/val/train arrows
    p = sub.add_parser("pack")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-n", type=int, default=500)
    p.add_argument("--val-n", type=int, default=200)


    a = ap.parse_args()
    if a.cmd == "run":
        run_shard(a.pool, a.out, a.shard, a.num_shards,
                  concurrency=a.concurrency, max_batch=a.max_batch,
                  max_num_seqs=a.max_num_seqs,
                  sf_judges=a.sf_judges,
                  limit=a.limit, total=a.total, cache_path=a.cache,
                  prefetch=not a.no_prefetch, indices_path=a.indices,
                  errors_path=a.errors, use_v1_vllm=a.use_v1_vllm)
    else:
        pack(a.shards, a.out, a.test_n, a.val_n)


if __name__ == "__main__":
    main()
