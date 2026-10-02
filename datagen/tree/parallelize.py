"""datagen/tree/parallelize.py — position-level parallelism for the HCE analysis.

Profiling shows `get_tree(fen)` is a single-threaded ~5 s/position job (dominated by the
per-node Stockfish *static* eval + Python search; verbalization is ~5% on top). Giving
Stockfish more threads doesn't help — the per-node call is a 0-node static eval — and the
verbalization primitives are too cheap to be worth splitting. The one axis that scales is
across POSITIONS.

The three engines each wrap a single stateful subprocess pipe (write a command, read stdout to
a sentinel), so they are NOT shareable across processes; every worker must own its own set.
This module owns that pool:

  * `_init_worker` — pin every engine to one thread, put the lc0 cudalibs on LD_LIBRARY_PATH,
    and spawn this worker's own Stockfish-eval / lc0-policy / Stockfish-judge subprocesses.
  * `analyze(fens)` — shard the FENs across the cpus SLURM gave us (`--cpus-per-task`) and
    return each position's machine-readable (token) narrative, aligned with the input.

Run from the repo root — the engine binary paths in search.py are repo-relative.
"""
from __future__ import annotations

import os
from multiprocessing import get_context
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CUDALIBS = _REPO / "data" / "engines" / "cudalibs"


def cpu_budget(requested: int | None = None) -> int:
    """Worker count: an explicit request, else SLURM's --cpus-per-task, else all cores."""
    if requested:
        return int(requested)
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _init_worker() -> None:
    os.chdir(_REPO)                                   # engine paths in search.py are repo-relative
    os.environ["OMP_NUM_THREADS"] = "1"               # keep each engine single-threaded so N
    os.environ["OPENBLAS_NUM_THREADS"] = "1"          # workers share N cpus cleanly
    ld = os.environ.get("LD_LIBRARY_PATH", "")        # lc0 needs libcublas*.so.13 at load time
    if str(_CUDALIBS) not in ld:                      # even on the eigen/CPU backend
        os.environ["LD_LIBRARY_PATH"] = f"{_CUDALIBS}:{ld}".rstrip(":")
    from datagen.tree import search
    search.engines()                                  # spawn this worker's own SF/lc0/judge


def _analyze_one(item):
    """(idx, fen) -> (idx, token-narrative or None). Retries once through fresh engines on a
    transient engine death; returns None on a second failure so one bad position can't sink a
    whole shard (the builder drops it)."""
    idx, fen = item
    from datagen.tree import get_tree, search
    for attempt in (1, 2):
        try:
            token = get_tree(fen).string("token")
            ev, pol, _ = search.engines()
            ev.cache.clear(); pol.cache.clear()       # bound memory across a long stream
            return idx, token
        except Exception:
            if attempt == 2:
                return idx, None
            search.close_engines(); _init_worker()    # rebuild engines and retry once


def analyze(fens, cpus: int | None = None, chunksize: int = 1, progress: bool = True):
    """Run `get_tree(...).string("token")` over `fens` across a worker pool (each worker owns
    its own engine subprocesses) and return a list of token narratives aligned with `fens`
    (None where analysis failed)."""
    items = list(enumerate(fens))
    out: list = [None] * len(items)
    if not items:
        return out
    n = min(cpu_budget(cpus), len(items))
    ctx = get_context("fork")
    with ctx.Pool(n, initializer=_init_worker) as pool:
        it = pool.imap_unordered(_analyze_one, items, chunksize=chunksize)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(items), desc="analyze", unit="pos")
        for idx, token in it:
            out[idx] = token
    return out


# ================= stage-5 sharded production run (resumable, de-skew filtered) =================
import functools, json, glob, random, argparse, re

DROP_CP, DROP_PROB = 200, 0.25            # de-skew: |NNUE static eval| > 2 pawns -> drop with this prob


def _shard_worker(item, cover, cover_deep, max_nodes, deskew):
    """(gidx, start_fen, moves, fen) -> (gidx, start_fen, moves, fen, token|None). De-skew filter
    first (cheap NNUE eval, deterministic per gidx), then get_tree (with the given pruning config)
    only on kept positions. token=None means the position was dropped (or failed after one retry)."""
    gidx, sf, mv, fen = item
    from datagen.tree import get_tree, search
    for attempt in (1, 2):
        try:
            ev, pol, _ = search.engines()
            if deskew and abs(ev.eval(fen)) > DROP_CP and random.Random(gidx).random() < DROP_PROB:
                return (gidx, sf, mv, fen, None)              # dropped to flatten the eval distribution
            token = get_tree(fen, cover=cover, cover_deep=cover_deep,
                             max_nodes=max_nodes).string("token")
            ev.cache.clear(); pol.cache.clear()
            return (gidx, sf, mv, fen, token)
        except Exception:
            if attempt == 2:
                return (gidx, sf, mv, fen, None)
            search.close_engines(); _init_worker()


def _pool_stream(paths):
    """Yield (global_idx, start_fen, moves, end_fen) over all pool jsonls, in sorted-path order."""
    gidx = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                yield gidx, rec[0], rec[1], rec[2]
                gidx += 1


def _load_done(path):
    """(done_gidx_set, kept_count); rewrites the file dropping any truncated trailing line first."""
    if not os.path.exists(path):
        return set(), 0
    valid = []
    with open(path) as f:
        for line in f:
            try:
                valid.append(json.loads(line))
            except Exception:
                break                                         # partial tail from a killed write -> stop
    with open(path, "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")
    return {r["idx"] for r in valid}, len(valid)


def run_shard(pool_glob, out_dir, shard, nshards, target, flush=2000,
              *, cover=0.35, cover_deep=0.15, max_nodes=100, deskew=True):
    """Shard `shard` of `nshards`: process pool positions with global_idx % nshards == shard, apply the
    de-skew filter, get_tree the kept ones, and STREAM {idx,start_fen,moves,fen,token} to
    shard_<shard>.jsonl (flush+fsync every `flush`) until `target` kept. Resumable: skips already-written
    idxs and stops early when the target is met."""
    os.makedirs(out_dir, exist_ok=True)
    outp = os.path.join(out_dir, f"shard_{shard:04d}.jsonl")
    done, kept = _load_done(outp)
    if kept >= target:
        print(f"[shard {shard}] already complete: {kept}/{target}", flush=True)
        return
    paths = sorted(glob.glob(pool_glob))

    def candidates():
        for gidx, sf, mv, fen in _pool_stream(paths):
            if gidx % nshards == shard and gidx not in done:
                yield (gidx, sf, mv, fen)

    processed = 0
    with open(outp, "a") as f, get_context("fork").Pool(cpu_budget(), initializer=_init_worker) as pool:
        worker = functools.partial(_shard_worker, cover=cover, cover_deep=cover_deep,
                                   max_nodes=max_nodes, deskew=deskew)
        for gidx, sf, mv, fen, token in pool.imap_unordered(worker, candidates(), chunksize=1):
            processed += 1
            if token:
                f.write(json.dumps({"idx": gidx, "start_fen": sf, "moves": mv, "fen": fen, "token": token}) + "\n")
                kept += 1
                if kept % flush == 0:
                    f.flush(); os.fsync(f.fileno())
                    print(f"[shard {shard}] kept {kept}/{target} processed {processed}", flush=True)
                if kept >= target:
                    break
        f.flush(); os.fsync(f.fileno())
    print(f"[shard {shard}] DONE kept {kept}/{target} processed {processed}", flush=True)


_IDX_RE = re.compile(r'^\{"idx":\s*(\d+)')


def pack(shard_glob, arrow_out, test_n=500, val_n=200, pov=True):
    """Split shard jsonls by global idx (lowest -> test, then val, rest -> train; all position-disjoint)
    and write the training-ready HF arrows: <arrow_out>/{train_all.arrow, val_dataset.arrow,
    test_dataset.arrow} — the paths/names utils.training_utils reads (train_all.arrow also carries a
    dataset_config.json with `pov`). Memory-light: an idx-only pass fixes the split, the small eval
    splits use from_list, and the large train split is STREAMED via Dataset.from_generator (bounded peak
    memory — a plain from_list on ~1M large rows OOMs)."""
    import shutil
    from datasets import Dataset
    from datagen.build_qa_dataset import _build_history, _save_dataset
    from datagen.tasks.hce_rationale import PROMPT, NAME
    files = sorted(glob.glob(shard_glob))

    idxs = []                                                # pass 1: idxs only (idx is the first key)
    for p in files:
        with open(p) as f:
            for l in f:
                m = _IDX_RE.match(l)
                if m:
                    idxs.append(int(m.group(1)))
    idxs.sort()
    test_idx, val_idx = set(idxs[:test_n]), set(idxs[test_n:test_n + val_n])

    def rec(r):
        return {"fen": r["fen"], "history": _build_history(r["start_fen"], r["moves"]),
                "prompt": PROMPT, "response": r["token"], "extra": {"task": NAME, "answer_class": None}}

    def scan():
        for p in files:
            with open(p) as f:
                for l in f:
                    try:
                        yield json.loads(l)
                    except Exception:
                        pass                                 # partial trailing line (shard still writing)

    test_rows, val_rows = [], []                             # pass 2: the small eval splits
    for r in scan():
        if r["idx"] in test_idx:
            test_rows.append(rec(r))
        elif r["idx"] in val_idx:
            val_rows.append(rec(r))
    os.makedirs(arrow_out, exist_ok=True)
    _save_dataset(test_rows, os.path.join(arrow_out, "test_dataset.arrow"))
    _save_dataset(val_rows, os.path.join(arrow_out, "val_dataset.arrow"))

    def train_gen():                                         # pass 3: train, streamed to disk in batches
        for r in scan():
            if r["idx"] not in test_idx and r["idx"] not in val_idx:
                yield rec(r)
    cache = os.path.join(arrow_out, "_hf_cache")
    train_dir = os.path.join(arrow_out, "train_all.arrow")   # the arrow the trainer loads directly
    Dataset.from_generator(train_gen, cache_dir=cache).save_to_disk(train_dir)
    shutil.rmtree(cache, ignore_errors=True)                 # from_generator leaves a redundant cache
    with open(os.path.join(train_dir, "dataset_config.json"), "w") as cf:   # trainer sources `pov` here
        json.dump({"pov": pov, "split": "train", "task": NAME}, cf, indent=2)
    print(f"packed: test={len(test_rows)} val={len(val_rows)} "
          f"train={len(idxs) - len(test_idx) - len(val_idx)} -> {arrow_out}/train_all.arrow", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage-5 sharded HCE-rationale runner (--mode run) + packer (--mode pack).")
    ap.add_argument("--mode", choices=["run", "pack"], required=True)
    ap.add_argument("--pool", default="data/lichess/stage5/train-*.jsonl")
    ap.add_argument("--out", default="data/stage5/shards")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=72)
    ap.add_argument("--target", type=int, default=17362)      # ceil(1.25M / 72)
    ap.add_argument("--flush", type=int, default=2000)
    ap.add_argument("--arrow-out", default="data/stage5")
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--cover",      type=float, default=0.35)   # get_tree root cover mass
    ap.add_argument("--cover-deep", type=float, default=0.15)   # cover mass at deeper nodes
    ap.add_argument("--max-nodes",  type=int,   default=100)    # pruned-tree size cap
    ap.add_argument("--no-deskew", action="store_true",
                    help="disable the |eval|>2pawn 25%% drop (keep every position, e.g. puzzles)")
    a = ap.parse_args()
    if a.mode == "run":
        run_shard(a.pool, a.out, a.shard, a.nshards, a.target, a.flush,
                  cover=a.cover, cover_deep=a.cover_deep, max_nodes=a.max_nodes,
                  deskew=not a.no_deskew)
    else:
        pack(os.path.join(a.out, "shard_*.jsonl"), a.arrow_out, a.test_n, a.val_n)
