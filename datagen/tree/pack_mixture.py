"""Pack the hce040 game + puzzle shards into the two stage-5 training arrows.

Outputs (all splits position-disjoint, one common eval split shared by both variants):
  <out-game>/{test_dataset,val_dataset}.arrow   common game-only test/val (lowest global idxs)
  <out-game>/train_all.arrow                    variant A: game positions only
  <out-mix>/train_all.arrow                     variant B: same game train + --mix-n random puzzles,
                                                randomly interleaved
  <heldout-out>.arrow + .jsonl                  --heldout-n random puzzles held out from BOTH
                                                variants for future evaluation

Usage: python -m datagen.tree.pack_mixture   (defaults = production paths)
"""
import argparse
import glob
import json
import os
import random
import shutil


from datasets import Dataset

from datagen.build_qa_dataset import _build_history, _save_dataset
from datagen.tasks.hce_rationale import PROMPT, NAME


def scan(files):
    for p in files:
        with open(p) as f:
            for l in f:
                yield json.loads(l)


def rec(r):
    return {"fen": r["fen"], "history": _build_history(r["start_fen"], r["moves"]),
            "prompt": PROMPT, "response": r["token"], "extra": {"task": NAME, "answer_class": None}}


def collect_idxs(files):
    return [r["idx"] for r in scan(files)]


def save_train(gen, out_dir, pov=True):
    cache = os.path.join(out_dir, "_hf_cache")
    train_dir = os.path.join(out_dir, "train_all.arrow")
    Dataset.from_generator(gen, cache_dir=cache).save_to_disk(train_dir)
    shutil.rmtree(cache, ignore_errors=True)
    with open(os.path.join(train_dir, "dataset_config.json"), "w") as cf:
        json.dump({"pov": pov, "split": "train", "task": NAME}, cf, indent=2)


def interleave(gen_a, gen_b, n_a, n_b, seed):
    """Randomly interleave two streams (each stays in its own order; counts may run short by a
    few truncated lines, so drained labels just fall through)."""
    labels = ["a"] * n_a + ["b"] * n_b
    random.Random(seed).shuffle(labels)
    ia, ib = iter(gen_a), iter(gen_b)
    for lab in labels:
        r = next(ia if lab == "a" else ib, None)
        if r is not None:
            yield r
    yield from ia
    yield from ib


def main(a):
    game_files = sorted(glob.glob(a.game_shards))
    puzzle_files = sorted(glob.glob(a.puzzle_shards))
    assert game_files and puzzle_files, "empty shard glob"

    game_idxs = sorted(collect_idxs(game_files))
    test_idx = set(game_idxs[:a.test_n])
    val_idx = set(game_idxs[a.test_n:a.test_n + a.val_n])
    n_game_train = len(game_idxs) - len(test_idx) - len(val_idx)

    puz_idxs = collect_idxs(puzzle_files)
    random.Random(a.seed).shuffle(puz_idxs)
    heldout_idx = set(puz_idxs[:a.heldout_n])
    mix_idx = set(puz_idxs[a.heldout_n:a.heldout_n + a.mix_n])
    assert len(mix_idx) == a.mix_n, f"only {len(puz_idxs) - a.heldout_n} puzzles left for the mix"
    print(f"game: {len(game_idxs)} (test {len(test_idx)} val {len(val_idx)} train {n_game_train}) | "
          f"puzzles: {len(puz_idxs)} (heldout {len(heldout_idx)} mix {len(mix_idx)})", flush=True)

    # common game test/val + puzzle heldout (small: in-memory)
    test_rows, val_rows = [], []
    for r in scan(game_files):
        if r["idx"] in test_idx:
            test_rows.append(rec(r))
        elif r["idx"] in val_idx:
            val_rows.append(rec(r))
    os.makedirs(a.out_game, exist_ok=True)
    _save_dataset(test_rows, os.path.join(a.out_game, "test_dataset.arrow"))
    _save_dataset(val_rows, os.path.join(a.out_game, "val_dataset.arrow"))

    heldout_raw = [r for r in scan(puzzle_files) if r["idx"] in heldout_idx]
    os.makedirs(os.path.dirname(a.heldout_out), exist_ok=True)
    with open(a.heldout_out + ".jsonl", "w") as f:
        for r in heldout_raw:
            f.write(json.dumps(r) + "\n")
    _save_dataset([rec(r) for r in heldout_raw], a.heldout_out + ".arrow")
    print(f"eval splits written: test {len(test_rows)} val {len(val_rows)} "
          f"heldout-puzzles {len(heldout_raw)}", flush=True)

    def game_train():
        for r in scan(game_files):
            if r["idx"] not in test_idx and r["idx"] not in val_idx:
                yield rec(r)

    def puzzle_mix():
        for r in scan(puzzle_files):
            if r["idx"] in mix_idx:
                yield rec(r)

    save_train(game_train, a.out_game)
    print(f"variant A (game only) -> {a.out_game}/train_all.arrow", flush=True)

    os.makedirs(a.out_mix, exist_ok=True)
    save_train(lambda: interleave(game_train(), puzzle_mix(), n_game_train, len(mix_idx), a.seed + 1),
               a.out_mix)
    print(f"variant B (game + {len(mix_idx)} puzzles) -> {a.out_mix}/train_all.arrow", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-shards", default="data/stage5/0-0-40/shards/shard_*.jsonl")
    ap.add_argument("--puzzle-shards", default="data/stage5/puzzles-0-0-40/shards/shard_*.jsonl")
    ap.add_argument("--out-game", default="data/stage5/0-0-40")
    ap.add_argument("--out-mix", default="data/stage5/0-0-40-puzzles100k")
    ap.add_argument("--heldout-out", default="data/stage5/puzzles-0-0-40/heldout_eval",
                    help="prefix; writes <prefix>.jsonl (raw rows) and <prefix>.arrow (eval-ready)")
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--heldout-n", type=int, default=10000)
    ap.add_argument("--mix-n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
