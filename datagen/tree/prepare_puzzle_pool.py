"""Puzzle position shards -> stage-5 pool-format jsonls.

Converts data/lichess/puzzles/shard-001..003.jsonl (300k positions) into the
3-array pool records parallelize --mode run consumes:
    [start_fen, moves, end_fen] = [meta.original_fen, history, fen]
(the setup move is the one-ply history; the tree roots on the post-setup fen).
Writes data/lichess/puzzles/pool/pool-000..002.jsonl.

    python -m datagen.tree.prepare_puzzle_pool
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = [REPO / f"data/lichess/puzzles/shard-{i:03d}.jsonl" for i in (1, 2, 3)]
OUT = REPO / "data/lichess/puzzles/pool"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k, src in enumerate(SRC):
        n = 0
        with open(src) as f, open(OUT / f"pool-{k:03d}.jsonl", "w") as g:
            for line in f:
                r = json.loads(line)
                g.write(json.dumps([r["meta"]["original_fen"], r["history"], r["fen"]]) + "\n")
                n += 1
        print(f"{src.name} -> pool-{k:03d}.jsonl: {n}", flush=True)


if __name__ == "__main__":
    main()
