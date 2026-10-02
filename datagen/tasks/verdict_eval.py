"""Stage-5.a task: the verdict — '{side} is {verdict} due to {reason}'.

The answer is not sampled off a BoardRepr: each position is classified through
the strong verdict primitives (datagen.tree.strong_primitives — mate/tactic,
positional at the frozen activity cutoffs, material, dynamic/dry balance) and
the response is the machine (POV-token) sentence of the class that admitted
it. A position no class admits yields no record, so this task expects a pool
pre-filtered by the verdict sampler (sample_positions.py `verdict: true`);
classification is seeded per position, so the sampler's class decisions are
re-derived here bit-for-bit.

Like hce_rationale this is a PARALLEL_ANALYZE task, but it carries its own
`analyze` (classification needs the setup move — the triple's last history
move — not just the fen, and its own Stockfish pool per worker).
"""
from concurrent.futures import ProcessPoolExecutor

import chess

NAME = "verdict_eval"
MAX_UNIQUE_QUERIES = 1
PARALLEL_ANALYZE = True     # build_qa_dataset routes this through module.analyze
PROMPT = "What is the evaluation of the current position, and why?"

_ENG = None                 # per-worker-process engine set


def _work(item):
    global _ENG
    from datagen.sim.tactics import Engines
    from datagen.tree.strong_primitives import classify
    if _ENG is None:
        _ENG = Engines()
    start_fen, moves, _present_fen, _seq = item
    if not moves:
        return None         # no setup move: the tactic tagger cannot run
    try:
        before = chess.Board(start_fen)
        for u in moves[:-1]:
            before.push_uci(u)
        setup = chess.Move.from_uci(moves[-1])
        res = classify(_ENG, before, setup)
    except Exception:
        return None
    return None if res is None else res["machine"]


def analyze(triples):
    """One machine verdict (or None) per (start_fen, moves, present_fen, seq)."""
    from datagen.tree.parallelize import cpu_budget
    with ProcessPoolExecutor(max_workers=cpu_budget()) as ex:
        return list(ex.map(_work, triples, chunksize=4))
