"""Stage-5.b task: both sides' plans, derived from stored selfplay games.

The response is the machine (POV-token) plan list for the side to move, then
the opponent — datagen.sim.plans.plans_from_games over the games the simulate
sampler stored in the position record's metadata slot (sample_positions.py
`simulate: true`). Games are never re-run here: a record without stored games,
or a position where either side has no renderable plan, yields no row.
"""
from concurrent.futures import ProcessPoolExecutor

NAME = "plan_eval"
MAX_UNIQUE_QUERIES = 1
PARALLEL_ANALYZE = True     # build_qa_dataset routes this through module.analyze
PROMPT = "What are the plans for each side in this chess position?"


def _work(item):
    from datagen.sim.plans import plans_from_games
    _start_fen, _moves, fen, meta = item
    if not isinstance(meta, dict) or not meta.get("games"):
        return None
    try:
        out = plans_from_games(fen, meta["games"])
    except Exception:
        return None
    pov, opp = out["plans"]["pov"], out["plans"]["opp"]
    if not pov or not opp:
        return None
    return ("<PLAYER>'s plans:\n" + "\n".join(p["machine"] for p in pov)
            + "\n\n<OPPONENT>'s plans:\n"
            + "\n".join(p["machine"] for p in opp))


def analyze(triples):
    """One machine plan response (or None) per pool record."""
    from datagen.tree.parallelize import cpu_budget
    with ProcessPoolExecutor(max_workers=cpu_budget()) as ex:
        return list(ex.map(_work, triples, chunksize=1))
