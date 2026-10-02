"""Task: piece_on_square — name the piece (or EMPTY) on a queried square."""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN
from datagen.prose import DISTINCT_CHANGED, DISTINCT_SAME

NAME = "piece_on_square"
MAX_UNIQUE_QUERIES = 64
SUPPORTS_FORWARD = True   # forward API synthesized in __init__.py (distinctness-balanced)

SQ_QUESTIONS_TOK = [
    "What piece is on {sq_tok}?",
    "In this position, what piece occupies {sq_tok}?",
    "{sq_tok} contains what piece?",
]

SQ_OCCUPIED_ANSWERS_TOK = [
    "There is {piece_tok} on {sq_tok}.",
    "{sq_tok} contains {piece_tok}.",
    "In this position, {piece_tok} is on {sq_tok}.",
]

SQ_EMPTY_ANSWERS_TOK = [
    "{sq_tok} is empty.",
    "There is no piece on {sq_tok}.",
]


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[int]) -> int:
    """Two independent shaping factors over `range(64) \\ exclude`:
      (1) piece-class balance — mult correction + piece_tok freq.
      (2) queried-square balance — sq_tok freq.
    """
    entities   = [sq for sq in range(64) if sq not in exclude]
    piece_toks = [board.piece_at(sq) for sq in entities]
    sq_toks    = [board.sq_tok(sq)   for sq in entities]

    mult: dict[str, int] = defaultdict(int)
    for p in piece_toks:
        mult[p] += 1
    weights = [
        1.0 / (
            (frequency.get(p, 0) + 1) * mult[p]
            * (frequency.get(s, 0) + 1)
        )
        for p, s in zip(piece_toks, sq_toks)
    ]
    return rng.choices(entities, weights=weights, k=1)[0]


def _render(sq: int, board: BoardRepr, rng: random.Random) -> dict:
    piece_tok = board.piece_at(sq)
    sq_tok    = board.sq_tok(sq)
    if piece_tok == EMPTY_TOKEN:
        fmt = {"sq_tok": sq_tok}
        q = rng.choice(SQ_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(SQ_EMPTY_ANSWERS_TOK).format(**fmt)
    else:
        fmt = {"sq_tok": sq_tok, "piece_tok": piece_tok}
        q = rng.choice(SQ_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(SQ_OCCUPIED_ANSWERS_TOK).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{sq_tok}{piece_tok}",
        "question_type": NAME,
        "answer_class":  [sq_tok, piece_tok],
    }


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set[int] = set()
    out: list[dict] = []
    while len(out) < n:
        sq = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(sq)
        out.append(_render(sq, board, rng))
    return out


def _distinctness_changed(board: BoardRepr, final: BoardRepr, sq: int) -> bool:
    """Did the queried square's occupant change over the sequence? (Drives the
    synthesized forward sampler's changed/unchanged balancing.)"""
    return board.piece_at(sq) != final.piece_at(sq)


def _choose_entity_forward(board: BoardRepr, final: BoardRepr, move_sequence,
                           frequency: dict, rng: random.Random,
                           exclude: set[int]) -> int:
    """Forward weighter. Distinctness is the PRIMARY axis (stages 3/4 are about
    looking ahead at what the moves change), so a frequency-balanced coin first
    picks the changed-vs-unchanged bucket — self-correcting toward 50/50 over a
    run — then piece-class + square balance happen WITHIN that bucket, over the
    FINAL board's occupants (the actual answer). Falls back to the other bucket
    when one is empty (e.g. a position whose sequence changes nothing visible)."""
    entities = [sq for sq in range(64) if sq not in exclude]
    changed  = [sq for sq in entities if board.piece_at(sq) != final.piece_at(sq)]
    same     = [sq for sq in entities if board.piece_at(sq) == final.piece_at(sq)]
    if changed and same:
        wc = 1.0 / (frequency.get(DISTINCT_CHANGED, 0) + 1)
        ws = 1.0 / (frequency.get(DISTINCT_SAME, 0) + 1)
        pool = changed if rng.random() < wc / (wc + ws) else same
    else:
        pool = changed or same

    pieces  = [final.piece_at(sq) for sq in pool]
    sq_toks = [final.sq_tok(sq)   for sq in pool]   # anchor-stable (== board's)
    mult: dict[str, int] = defaultdict(int)
    for p in pieces:
        mult[p] += 1
    weights = [
        1.0 / ((frequency.get(p, 0) + 1) * mult[p] * (frequency.get(s, 0) + 1))
        for p, s in zip(pieces, sq_toks)
    ]
    return rng.choices(pool, weights=weights, k=1)[0]


# `sample_one`/`sample_all` and the forward API (`_render_forward`,
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`. This task
# opts into distinctness-balanced forward sampling via `_choose_entity_forward`
# + `_distinctness_changed` above.
