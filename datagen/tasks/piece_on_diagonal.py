"""Task: piece_on_diagonal — "what pieces are on this diagonal?"

26 diagonals (13 up-right + 13 up-left); both walk bottom-up. Compact
piece-with-count encoding `<PIECE>?N` in parse_tag + answer_class via
`datagen.prose.line_facts`.
"""
import random

from utils.board_representation import BoardRepr
from datagen.prose import (
    format_piece_counts,
    format_square_breakdown,
    line_facts,
)
from utils.utils import EMPTY_TOKEN

NAME = "piece_on_diagonal"
MAX_UNIQUE_QUERIES = 26
SUPPORTS_FORWARD = True   # _render_forward / sample_n_forward synthesized in __init__.py

# Probability of routing a question through the CoT (per-square walk) template
# family vs the direct family.
COT_RATIO = 0.5

DIAGONAL_QUESTIONS_TOK = [
    "What piece(s) are on the diagonal from {start_tok} to {end_tok}?",
    "Identify the piece(s) that occupy the {start_tok}{end_tok} diagonal.",
    "In this position, the {start_tok}{end_tok} diagonal contains what piece(s)?"
]

DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the diagonal from {start_tok} to {end_tok}.",
    "The {start_tok}{end_tok} diagonal contains {piece_counts}.",
    "In this position, {piece_counts} occupy the {start_tok}{end_tok} diagonal."
]

DIAGONAL_COT_PRESENT_ANSWERS_TOK = [
    "Walking the diagonal from {start_tok} to {end_tok}: {square_breakdown}.",
    "Going up the {start_tok}{end_tok} diagonal: {square_breakdown}."
]

DIAGONAL_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the diagonal from {start_tok} to {end_tok}. It is an open diagonal.",
    "The {start_tok}{end_tok} diagonal is open. There are no pieces on it.",
    "In this position, the {start_tok}{end_tok} is an open diagonal. It contains no pieces."
]


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[tuple[int, ...]]) -> tuple[int, ...]:
    """Length-proportional pick over `board.diagonals() \\ exclude`.

    Diagonals span 2..8 squares; weighting by length gives every square an
    equal chance of being in the queried diagonal (modulo the two diagonals
    per square).
    """
    entities = [d for d in board.diagonals() if d not in exclude]
    return rng.choices(entities, weights=[len(d) for d in entities], k=1)[0]


def _render(diag_sqs: tuple[int, ...], board: BoardRepr, rng: random.Random) -> dict:
    f = line_facts(diag_sqs, board)
    use_cot = rng.random() < COT_RATIO

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = DIAGONAL_EMPTY_ANSWERS_TOK
    elif use_cot:
        items = [
            (board.sq_tok(sq),
             None if board.piece_at(sq) == EMPTY_TOKEN else board.piece_at(sq))
            for sq in diag_sqs
        ]
        fmt = {
            "start_tok":        f["start_tok"],
            "end_tok":          f["end_tok"],
            "square_breakdown": format_square_breakdown(items),
        }
        a_t = DIAGONAL_COT_PRESENT_ANSWERS_TOK
    else:
        fmt = {
            "start_tok":    f["start_tok"],
            "end_tok":      f["end_tok"],
            "piece_counts": format_piece_counts(f["ordered"]),
        }
        a_t = DIAGONAL_DIRECT_PRESENT_ANSWERS_TOK
    q = rng.choice(DIAGONAL_QUESTIONS_TOK).format(**fmt)
    a = rng.choice(a_t).format(**fmt)
    return {
        "question":      q,
        "answer":        f"{a}\n\n{f['parse_tag']}",
        "question_type": NAME,
        "answer_class":  None,
    }


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set[tuple[int, ...]] = set()
    out: list[dict] = []
    while len(out) < n:
        e = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(e)
        out.append(_render(e, board, rng))
    return out


# `sample_one`/`sample_all` and the forward API (`_render_forward`,
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`. Line tasks
# have a high natural changed rate, so this uses the uniform fallback forward
# sampler — no distinctness balancer.
