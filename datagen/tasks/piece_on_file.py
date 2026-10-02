"""Task: piece_on_file — "what pieces are on this file?"

Compact piece-with-count encoding `<PIECE>?N` in parse_tag + answer_class
via `datagen.prose.line_facts`.
"""
import random

from utils.board_representation import BoardRepr
from datagen.prose import (
    format_piece_counts,
    format_square_breakdown,
    line_facts,
)
from utils.utils import EMPTY_TOKEN

NAME = "piece_on_file"
MAX_UNIQUE_QUERIES = 8
SUPPORTS_FORWARD = True   # _render_forward / sample_n_forward synthesized in __init__.py

# Probability of routing a question through the CoT (per-square walk) template
# family vs the direct family.
COT_RATIO = 0.5

FILE_QUESTIONS_TOK = [
    "What piece(s) are on the file from {start_tok} to {end_tok}?",
    "Identify the piece(s) that occupy the {start_tok}{end_tok} file.",
    "In this position, the {start_tok}{end_tok} file contains what piece(s)?"
]

FILE_DIRECT_PRESENT_ANSWERS_TOK = [
    "There are {piece_counts} on the file from {start_tok} to {end_tok}.",
    "The {start_tok}{end_tok} file contains {piece_counts}.",
    "In this position, {piece_counts} occupy the {start_tok}{end_tok} file."
]

FILE_COT_PRESENT_ANSWERS_TOK = [
    "Walking the file from {start_tok} to {end_tok}: {square_breakdown}.",
    "Going up the {start_tok}{end_tok} file: {square_breakdown}."
]

FILE_EMPTY_ANSWERS_TOK = [
    "There are no pieces on the file from {start_tok} to {end_tok}. It is an open file.",
    "The {start_tok}{end_tok} file is open. There are no pieces on it.",
    "In this position, the {start_tok}{end_tok} is an open file. It contains no pieces."
]


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[tuple[int, ...]]) -> tuple[int, ...]:
    """Uniform pick over `board.files() \\ exclude`."""
    entities = [f for f in board.files() if f not in exclude]
    return rng.choice(entities)


def _render(file_sqs: tuple[int, ...], board: BoardRepr, rng: random.Random) -> dict:
    f = line_facts(file_sqs, board)
    use_cot = rng.random() < COT_RATIO

    if not f["ordered"]:
        fmt = {"start_tok": f["start_tok"], "end_tok": f["end_tok"]}
        a_t = FILE_EMPTY_ANSWERS_TOK
    elif use_cot:
        items = [
            (board.sq_tok(sq),
             None if board.piece_at(sq) == EMPTY_TOKEN else board.piece_at(sq))
            for sq in file_sqs
        ]
        fmt = {
            "start_tok":        f["start_tok"],
            "end_tok":          f["end_tok"],
            "square_breakdown": format_square_breakdown(items),
        }
        a_t = FILE_COT_PRESENT_ANSWERS_TOK
    else:
        fmt = {
            "start_tok":    f["start_tok"],
            "end_tok":      f["end_tok"],
            "piece_counts": format_piece_counts(f["ordered"]),
        }
        a_t = FILE_DIRECT_PRESENT_ANSWERS_TOK
    q = rng.choice(FILE_QUESTIONS_TOK).format(**fmt)
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
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`.
# Files change whenever any of their squares do (~43% of the time), so this task
# uses the uniform fallback forward sampler — no distinctness balancer.
