"""Task: square_of_piece — list squares holding a queried piece species."""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.prose import DISTINCT_CHANGED, DISTINCT_SAME, join_and
from utils.utils import EMPTY_TOKEN

NAME = "square_of_piece"
MAX_UNIQUE_QUERIES = 12
SUPPORTS_FORWARD = True   # forward API synthesized in __init__.py (distinctness-balanced)

PC_QUESTIONS_TOK = [
    "What square(s) is {piece_tok} on?",
    "Where is {piece_tok} in this position?",
    "Locate {piece_tok} on the board.",
]

PC_PRESENT_ANSWERS_TOK = [
    "{piece_tok} can be found on {squares}.",
    "In this position, {piece_tok} is on {squares}.",
    "{piece_tok} occupies {squares}.",
]

PC_ABSENT_ANSWERS_TOK = [
    "There is no {piece_tok} in this position.",
    "This position has no {piece_tok}.",
]


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[str]) -> str:
    """Answer-side balance over `board.piece_tokens \\ exclude`.

    Summed sq_tok freqs naturally downweight high-cardinality species; mult
    correction collapses the (EMPTY,) group when several pieces are absent.
    """
    entities = [p for p in board.piece_tokens if p not in exclude]
    answer_tuples: list[tuple[str, ...]] = []
    for piece_tok in entities:
        board_sqs = board.squares_with(piece_tok)
        if board_sqs:
            answer_tuples.append(tuple(board.sq_tok(s) for s in board_sqs))
        else:
            answer_tuples.append((EMPTY_TOKEN,))

    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
    ]
    return rng.choices(entities, weights=weights, k=1)[0]


def _render(piece_tok: str, board: BoardRepr, rng: random.Random) -> dict:
    board_sqs = board.squares_with(piece_tok)
    if not board_sqs:
        fmt = {"piece_tok": piece_tok}
        q = rng.choice(PC_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(PC_ABSENT_ANSWERS_TOK).format(**fmt)
        parse_tag    = f"{piece_tok}{EMPTY_TOKEN}"
        answer_class = [EMPTY_TOKEN]
    else:
        sq_toks = [board.sq_tok(s) for s in board_sqs]
        # Randomize listing order so the model doesn't memorize a canonical
        # enumeration. Prose and parse_tag share the same shuffled order.
        rng.shuffle(sq_toks)
        fmt = {"piece_tok": piece_tok, "squares": join_and(sq_toks)}
        q = rng.choice(PC_QUESTIONS_TOK).format(**fmt)
        a = rng.choice(PC_PRESENT_ANSWERS_TOK).format(**fmt)
        parse_tag    = f"{piece_tok}{''.join(sq_toks)}"
        answer_class = list(sq_toks)

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  answer_class,
    }


def _square_set(board: BoardRepr, piece_tok: str) -> frozenset:
    """The set of square tokens holding `piece_tok` (order-independent answer key
    — the rendered list is shuffled, so compare sets to detect a changed answer)."""
    return frozenset(board.sq_tok(s) for s in board.squares_with(piece_tok))


def _distinctness_changed(board: BoardRepr, final: BoardRepr, piece_tok: str) -> bool:
    """Did this species' square set change over the sequence? (Drives the
    synthesized forward sampler's changed/unchanged balancing.)"""
    return _square_set(board, piece_tok) != _square_set(final, piece_tok)


def _choose_entity_forward(board: BoardRepr, final: BoardRepr, move_sequence,
                           frequency: dict, rng: random.Random,
                           exclude: set[str]) -> str:
    """Forward weighter. Distinctness is the PRIMARY axis: a frequency-balanced
    coin first picks the changed-vs-unchanged species bucket (a species' square
    set on the final board vs the initial board), self-correcting toward 50/50;
    then the existing answer-side balance (sq_tok freqs + mult collapse) runs
    WITHIN that bucket over the FINAL board. Falls back when a bucket is empty."""
    entities = [p for p in final.piece_tokens if p not in exclude]
    changed = [p for p in entities if _square_set(board, p) != _square_set(final, p)]
    same    = [p for p in entities if _square_set(board, p) == _square_set(final, p)]
    if changed and same:
        wc = 1.0 / (frequency.get(DISTINCT_CHANGED, 0) + 1)
        ws = 1.0 / (frequency.get(DISTINCT_SAME, 0) + 1)
        pool = changed if rng.random() < wc / (wc + ws) else same
    else:
        pool = changed or same

    answer_tuples: list[tuple[str, ...]] = []
    for p in pool:
        board_sqs = final.squares_with(p)
        answer_tuples.append(tuple(final.sq_tok(s) for s in board_sqs)
                             if board_sqs else (EMPTY_TOKEN,))
    mult: dict[tuple, int] = defaultdict(int)
    for a in answer_tuples:
        mult[a] += 1
    weights = [
        1.0 / ((sum(frequency.get(t, 0) for t in a) + 1) * mult[a])
        for a in answer_tuples
    ]
    return rng.choices(pool, weights=weights, k=1)[0]


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < n:
        p = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(p)
        out.append(_render(p, board, rng))
    return out


# `sample_one`/`sample_all` and the forward API (`_render_forward`,
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`. This task
# opts into distinctness-balanced forward sampling via `_choose_entity_forward`
# + `_distinctness_changed` above.
