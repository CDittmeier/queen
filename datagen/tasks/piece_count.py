"""Task: piece_count — list piece counts for both sides.

Deterministic: every question lists counts for both sides. Side markers
are literal strings (`"white"`/`"black"` in abs, `"player"`/`"opponent"`
in POV). Each side is guaranteed a king, so per-piece absences are simply
omitted (no `<PIECE>0` or `<EMPTY>` entries).

parse_tag layout (newline-separated):

    {side1_label}
    {side1 compact piece list}
    {side2_label}
    {side2 compact piece list}

Graded by `utils.eval_utils._piece_count_grade`: side labels matched
exactly; each side's compact piece line graded as a counted multiset.
"""
import random

from utils.board_representation import BoardRepr
from datagen.prose import (
    encode_piece_count,
    format_piece_counts,
    join_and,
)

NAME = "piece_count"
MAX_UNIQUE_QUERIES = 1
SUPPORTS_FORWARD = True   # _render_forward / sample_n_forward synthesized in __init__.py

# Probability of routing a question through the CoT (per-piece location walk)
# template family vs the direct family.
COT_RATIO = 0.5

PIECE_COUNT_QUESTIONS_TOK = [
    "What are the piece counts of both sides on the board?",
    "How many of each piece does each side have in this position?",
    "List the piece counts by side.",
]

# Per-side templates — formatted once for side1, once for side2, then joined.
# Slots: {side}, {side_cap} — side label (lowercase / capitalized)
#        {counts} — "2 <PIECE_WP> and 1 <PIECE_WR>" prose
#        {walk}   — "<PIECE_WP> on <SQUARE_A2> and <SQUARE_B2>, ..."  (CoT only)
PIECE_COUNT_DIRECT_TEMPLATES = [
    "In this position, {side} has {counts}.",
    "The pieces of {side} are as follows: {counts}.",
]

PIECE_COUNT_COT_TEMPLATES = [
    "Going through {side}'s pieces, there is {walk}. So {side} has {counts}.",
    "Counting all of {side}'s pieces, there is {walk}. In all, {side} has {counts}."
]


def _side_labels(board: BoardRepr) -> tuple[str, str]:
    return ("player", "opponent") if board.pov else ("white", "black")


def _ordered_counts(board: BoardRepr, piece_toks: list[str]) -> list[tuple[str, int]]:
    """[(piece_tok, n), ...] for one side, n>0 only, in canonical order."""
    out = []
    for p in piece_toks:
        n = len(board.squares_with(p))
        if n > 0:
            out.append((p, n))
    return out


def _side_walk(board: BoardRepr, counts: list[tuple[str, int]], rng: random.Random) -> str:
    """Per-piece-species walk: '<PIECE_WP> on A and B; <PIECE_WN> on C; ...'.

    Square listing order within each species is shuffled (grader is on parse_tag,
    not prose, so listing order is free for prose diversity). Piece groups are
    separated by '; ' so the inner comma-joined square lists don't visually
    merge with the outer joiner.
    """
    parts = []
    for p, _ in counts:
        sq_toks = [board.sq_tok(s) for s in board.squares_with(p)]
        rng.shuffle(sq_toks)
        parts.append(f"{p} on {join_and(sq_toks)}")
    return "; ".join(parts)


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set) -> None:
    """Only one possible query per position."""
    return None


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    # piece_tokens layout: indices 0..5 = side1 (W* / M*), 6..11 = side2 (B* / O*).
    s1_label, s2_label = _side_labels(board)
    s1_toks = list(board.piece_tokens[:6])
    s2_toks = list(board.piece_tokens[6:])

    s1_counts = _ordered_counts(board, s1_toks)
    s2_counts = _ordered_counts(board, s2_toks)

    s1_encoded = [encode_piece_count(p, c) for p, c in s1_counts]
    s2_encoded = [encode_piece_count(p, c) for p, c in s2_counts]

    # parse_tag: literal side labels + encoded pieces, newline-separated.
    parse_tag = "\n".join([
        s1_label, "".join(s1_encoded),
        s2_label, "".join(s2_encoded),
    ])

    q = rng.choice(PIECE_COUNT_QUESTIONS_TOK)

    # Family is chosen once per question (so prose stays consistent across
    # sides); the specific template within the family is picked independently
    # for each side so the two halves can phrase differently.
    use_cot = rng.random() < COT_RATIO
    family  = PIECE_COUNT_COT_TEMPLATES if use_cot else PIECE_COUNT_DIRECT_TEMPLATES

    parts = []
    for label, counts in [(s1_label, s1_counts), (s2_label, s2_counts)]:
        fmt = {"side": label, "counts": format_piece_counts(counts)}
        if use_cot:
            fmt["walk"] = _side_walk(board, counts, rng)
        parts.append(rng.choice(family).format(**fmt))
    a = " ".join(parts)

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  None,
    }


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set = set()
    out: list[dict] = []
    while len(out) < n:
        e = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(e)
        out.append(_render(e, board, rng))
    return out


# `sample_one`/`sample_all` and the forward API (`_render_forward`,
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`. Counts
# change iff the sequence captures or promotes (entity-less, one deterministic
# answer), so this uses the uniform fallback forward sampler — no balancer.
