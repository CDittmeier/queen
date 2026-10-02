"""Task: material_count — total points of material per side.

Deterministic: every question lists material totals for both sides.
Material values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9. King is
excluded.

parse_tag layout (newline-separated):

    {side1_label}
    {side1 material integer}
    {side2_label}
    {side2 material integer}

Graded by `utils.eval_utils._exact_grade`.
"""
import random

from utils.board_representation import BoardRepr
from datagen.prose import format_piece_counts

NAME = "material_count"
MAX_UNIQUE_QUERIES = 1
SUPPORTS_FORWARD = True   # _render_forward / sample_n_forward synthesized in __init__.py

# Probability of routing a question through the CoT (per-piece value walk)
# template family vs the direct family.
COT_RATIO = 0.5

# Standard chess material values, indexed by the piece-type letter (the
# character before ">" in piece tokens like "<PIECE_WP>").
_PIECE_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

MATERIAL_COUNT_QUESTIONS_TOK = [
    "In this position, what are the material points of both sides?",
    "How many points of material does each side have in this position?",
    "List the material points of each side.",
]

# Per-side templates — formatted once for side1, once for side2, then joined.
# Slots: {side}         — side label (lowercase)
#        {material}     — integer total material points
#        {counts}       — "2 <PIECE_WP>, 1 <PIECE_WN>, and 1 <PIECE_WK>" prose (CoT only)
#        {material_cot} — per-piece value walk; ends with "." or is "" if king-only (CoT only)
MATERIAL_COUNT_DIRECT_TEMPLATES = [
    "In this position, {side} has {material} points of material.",
    "The pieces of {side} total up to {material} points.",
]

MATERIAL_COUNT_COT_TEMPLATES = [
    "Going through {side}'s pieces, there is {counts}. {material_cot} So {side} has {material} points of material.",
    "Counting up all of {side}'s pieces, there is {counts}. {material_cot} In all, {side}'s pieces total up to {material} points."
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


def _piece_value(piece_tok: str) -> int:
    return _PIECE_VALUE[piece_tok[-2]]


def _material_total(counts: list[tuple[str, int]]) -> int:
    return sum(_piece_value(p) * n for p, n in counts)


def _material_cot(counts: list[tuple[str, int]]) -> str:
    """Per-piece value walk for non-king pieces.

    Each non-king piece species contributes one sentence ending with a period;
    sentences are space-joined. Returns the empty string when the side has
    only its king.
    """
    parts = []
    for p, n in counts:
        if p[-2] == "K":
            continue
        v = _piece_value(p)
        parts.append(f"The {p} are worth {v} each, equaling {n} * {v} = {n * v} points.")
    return " ".join(parts)


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set) -> None:
    """Only one possible query per position."""
    return None


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    # piece_tokens layout: indices 0..5 = side1 (W* / M*), 6..11 = side2 (B* / O*).
    s1_label, s2_label = _side_labels(board)
    s1_toks = list(board.piece_tokens[:6])
    s2_toks = list(board.piece_tokens[6:])

    s1_counts   = _ordered_counts(board, s1_toks)
    s2_counts   = _ordered_counts(board, s2_toks)
    s1_material = _material_total(s1_counts)
    s2_material = _material_total(s2_counts)

    # parse_tag: literal side labels + integer material, newline-separated.
    parse_tag = "\n".join([
        s1_label, str(s1_material),
        s2_label, str(s2_material),
    ])

    q = rng.choice(MATERIAL_COUNT_QUESTIONS_TOK)

    # Family is chosen once per question (so prose stays consistent across
    # sides); the specific template within the family is picked independently
    # for each side so the two halves can phrase differently.
    use_cot = rng.random() < COT_RATIO
    family  = MATERIAL_COUNT_COT_TEMPLATES if use_cot else MATERIAL_COUNT_DIRECT_TEMPLATES

    parts = []
    for label, counts, material in [
        (s1_label, s1_counts, s1_material),
        (s2_label, s2_counts, s2_material),
    ]:
        fmt = {"side": label, "material": material}
        if use_cot:
            fmt["counts"]       = format_piece_counts(counts)
            fmt["material_cot"] = _material_cot(counts)
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
# `sample_n_forward`) are synthesized in `datagen/tasks/__init__.py`. Material
# changes iff the sequence captures or promotes (entity-less, one deterministic
# answer), so this uses the uniform fallback forward sampler — no balancer.
