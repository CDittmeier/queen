"""Task: square_attackers — who attacks (and defends) a queried square.

Stage-2 dynamic task. The entity is a board square, which may be empty, hold one
of our pieces, or hold an opponent piece. Attack/defence comes from
position_features.get_attackers_and_defenders (control-based: pinned pieces and
kings count; pawns only their diagonals; sliders respect blockers; en passant not
modelled). For an empty square every controller is an attacker and there are no
defenders.

Question framing differs for empty vs occupied squares; the answer/CoT structure
is shared. The CoT lists attackers first, then defenders, each flattened by
prose.format_piece_square_list (heavier piece first, then ascending square). Empty
squares list only attackers (no defender mention). Occupied squares say "not
attacked by any piece" when there is no attacker and "undefended" when there is no
defender (no colour is named).

Label balance is over the thirteen categories — EMPTY plus the twelve occupant
tokens (six own + six opponent). Empty squares take ~50% of the queries; the other
~50% is spread evenly across the twelve occupant types (frequency-shaped).

parse_tag: `<prefix> <attackers> <defenders>` where the prefix is `<square>` for an
empty square and `<piece><square>` for an occupied one, each list being the
spaceless `prose.format_piece_squares` flattening (the defender list is omitted for
empty squares). Graded by utils.eval_utils._attacker_defender_grade.
"""
import random
from collections import Counter

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN
from datagen.position_features import get_attackers_and_defenders
from datagen.prose import format_piece_squares, format_piece_square_list

NAME = "square_attackers"
MAX_UNIQUE_QUERIES = 64
EMPTY_RATIO = 0.5    # empty squares take ~half of the queries
# Forward (stage-4) capable. This is the one task whose prompt must change for
# lookahead: the question can't name the occupant (the model can't see the final
# board), so forward questions are posed purely on the square — "what attacks or
# defends <sq>, or the piece on it?" — and the answer follows final-board
# occupancy (attackers-only when empty, attackers+defenders when occupied).
SUPPORTS_FORWARD = True

EMPTY_QUESTIONS = [
    "Which pieces attack {sq}?",
    "List all pieces that attack {sq}.",
    "What pieces can attack {sq}?",
    "Which pieces are attacking {sq}?",
]

OCC_QUESTIONS = [
    "Which pieces attack or defend {piece} on {sq}?",
    "What attacks or defends {piece} on {sq}?",
    "List the attackers and defenders of {piece} on {sq}.",
    "Which pieces attack or defend the {piece} on {sq}?",
]

# Forward (lookahead) questions — posed on the square only, since the final
# occupant is unknown to the model. The "or the piece on it" clause covers the
# occupied case; for an empty final square it is simply vacuous.
FORWARD_QUESTIONS = [
    "What pieces attack or defend {sq}, or the piece on it?",
    "Which pieces attack or defend {sq}, or whatever piece occupies it?",
    "List the pieces that attack or defend {sq}, or the piece on it.",
    "What attacks or defends {sq}, or any piece standing on it?",
]


def _items(board: BoardRepr, squares) -> list[tuple[str, str]]:
    """(piece_tok, sq_tok) pairs for board squares; the prose/tag flatteners impose
    the canonical heavier-piece-first then ascending-square order."""
    return [(board.piece_at(s), board.sq_tok(s)) for s in squares]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _answer_empty(sq_tok: str, att: list, rng: random.Random) -> tuple[str, str, list]:
    if att:
        prose = format_piece_square_list(att)
        a = rng.choice([f"{sq_tok} is attacked by {prose}.",
                        f"The attackers of {sq_tok} are {prose}."])
    else:
        a = rng.choice([f"{sq_tok} is not attacked by any piece.",
                        f"No piece attacks {sq_tok}."])
    parse_tag = f"{sq_tok} {format_piece_squares(att)}"   # prefix = square; no defenders
    return a, parse_tag, [EMPTY_TOKEN]


def _answer_occupied(sq_tok: str, occ_tok: str, att: list, dfd: list,
                     rng: random.Random) -> tuple[str, str, list]:
    att_clause = f"attacked by {format_piece_square_list(att)}" if att else "not attacked by any piece"
    dfd_clause = f"defended by {format_piece_square_list(dfd)}" if dfd else "undefended"
    a = rng.choice([
        f"The {occ_tok} on {sq_tok} is {att_clause} and {dfd_clause}.",
        f"The {occ_tok} on {sq_tok} is {att_clause}. It is {dfd_clause}.",
        f"On {sq_tok}, the {occ_tok} is {att_clause} and {dfd_clause}.",
    ])
    parse_tag = f"{occ_tok}{sq_tok} {format_piece_squares(att)} {format_piece_squares(dfd)}"
    return a, parse_tag, [occ_tok]


def _render(entity, board: BoardRepr, rng: random.Random) -> dict:
    # Static (stage-2): `entity` is a square int; the question framing depends on
    # occupancy and names the occupant. Forward (stage-4): `entity` is a 1-tuple
    # (square,) — the question is posed purely on the SQUARE (the occupant on the
    # final board is unknown to the model), while the answer still follows
    # final-board occupancy: attackers-only when empty, attackers+defenders when
    # occupied. Question choice precedes answer choice in both paths, so the
    # static rng draw order (hence its output) is unchanged.
    forward = isinstance(entity, tuple)
    sq = entity[0] if forward else entity
    cb = board.chess_board
    sq_tok = board.sq_tok(sq)
    attackers, defenders = get_attackers_and_defenders(cb, sq)
    att = _items(board, [s for _, s in attackers])
    dfd = _items(board, [s for _, s in defenders])
    occupied = cb.piece_at(sq) is not None

    if forward:
        q = rng.choice(FORWARD_QUESTIONS).format(sq=sq_tok)
    elif occupied:
        q = rng.choice(OCC_QUESTIONS).format(piece=board.piece_at(sq), sq=sq_tok)
    else:
        q = rng.choice(EMPTY_QUESTIONS).format(sq=sq_tok)

    if occupied:
        a, parse_tag, ac = _answer_occupied(sq_tok, board.piece_at(sq), att, dfd, rng)
    else:
        a, parse_tag, ac = _answer_empty(sq_tok, att, rng)

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  ac,
    }


# ---------------------------------------------------------------------------
# Entity selection — 50% empty, 50% spread evenly over the 12 occupant types
# (the attacked/unattacked split is left to fall where it may, to preserve the
# type balance — attacked pieces are pawn-heavy, so balancing it would skew types)
# ---------------------------------------------------------------------------

def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[int]) -> int:
    cb = board.chess_board
    avail = [sq for sq in range(64) if sq not in exclude]
    empty = [sq for sq in avail if cb.piece_at(sq) is None]
    occ   = [sq for sq in avail if cb.piece_at(sq) is not None]

    # Empty is a single label taking ~half the mass (chosen uniformly); the other
    # half is balanced across the twelve occupant tokens via frequency and a
    # per-board multiplicity correction (so abundant species don't crowd out rare).
    if (rng.random() < EMPTY_RATIO and empty) or not occ:
        return rng.choice(empty if empty else occ)

    toks = [board.piece_at(s) for s in occ]
    mult = Counter(toks)
    weights = [1.0 / ((frequency.get(t, 0) + 1) * mult[t]) for t in toks]
    return rng.choices(occ, weights=weights, k=1)[0]


def _choose_entity_forward(board: BoardRepr, final: BoardRepr, move_sequence,
                           frequency: dict, rng: random.Random,
                           exclude: set) -> tuple | None:
    """Pick the queried square on the FINAL board (no backtracking — a square is a
    fixed coordinate, unlike a piece). The 50% empty / 50%-over-occupants balance
    is measured on the final board, since that is where the answer and its
    empty/occupied shape are computed. Returns a 1-tuple (square,) marking the
    forward variant for _render, or None once every square has been used."""
    cb = final.chess_board
    used = {e[0] for e in exclude}
    avail = [sq for sq in range(64) if sq not in used]
    if not avail:
        return None
    empty = [sq for sq in avail if cb.piece_at(sq) is None]
    occ   = [sq for sq in avail if cb.piece_at(sq) is not None]

    if (rng.random() < EMPTY_RATIO and empty) or not occ:
        return (rng.choice(empty if empty else occ),)

    toks = [final.piece_at(s) for s in occ]
    mult = Counter(toks)
    weights = [1.0 / ((frequency.get(t, 0) + 1) * mult[t]) for t in toks]
    return (rng.choices(occ, weights=weights, k=1)[0],)


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    n = min(n, MAX_UNIQUE_QUERIES)
    seen: set[int] = set()
    out: list[dict] = []
    while len(out) < n:
        sq = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(sq)
        out.append(_render(sq, board, rng))
    return out


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
