"""Task: piece_moves — list every legal move of a queried side-to-move piece.

Stage-2 dynamic task. The entity is a *specific* own piece, identified by its
square (several pieces of a species can coexist with different moves). All board
state comes from `position_features`: `moves_from(board, sq)` returns the piece's
legal moves as token-resolved move dicts. The chain of thought verbalizes that
move list with `prose.format_move_cot_list` dropped into one of several answer
templates; the parse tag is the queried piece + its square, a space, then the
moves flattened by `prose.format_move_tag_list`. Both the CoT and the tag order
moves identically (checks before captures before quiet; see prose._move_order_key).

Class balance is over own piece species (frequency-shaped, mirroring
piece_on_square) with a per-species multiplicity correction so abundant species
(e.g. eight pawns) don't crowd out rarer ones. Rare *move types* (castling,
en passant, promotion, check) are NOT balanced here — their prevalence is set
upstream by the position-source mix, not by in-task boosts.

parse_tag: `<piece><square> <move>,<move>,...` (or just `<piece><square>` when the
piece has no legal move). Graded by `utils.eval_utils._move_list_grade`.
"""
import random
from collections import defaultdict

from utils.board_representation import BoardRepr
from datagen.position_features import moves_from, origin_squares
from datagen.prose import format_move_cot_list, format_move_tag_list

NAME = "piece_moves"
# Max distinct queries per position = number of side-to-move pieces (<= 16).
# sample_n additionally caps to the actual count for the position.
MAX_UNIQUE_QUERIES = 16
# Forward (stage-4) capable: see _choose_entity_forward + the tuple branch of
# _render. No distinctness balancing (the answer changes very often already).
SUPPORTS_FORWARD = True

QUESTIONS_TOK = [
    "What moves can {piece_tok} on {sq_tok} make?",
    "Which moves are available to {piece_tok} on {sq_tok}?",
    "List the legal moves for {piece_tok} on {sq_tok}.",
    "Where can {piece_tok} on {sq_tok} go?",
]

# Each template wraps the flattened CoT move list ({moves}); the moves themselves
# are full clauses ("<piece> on <from> ... to <to>"), so the template just frames
# the list and supplies the closing period.
ANSWER_TEMPLATES = [
    "Here are the legal moves for {piece_tok} on {sq_tok}: {moves}.",
    "The {piece_tok} on {sq_tok} has these moves: {moves}.",
    "Considering {piece_tok} on {sq_tok}: {moves}.",
    "{piece_tok} on {sq_tok} can play the following: {moves}.",
]

NO_MOVE_ANSWERS_TOK = [
    "The {piece_tok} on {sq_tok} has no legal moves in this position.",
    "There are no legal moves for {piece_tok} on {sq_tok}.",
]


# ---------------------------------------------------------------------------
# Entity selection — balance over own piece species
# ---------------------------------------------------------------------------

def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random,
                   exclude: set[int]) -> int:
    """Pick an own square, balancing piece species: downweight species already
    frequent (frequency) and species with many instances on the board (so eight
    pawns don't dominate the six other species)."""
    entities = [sq for sq in board.own_squares() if sq not in exclude]
    piece_toks = [board.piece_at(sq) for sq in entities]

    mult: dict[str, int] = defaultdict(int)
    for p in piece_toks:
        mult[p] += 1

    weights = [1.0 / ((frequency.get(p, 0) + 1) * mult[p]) for p in piece_toks]
    return rng.choices(entities, weights=weights, k=1)[0]


def _choose_entity_forward(board: BoardRepr, final: BoardRepr, move_sequence,
                           frequency: dict, rng: random.Random,
                           exclude: set) -> tuple[int, int] | None:
    """Pick the queried piece on the FINAL board, but identify it by its INITIAL
    square (the model sees only the initial board and must look ahead). We query
    the side to MOVE on the final board — the only side whose legal moves are
    defined — so after an odd-length sequence the queried piece is an opponent
    piece, which is intended. The entity is the (initial_sq, final_sq) pair; the
    initial square comes from backtracking the final piece through the sequence.

    Pieces that promoted along the way are skipped: their species differs between
    the two boards, so naming them by the initial square would mislabel them.
    Species balance (frequency + multiplicity) matches the static chooser, over
    the final-board occupants. Returns None when no fresh piece remains."""
    origin = origin_squares(board.chess_board, move_sequence)  # final_sq -> initial_sq
    stm = final.chess_board.turn

    entities: list[tuple[int, int]] = []
    for sq_f in final.order:
        pc = final.chess_board.piece_at(sq_f)
        if pc is None or pc.color != stm:
            continue
        sq_i = origin.get(sq_f)
        if sq_i is None:
            continue
        if board.chess_board.piece_at(sq_i).piece_type != pc.piece_type:
            continue                                    # promoted: species changed
        e = (sq_i, sq_f)
        if e not in exclude:
            entities.append(e)
    if not entities:
        return None

    piece_toks = [final.piece_at(sq_f) for _, sq_f in entities]
    mult: dict[str, int] = defaultdict(int)
    for p in piece_toks:
        mult[p] += 1
    weights = [1.0 / ((frequency.get(p, 0) + 1) * mult[p]) for p in piece_toks]
    return rng.choices(entities, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render(sq, board: BoardRepr, rng: random.Random) -> dict:
    # Static (stage-2): `sq` is one square, used for both the question and the
    # answer. Forward (stage-4): `sq` is an (initial_sq, final_sq) pair — the
    # question names the INITIAL square (ask_tok) the model is shown, while the
    # piece, its current square (from_tok) and its moves are read off the FINAL
    # board where the piece has travelled to.
    if isinstance(sq, tuple):
        sq_i, sq_f = sq
        ask_tok = board.sq_tok(sq_i)
    else:
        sq_f = sq
        ask_tok = board.sq_tok(sq_f)
    piece_tok = board.piece_at(sq_f)
    from_tok  = board.sq_tok(sq_f)
    moves     = moves_from(board, sq_f)
    component = f"{piece_tok}{from_tok}"   # anchors the piece on the answered board

    q = rng.choice(QUESTIONS_TOK).format(piece_tok=piece_tok, sq_tok=ask_tok)

    if not moves:
        a = rng.choice(NO_MOVE_ANSWERS_TOK).format(piece_tok=piece_tok, sq_tok=from_tok)
        parse_tag = component
    else:
        cot = format_move_cot_list(moves, rng)
        a = rng.choice(ANSWER_TEMPLATES).format(piece_tok=piece_tok, sq_tok=from_tok, moves=cot)
        parse_tag = f"{component} {format_move_tag_list(moves)}"

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  [piece_tok],
    }


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    # Cap to the number of own pieces (varies per position) so we never request
    # more distinct entities than exist.
    entities = board.own_squares()
    n = min(n, MAX_UNIQUE_QUERIES, len(entities))

    seen: set[int] = set()
    out: list[dict] = []
    while len(out) < n:
        sq = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(sq)
        out.append(_render(sq, board, rng))
    return out


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
