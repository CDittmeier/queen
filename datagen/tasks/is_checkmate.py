"""Task: is_checkmate — decide whether the position is checkmate, and say why.

Stage-2 dynamic task. One question per position (MAX_UNIQUE_QUERIES = 1). The
intended position mix (set by the build's source interleave) is 30% checkmates,
20% near-mate checks, 10% random checks, 20% stalemates, 20% random positions.

The CoT picks ONE explanation, in priority order:
  no-check  — not in check: not checkmate (and if there is also no legal move, it
              is stalemate, which is called out).
  capture   — in check, but the checking piece can be captured (named: checker +
              capturer); not checkmate.
  block     — in check, but the check can be blocked (one example move); not mate.
  king-move — in check, but the king can step to safety (one example move); not mate.
  checkmate — in check with no parry: every king-adjacent square (in increasing
              square-number order) is either occupied by an own piece or attacked
              by an opponent piece (computed with the king lifted off its square so
              a checker covering a retreat square along its line is seen).
Move descriptions use prose.format_move_cot; each branch has a few templates.

(a) in-check and (b) any-legal-move come from position_features.in_check / all_moves.

parse_tag: `<king><king-square> yes` (checkmate) or `<king><king-square> no`.
Graded by exact match.
"""
import random

from utils.board_representation import BoardRepr
from datagen.position_features import in_check, all_moves, mover_label
from datagen.prose import format_move_cot, format_mate_adjacency

NAME = "is_checkmate"
MAX_UNIQUE_QUERIES = 1
# Forward (stage-4) capable. No entity; _render answers on the final board, where
# the king anchor (board.piece_at(cb.king(cb.turn)) — turn-derived, no
# own_piece_tok fix), in_check, all_moves, format_mate_adjacency, and mover_label
# are all turn-correct. The yes/no verdict BALANCE comes entirely from the source
# mix, so the stage-4 build must assemble it from the forward_* pools
# (forward_checkmate / forward_check_near_mate / forward_check / forward_stalemate
# / forward), mirroring stage-2's is_checkmate interleave. No code balance here.
SUPPORTS_FORWARD = True

QUESTIONS = [
    "Is this position checkmate?",
    "Is it checkmate?",
    "Is the side to move checkmated?",
    "Has {mover} been checkmated?",
    "Is {mover} in checkmate?",
]

NOTCHECK_TEMPLATES = [
    "The {king} at {ksq} is not in check, so it is not checkmate.",
    "No — the {king} at {ksq} is not in check, hence not checkmate.",
    "The {king} at {ksq} is not in check; it cannot be checkmate.",
]

STALEMATE_TEMPLATES = [
    "The {king} at {ksq} is not in check, but there are no legal moves, so it is stalemate.",
    "The {king} at {ksq} is not in check yet has no legal move — it is stalemate, not checkmate.",
]

CAPTURE_TEMPLATES = [
    "The {king} at {ksq} is in check, but the checking {checker} at {csq} can be "
    "captured by the {capturer} at {cfrom}, so it is not checkmate.",
    "This is check, but the checking {checker} at {csq} can be taken by the "
    "{capturer} at {cfrom} — not checkmate.",
]

BLOCK_TEMPLATES = [
    "The {king} at {ksq} is in check, but it can be blocked: {move}. So it is not checkmate.",
    "This is check, but the check can be interposed by {move} — not checkmate.",
]

KING_TEMPLATES = [
    "The {king} at {ksq} is in check, but the king can move to safety: {move}. Not checkmate.",
    "This is check, but the king can escape: {move} — not checkmate.",
]

MATE_TEMPLATES = [
    "Yes — the {king} at {ksq} is checkmated. {adj}.",
    "This is checkmate: the {king} at {ksq} has no escape. {adj}.",
]


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    cb = board.chess_board
    mover = mover_label(board)
    king_sq = cb.king(cb.turn)
    king_tok = board.piece_at(king_sq)
    ksq_tok = board.sq_tok(king_sq)
    prefix = f"{king_tok}{ksq_tok}"
    q = rng.choice(QUESTIONS).format(mover=mover)
    moves = all_moves(board)

    if not in_check(cb):
        if moves:
            cls, verdict = "not_check", "no"
            a = rng.choice(NOTCHECK_TEMPLATES).format(king=king_tok, ksq=ksq_tok)
        else:
            cls, verdict = "stalemate", "no"
            a = rng.choice(STALEMATE_TEMPLATES).format(king=king_tok, ksq=ksq_tok)
    else:
        checker_sqs = {board.sq_tok(c) for c in cb.checkers()}
        caps, blocks, kings = [], [], []
        for m in moves:
            if m["to_sq"] in checker_sqs or m["en_passant_square"] in checker_sqs:
                caps.append(m)
            elif m["from_sq"] == ksq_tok:
                kings.append(m)
            else:
                blocks.append(m)

        if caps:                                       # priority: capture > block > king-move
            cls, verdict = "check_parry", "no"
            m = rng.choice(caps)
            csq = m["en_passant_square"] or m["to_sq"]
            a = rng.choice(CAPTURE_TEMPLATES).format(
                king=king_tok, ksq=ksq_tok, checker=m["captured_piece"], csq=csq,
                capturer=m["piece"], cfrom=m["from_sq"])
        elif blocks:
            cls, verdict = "check_parry", "no"
            a = rng.choice(BLOCK_TEMPLATES).format(
                king=king_tok, ksq=ksq_tok, move=format_move_cot(rng.choice(blocks), rng))
        elif kings:
            cls, verdict = "check_parry", "no"
            a = rng.choice(KING_TEMPLATES).format(
                king=king_tok, ksq=ksq_tok, move=format_move_cot(rng.choice(kings), rng))
        else:
            cls, verdict = "checkmate", "yes"
            a = rng.choice(MATE_TEMPLATES).format(
                king=king_tok, ksq=ksq_tok, adj=format_mate_adjacency(board))

    return {
        "question":      q,
        "answer":        f"{a}\n\n{prefix} {verdict}",
        "question_type": NAME,
        "answer_class":  None,
    }


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random, exclude: set) -> None:
    return None


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    return [_render(None, board, rng)]


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
