"""Task: capture_moves — list every capture the side to move can make.

Stage-2 dynamic task. Deterministic: one question per position
(MAX_UNIQUE_QUERIES = 1, nothing to balance over, so `_choose_entity` returns
None). Capturing moves come from position_features.capture_moves and are
CoT-verbalized and parse-tagged exactly like piece_moves / checking_moves
(prose.format_move_cot_list / format_move_tag_list — same ordering: checkmate >
check-capture > check, then piece value, from-square, to-square). En passant and
capture-promotions are included; castling is never a capture.

There is no single queried piece, so the parse tag is anchored only by the side's
own king token (no square). parse_tag: `<side-king> <move>,<move>,...` (or just
`<side-king>` when there is no capture). Graded by utils.eval_utils._move_list_grade
(shared with piece_moves / checking_moves).
"""
import random

import chess

from utils.board_representation import BoardRepr
from datagen.position_features import capture_moves, mover_label
from datagen.prose import format_move_cot_list, format_move_tag_list

NAME = "capture_moves"
MAX_UNIQUE_QUERIES = 1
# Forward (stage-4) capable. No entity (one question/position); _render answers on
# the final board. The tag's king anchor is taken turn-aware (piece_tok_for on the
# side to move) so it is the opponent's king after an odd-length sequence, matching
# the capturing side. mover_label supplies "player"/"opponent".
SUPPORTS_FORWARD = True

QUESTIONS = [
    "List all captures that {mover} can make.",
    "What captures can {mover} make?",
    "Which captures are available to {mover}?",
    "List the capturing moves for {mover}.",
]

INTROS = [
    "{Mover} can capture with {captures}.",
    "The captures available to {mover} are {captures}.",
    "{Mover} has the following captures: {captures}.",
    "{Mover} can play the following captures: {captures}.",
]

NONE_ANSWERS = [
    "{Mover} has no captures available.",
    "There are no captures for {mover}.",
    "{Mover} cannot capture anything in this position.",
]


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    # Side-to-move king anchors the tag. Turn-aware (not own_piece_tok, which is
    # anchor-fixed to M) so after an odd-length forward sequence it is the
    # opponent's king — the side actually capturing. Identical for static boards.
    king_tok = board.piece_tok_for(board.chess_board.turn, chess.KING)
    mover = mover_label(board)
    Mover = mover[:1].upper() + mover[1:]

    moves = capture_moves(board)
    q = rng.choice(QUESTIONS).format(mover=mover)

    if not moves:
        a = rng.choice(NONE_ANSWERS).format(mover=mover, Mover=Mover)
        parse_tag = king_tok
    else:
        a = rng.choice(INTROS).format(mover=mover, Mover=Mover,
                                      captures=format_move_cot_list(moves, rng))
        parse_tag = f"{king_tok} {format_move_tag_list(moves)}"

    return {
        "question":      q,
        "answer":        f"{a}\n\n{parse_tag}",
        "question_type": NAME,
        "answer_class":  None,
    }


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random, exclude: set) -> None:
    return None


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    return [_render(None, board, rng)]


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
