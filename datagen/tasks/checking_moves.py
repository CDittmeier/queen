"""Task: checking_moves — name every check the side to move can give.

Stage-2 dynamic task. Deterministic: one question per position
(MAX_UNIQUE_QUERIES = 1, nothing to balance over, so `_choose_entity` returns
None). The side to move is the checking side. Checking moves come from
position_features.check_moves and are CoT-verbalized and parse-tagged exactly like
piece_moves (prose.format_move_cot_list / format_move_tag_list — same ordering:
checkmate > check-capture > check, then piece value, from-square, to-square). Both
the CoT and the parse tag name the opponent king and its square.

parse_tag: `<opp-king><king-square> <move>,<move>,...` (or just
`<opp-king><king-square>` when no check is available). Graded by
utils.eval_utils._move_list_grade (shared with piece_moves).
"""
import random

from utils.board_representation import BoardRepr
from datagen.position_features import check_moves, mover_label
from datagen.prose import format_move_cot_list, format_move_tag_list

NAME = "checking_moves"
MAX_UNIQUE_QUERIES = 1
# Forward (stage-4) capable. No entity (one question per position) and no prompt
# change: _render answers on the final board, where check_moves / the opponent
# king / mover_label are all turn-correct (mover_label names "player" or
# "opponent" by the final side to move).
SUPPORTS_FORWARD = True

QUESTIONS = [
    "What checks can {mover} give?",
    "List all checking moves for {mover}.",
    "How can {mover} give check?",
    "Which moves deliver check in this position?",
]

INTROS = [
    "The {king} on {ksq} can be checked by {checks}.",
    "{Mover} can check the {king} on {ksq} with {checks}.",
    "The available checks against the {king} on {ksq} are {checks}.",
]

NONE_ANSWERS = [
    "The {king} on {ksq} cannot be put in check.",
    "{Mover} has no checking move available.",
    "There are no checks available against the {king} on {ksq}.",
]


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    cb = board.chess_board
    king_sq = cb.king(not cb.turn)
    king_tok = board.piece_at(king_sq)
    ksq_tok = board.sq_tok(king_sq)
    component = f"{king_tok}{ksq_tok}"   # opponent king + its square anchors the query
    mover = mover_label(board)
    Mover = mover[:1].upper() + mover[1:]

    moves = check_moves(board)
    q = rng.choice(QUESTIONS).format(mover=mover)

    if not moves:
        a = rng.choice(NONE_ANSWERS).format(king=king_tok, ksq=ksq_tok, Mover=Mover)
        parse_tag = component
    else:
        a = rng.choice(INTROS).format(king=king_tok, ksq=ksq_tok, Mover=Mover,
                                      checks=format_move_cot_list(moves, rng))
        parse_tag = f"{component} {format_move_tag_list(moves)}"

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
