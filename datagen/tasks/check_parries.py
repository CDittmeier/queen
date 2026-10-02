"""Task: check_parries — list every way to get out of the current check.

Stage-2 dynamic task, sampled ONLY on positions where the side to move is in
check (build filters via the have-check / in_check position source). Deterministic:
one question per position (MAX_UNIQUE_QUERIES = 1).

Every legal move (position_features.all_moves) is a parry; each is classed as:

  (b) capture the checking piece — lands on a checker's square, or en-passant
                                   captures a checking pawn (a king capture of the
                                   checker counts here, not as a king move)
  (c) block the check            — interpose on the checking line (a non-king move
                                   that doesn't capture the checker)
  (a) move the king away         — any other king move

The CoT names the checker(s), then verbalizes each non-empty class with its own
templates, each move list rendered by prose.format_move_cot_list. A double check
is called out up front (both checkers named); a checkmate (no legal move) is
stated explicitly. Empty classes are simply not mentioned.

parse_tag: `<king><king-square> <moves>` — one unified, comma-separated move list
in class order capture-checker -> block -> king-move (each class internally ordered
and formatted by the shared move helpers; no space between classes), or just
`<king><king-square>` at checkmate. Graded by utils.eval_utils._move_list_grade.
"""
import random

from utils.board_representation import BoardRepr
from datagen.position_features import all_moves, mover_label
from datagen.prose import join_and, format_move_cot_list, format_move_tag_list

NAME = "check_parries"
MAX_UNIQUE_QUERIES = 1
# Forward (stage-4) capable. No entity; _render answers on the final board, where
# the king anchor (board.piece_at(cb.king(cb.turn)) — already turn-derived, so no
# own_piece_tok fix is needed), checkers, all_moves, and mover_label are all
# turn-correct. PREMISE: the side to move must be in check, so the stage-4 build
# must source this from the forward_check (in_check-on-final) pool — exactly as
# stage-2 sourced it from check. _render does not guard this, same as stage-2.
SUPPORTS_FORWARD = True

QUESTIONS = [
    "How can {mover} get out of check?",
    "List all ways to parry the check.",
    "What are {mover}'s legal responses to this check?",
    "How can {mover} escape the check?",
]

INTROS = [
    "The {king} at {ksq} is under check by {checkers}.",
    "{Mover}'s {king} at {ksq} is in check from {checkers}.",
]

DOUBLE_INTROS = [
    "The {king} at {ksq} is being checked by two pieces: {checkers}.",
    "{Mover}'s {king} at {ksq} is in double check from two pieces: {checkers}.",
]

MATE_ANSWERS = [
    "There is no legal response — it is checkmate.",
    "It is checkmate; there is no way out.",
    "No move escapes the check: it is checkmate.",
]

CAP_TEMPLATES   = ["The checking piece can be captured by {x}.",
                   "It can be answered by capturing the checker: {x}.",
                   "The checker can be taken by {x}."]
KING_TEMPLATES  = ["The king can move: {x}.",
                   "The king can escape by {x}.",
                   "The king can step away via {x}."]
BLOCK_TEMPLATES = ["The check can be blocked by {x}.",
                   "The check can be interposed by {x}.",
                   "It can be blocked: {x}."]


def _render(_: None, board: BoardRepr, rng: random.Random) -> dict:
    cb = board.chess_board
    mover = mover_label(board)
    Mover = mover[:1].upper() + mover[1:]
    king_sq = cb.king(cb.turn)
    king_tok = board.piece_at(king_sq)
    ksq_tok = board.sq_tok(king_sq)
    checkers = sorted(cb.checkers())
    double = len(checkers) >= 2

    checker_join = join_and([f"the {board.piece_at(c)} at {board.sq_tok(c)}" for c in checkers])
    intros = DOUBLE_INTROS if double else INTROS
    intro = rng.choice(intros).format(king=king_tok, ksq=ksq_tok, checkers=checker_join, Mover=Mover)
    q = rng.choice(QUESTIONS).format(mover=mover)
    prefix = f"{king_tok}{ksq_tok}"

    moves = all_moves(board)
    if not moves:                       # in check with no legal move == checkmate
        a = intro + " " + rng.choice(MATE_ANSWERS)
        return {"question": q, "answer": f"{a}\n\n{prefix}",
                "question_type": NAME, "answer_class": None}

    checker_sqs = {board.sq_tok(c) for c in checkers}
    caps, blocks, kings = [], [], []
    for m in moves:
        if m["to_sq"] in checker_sqs or m["en_passant_square"] in checker_sqs:
            caps.append(m)            # capture of a checking piece (king or otherwise)
        elif m["from_sq"] == ksq_tok:
            kings.append(m)           # king stepping away
        else:
            blocks.append(m)          # interposing on the checking line

    sentences = []
    if caps:
        sentences.append(rng.choice(CAP_TEMPLATES).format(x=format_move_cot_list(caps, rng)))
    if blocks:
        sentences.append(rng.choice(BLOCK_TEMPLATES).format(x=format_move_cot_list(blocks, rng)))
    if kings:
        sentences.append(rng.choice(KING_TEMPLATES).format(x=format_move_cot_list(kings, rng)))
    a = intro + " " + " ".join(sentences)

    # Unified move list in class order capture -> block -> king; each class is
    # internally ordered/formatted by format_move_tag_list, classes joined with no
    # extra space (one comma-separated list).
    groups = [format_move_tag_list(caps), format_move_tag_list(blocks), format_move_tag_list(kings)]
    parse_tag = f"{prefix} " + ",".join(g for g in groups if g)
    return {"question": q, "answer": f"{a}\n\n{parse_tag}",
            "question_type": NAME, "answer_class": None}


def _choose_entity(board: BoardRepr, frequency: dict, rng: random.Random, exclude: set) -> None:
    return None


def sample_n(board: BoardRepr, frequency: dict, rng: random.Random, n: int) -> list[dict]:
    return [_render(None, board, rng)]


# `sample_one` and `sample_all` are synthesized in `datagen/tasks/__init__.py`.
