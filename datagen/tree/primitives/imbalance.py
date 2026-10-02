"""IMBALANCE term (material.cpp): second-degree polynomial material imbalance.

This term is intrinsically a whole-army quadratic (each piece count interacts with
every other), so it does not decompose per-square; it is a single leaf whose weight
tables (QuadraticOurs / QuadraticTheirs) live here.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import get_context
from datagen.tree.primitives.core.score import SCORE_ZERO, S, Score, cdiv
from datagen.tree.primitives.base import Factor

# QuadraticOurs / QuadraticTheirs[pt1][pt2], lower triangle (pt2 <= pt1).
# Index: 0 = bishop-pair flag, then PAWN..QUEEN.
QUADRATIC_OURS = {
    0: [1438],
    1: [40, 38],
    2: [32, 255, -62],
    3: [0, 104, 4, 0],
    4: [-26, -2, 47, 105, -208],
    5: [-189, 24, 117, 133, -134, -6],
}
QUADRATIC_THEIRS = {
    0: [0],
    1: [36, 0],
    2: [9, 63, 0],
    3: [59, 65, 42, 0],
    4: [46, 39, 24, -24, 0],
    5: [97, 100, -42, 137, 268, 0],
}


def piece_counts(board: chess.Board, color: chess.Color) -> list[int]:
    nb = len(board.pieces(chess.BISHOP, color))
    return [1 if nb > 1 else 0,
            len(board.pieces(chess.PAWN, color)),
            len(board.pieces(chess.KNIGHT, color)),
            nb,
            len(board.pieces(chess.ROOK, color)),
            len(board.pieces(chess.QUEEN, color))]


def imbalance_one(us: list[int], them: list[int]) -> int:
    bonus = 0
    for pt1 in range(6):
        if not us[pt1]:
            continue
        v = 0
        for pt2 in range(pt1 + 1):
            v += QUADRATIC_OURS[pt1][pt2] * us[pt2] + QUADRATIC_THEIRS[pt1][pt2] * them[pt2]
        bonus += us[pt1] * v
    return bonus


class Imbalance(Factor):
    tag = "imbalance"
    KIND = "whole"

    def score(self, board, ctx=None):
        pcw = piece_counts(board, chess.WHITE)
        pcb = piece_counts(board, chess.BLACK)
        v = cdiv(imbalance_one(pcw, pcb) - imbalance_one(pcb, pcw), 16)
        return S(v, v), {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        us = chess.WHITE if mover == "w" else chess.BLACK
        return {self.tag: v_imbalance(before, after, us, self.whole_direction(mover, before_eval, after_eval)) or None}



# ============================================================================== verbalization primitives
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ)


def elu_imbalance(parent_fen, child_fen, mover, sign):
    """The imbalance term is a whole-army quadratic; its change on a move is driven
    entirely by the piece-count change. Report bishop-pair flips (the single most
    interpretable driver), else the piece whose count changed."""
    pb, cb = chess.Board(parent_fen), chess.Board(child_fen)
    notes = []                                        # level 1: bishop-pair flips
    for color in (chess.WHITE, chess.BLACK):
        had = len(pb.pieces(chess.BISHOP, color)) > 1
        now = len(cb.pieces(chess.BISHOP, color)) > 1
        who = "White" if color == chess.WHITE else "Black"
        if had and not now:
            notes.append(f"{who} loses the bishop pair")
        elif now and not had:
            notes.append(f"{who} gains the bishop pair")
    if not notes:                                     # level 2: the count change that drove it
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            for color in (chess.WHITE, chess.BLACK):
                dn = len(cb.pieces(pt, color)) - len(pb.pieces(pt, color))
                if dn:
                    who = "White" if color == chess.WHITE else "Black"
                    notes.append(f"{who} {'gains' if dn > 0 else 'loses'} a {chess.piece_name(pt)}")
    return ", ".join(notes[:2])


def v_imbalance(pf, cf, mover, sign):
    """A plain piece-count change is already implied by the move's capture clause, so
    the imbalance term speaks up only for the interpretable case it adds: a bishop-pair
    flip ('deprives White of the bishop pair')."""
    pb, cb = chess.Board(pf), chess.Board(cf)
    for color in (chess.WHITE, chess.BLACK):
        had = len(pb.pieces(chess.BISHOP, color)) > 1
        now = len(cb.pieces(chess.BISHOP, color)) > 1
        name = "White" if color == chess.WHITE else "Black"
        if had and not now:
            return f"deprives {name} of the bishop pair"
        if now and not had:
            return f"gives {name} the bishop pair"
    return ""


# direction-preserving verbs for king-danger templates (do NOT re-derive from sign:
# a positive mover-POV delta can mean OUR king got safer, not that we attack theirs)



