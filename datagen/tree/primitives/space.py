"""SPACE term (evaluate.cpp): safe squares behind the pawn front in the centre.

Space -> one leaf per side (White/Black, signed). The term is a whole-side
count weighted by a piece-count weight, so it does not decompose per-square.
"""
from __future__ import annotations

import chess
import re

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO, cdiv
from datagen.tree.primitives.base import Factor

SpaceThreshold = 12222


def space_term(board: chess.Board, ctx: Context, us: chess.Color) -> Score:
    them = not us
    if ctx.non_pawn_material(chess.WHITE) + ctx.non_pawn_material(chess.BLACK) < SpaceThreshold:
        return SCORE_ZERO
    own_pawns = int(board.pawns & board.occupied_co[us])
    if us == chess.WHITE:
        mask = bb.CENTER_FILES & (chess.BB_RANKS[1] | chess.BB_RANKS[2] | chess.BB_RANKS[3])
    else:
        mask = bb.CENTER_FILES & (chess.BB_RANKS[6] | chess.BB_RANKS[5] | chess.BB_RANKS[4])
    safe = mask & bb.bb_not(own_pawns) & bb.bb_not(ctx.att[them][chess.PAWN])
    behind = own_pawns
    behind |= bb.shift_down(us, behind)
    behind |= bb.shift_down(us, bb.shift_down(us, behind))
    bonus = bb.popcount(safe) + bb.popcount(behind & safe & bb.bb_not(ctx.att[them]["ALL"]))
    weight = bb.popcount(board.occupied_co[us]) - 1
    return S(cdiv(bonus * weight * weight, 16), 0)



class Space(Factor):
    tag = "space"
    KIND = "whole"
    ACTIVITY = ("space",)

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        return space_term(board, ctx, chess.WHITE) - space_term(board, ctx, chess.BLACK), {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        us = chess.WHITE if mover == "w" else chess.BLACK
        return {self.tag: v_space(before, after, us, self.whole_direction(mover, before_eval, after_eval)) or None}




# ============================================================================== verbalization primitives
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives import space as space_mod
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, derive_move, pcsym)


def elu_space(parent_fen, child_fen, mover, sign):
    cands = []
    for col in (chess.WHITE, chess.BLACK):
        dp = space_mod.space_term(chess.Board(parent_fen), get_context(chess.Board(parent_fen)), col).mg
        dc = space_mod.space_term(chess.Board(child_fen), get_context(chess.Board(child_fen)), col).mg
        d = dc - dp
        mb = d if col == mover else -d
        if mb != 0:
            cands.append((mb, col, d))
    cands = [c for c in cands if (c[0] > 0) == (sign > 0)]
    if not cands:
        return ""
    cands.sort(key=lambda z: -abs(z[0]))
    _, col, d = cands[0]
    name = "White" if col == chess.WHITE else "Black"
    detail = ""
    m = derive_move(parent_fen, child_fen)
    if m is not None:
        pc = chess.Board(parent_fen).piece_at(m.from_square)
        if pc and pc.piece_type == chess.PAWN and (m.to_square & 7) in (2, 3, 4, 5):
            detail = f" ({chess.square_name(m.from_square)}→{chess.square_name(m.to_square)})"
    return f"{'more' if d > 0 else 'less'} central space for {name}{detail}"


def v_space(pf, cf, mover, sign):
    d = elu_space(pf, cf, mover, sign)
    if not d:
        return ""
    return "gains space" if sign > 0 else "loses space"   # from the mover's POV, not raw "more/less"



