"""MOBILITY term (evaluate.cpp): MobilityBonus over each piece's safe squares.

Mobility -> one leaf per piece (knight/bishop/rook/queen, both colours,
signed, by square). The MobilityBonus tables (the weights) live here; the parent
looks up each piece's bonus from its mobility count (in Context) and injects it.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import SCORE_ZERO, S, Score
from datagen.tree.primitives.base import Factor

# MobilityBonus[piece type][# attacked mobility squares]
MOBILITY_BONUS = {
    chess.KNIGHT: [S(-62, -81), S(-53, -56), S(-12, -30), S(-4, -14), S(3, 8),
                   S(13, 15), S(22, 23), S(28, 27), S(33, 33)],
    chess.BISHOP: [S(-48, -59), S(-20, -23), S(16, -3), S(26, 13), S(38, 24),
                   S(51, 42), S(55, 54), S(63, 57), S(63, 65), S(68, 73),
                   S(81, 78), S(81, 86), S(91, 88), S(98, 97)],
    chess.ROOK: [S(-58, -76), S(-27, -18), S(-15, 28), S(-10, 55), S(-5, 69),
                 S(-2, 82), S(9, 112), S(16, 118), S(30, 132), S(29, 142),
                 S(32, 155), S(38, 165), S(46, 166), S(48, 169), S(58, 171)],
    chess.QUEEN: [S(-39, -36), S(-21, -15), S(3, 8), S(3, 18), S(14, 34),
                  S(22, 54), S(28, 61), S(41, 73), S(43, 79), S(48, 92),
                  S(56, 94), S(60, 104), S(60, 113), S(66, 120), S(67, 123),
                  S(70, 126), S(71, 133), S(73, 136), S(79, 140), S(88, 143),
                  S(88, 148), S(99, 166), S(102, 170), S(102, 175), S(106, 184),
                  S(109, 191), S(113, 206), S(116, 212)],
}
SYMBOL = {chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}
ORDER = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def mobility_score(ctx: Context, color: chess.Color) -> Score:
    """Own-side mobility total (not yet signed) -- used by the King term too."""
    total = S(0, 0)
    for pt in ORDER:
        for rec in ctx.piece_recs(color, pt):
            total = total + MOBILITY_BONUS[pt][rec.mob_count]
    return total



class Mobility(Factor):
    tag = "mobility"
    KIND = "whole"

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        total = SCORE_ZERO
        for pt in ORDER:
            for rec in ctx.piece_recs_ordered(pt):
                b = MOBILITY_BONUS[pt][rec.mob_count]
                total = total + (b if rec.color == chess.WHITE else -b)
        return total, {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        us = chess.WHITE if mover == "w" else chess.BLACK
        return {self.tag: v_mobility(before, after, us, self.whole_direction(mover, before_eval, after_eval)) or None}




# ============================================================================== verbalization primitives
from datagen.tree.primitives import mobility as mobility_mod
from datagen.tree.primitives import toplevel as toplevel
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, derive_move, pcsym)


def piece_mob_cp(fen):
    b = chess.Board(fen)
    ctx = get_context(b)
    weigh = toplevel.position_weigher(b, ctx)
    return {(r.color, r.pt, r.sq): weigh(mobility_mod.MOBILITY_BONUS[r.pt][r.mob_count]) * 100 / 213
            for r in ctx.pieces if r.pt in mobility_mod.ORDER}


def elu_mobility(parent_fen, child_fen, mover, sign):
    P, C = piece_mob_cp(parent_fen), piece_mob_cp(child_fen)
    changes = []  # (mover_benefit, raw_delta, label) -- rank by benefit, word by raw
    m = derive_move(parent_fen, child_fen)
    if m is not None:
        pc = chess.Board(parent_fen).piece_at(m.from_square)
        if pc and pc.piece_type in mobility_mod.ORDER:                       # the moved piece
            raw = (C.get((pc.color, pc.piece_type, m.to_square), 0.0)
                   - P.get((pc.color, pc.piece_type, m.from_square), 0.0))
            s = 1 if pc.color == mover else -1
            changes.append((s * raw, raw,
                            f"{pcsym(pc.piece_type)}{chess.square_name(m.from_square)}→{chess.square_name(m.to_square)}"))
    for key in set(P) & set(C):                                            # pieces whose lines shifted
        raw = C[key] - P[key]
        if abs(raw) >= 3:
            color, pt, sq = key
            s = 1 if color == mover else -1
            changes.append((s * raw, raw, f"{pcsym(pt)}{chess.square_name(sq)}"))
    changes = [c for c in changes if (c[0] > 0) == (sign > 0)]             # align with the term's sign
    if not changes:
        return ""
    changes.sort(key=lambda c: -abs(c[0]))
    _, raw, label = changes[0]
    return f"{label} {'more active' if raw > 0 else 'less active'}"


# ---- piece positional factors (pieces.cpp) ----


def v_mobility(pf, cf, mover, sign):
    d = elu_mobility(pf, cf, mover, sign)
    if not d:
        return ""
    more = "more active" in d
    tok = d.replace(" more active", "").replace(" less active", "")
    name = PIECE.get({"N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK,
                       "Q": chess.QUEEN, "K": chess.KING, "P": chess.PAWN}.get(tok[0]), "piece")
    piece = f"the {name} now on {tok.split('→')[1]}" if "→" in tok else f"the {name} on {tok[1:]}"
    if more:
        return pick(d, [f"gives {piece} more scope", f"activates {piece}", f"frees {piece}"])
    return pick(d, [f"restricts {piece}", f"boxes in {piece}", f"leaves {piece} with little scope"])



