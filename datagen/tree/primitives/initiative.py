"""INITIATIVE correction (evaluate.cpp): a complexity-based adjustment applied to
the summed static terms. It depends on the full pre-initiative term sum, so it is a
single White-POV leaf computed from that sum plus board structure.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import SCORE_ZERO, S, Score
from datagen.tree.primitives.base import Factor


def initiative_from(board: chess.Board, ctx: Context, score: Score) -> Score:
    mg, eg = score.mg, score.eg
    kw, kb = board.king(chess.WHITE), board.king(chess.BLACK)
    outflanking = abs((kw & 7) - (kb & 7)) - abs((kw >> 3) - (kb >> 3))
    infiltration = (kw >> 3) > 3 or (kb >> 3) < 4
    pawns_both = bool(board.pawns & bb.QUEENSIDE) and bool(board.pawns & bb.KINGSIDE)
    passed_count = bb.popcount(ctx.pawn[chess.WHITE].passed_pawns | ctx.pawn[chess.BLACK].passed_pawns)
    npm = ctx.non_pawn_material(chess.WHITE) + ctx.non_pawn_material(chess.BLACK)
    almost_unwinnable = (not passed_count) and outflanking < 0 and not pawns_both
    complexity = (9 * passed_count + 11 * bb.popcount(int(board.pawns)) + 9 * outflanking
                  + 12 * infiltration + 21 * pawns_both + 51 * (1 if npm == 0 else 0)
                  - 43 * (1 if almost_unwinnable else 0) - 100)
    sg = lambda x: (x > 0) - (x < 0)
    u = sg(mg) * max(min(complexity + 50, 0), -abs(mg))
    v = sg(eg) * max(complexity, -abs(eg))
    return S(u, v)


class Initiative(Factor):
    tag = "initiative"
    KIND = "whole"

    # dominant structural driver of the complexity change -> its (rise, fall) phrasing.
    # These read straight after "It ..."/"which ...", so each must be a verb phrase.
    COMPONENT_NAMES = {
        "passers":          ("leaves more passed pawns",        "leaves fewer passed pawns"),
        "pawn count":       ("leaves more pawns on the board",  "leaves fewer pawns on the board"),
        "outflanking":      ("makes the kings more outflanked", "makes the kings less outflanked"),
        "king infiltration": ("infiltrates with a king",        "pulls a king back from enemy territory"),
        "both flanks":      ("spreads the pawns to both flanks", "reduces the pawns to one flank"),
        "pawn endgame":     ("liquidates into a pure pawn endgame", "leaves the pure pawn endgame"),
    }

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        from datagen.tree.primitives.toplevel import pre_initiative_sum
        return initiative_from(board, ctx, pre_initiative_sum(board, ctx)), {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """Name the single structural component whose change most drove the complexity
        term (mover-independent — the driver names itself by its own direction)."""
        b4, af = self.components(before), self.components(after)
        best = None
        for k in b4:
            d = af[k] - b4[k]
            if d and (best is None or abs(d) > abs(best[1])):
                best = (k, d)
        if best is None:
            return {self.tag: None}
        up, down = self.COMPONENT_NAMES[best[0]]
        return {self.tag: up if best[1] > 0 else down}

    @staticmethod
    def components(fen):
        """The six board-structure inputs to the complexity formula (see initiative_from)."""
        b = chess.Board(fen)
        ctx = get_context(b)
        kw, kb = b.king(chess.WHITE), b.king(chess.BLACK)
        pawns = int(b.pawns)
        npm = ctx.non_pawn_material(chess.WHITE) + ctx.non_pawn_material(chess.BLACK)
        passed = bb.popcount(ctx.pawn[chess.WHITE].passed_pawns | ctx.pawn[chess.BLACK].passed_pawns)
        return {
            "passers": 9 * passed,
            "pawn count": 11 * bb.popcount(pawns),
            "outflanking": 9 * (abs((kw & 7) - (kb & 7)) - abs((kw >> 3) - (kb >> 3))),
            "king infiltration": 12 * (1 if ((kw >> 3) > 3 or (kb >> 3) < 4) else 0),
            "both flanks": 21 * (1 if (pawns & bb.QUEENSIDE and pawns & bb.KINGSIDE) else 0),
            "pawn endgame": 51 * (1 if npm == 0 else 0),
        }



