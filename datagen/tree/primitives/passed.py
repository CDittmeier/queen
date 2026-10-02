"""PASSED term (evaluate.cpp): passed-pawn bonuses.

Passed -> one leaf per passed pawn (both colours, signed, by square). The
rank/file bonus tables and the blockade/king-proximity logic (with weights) live
here; the parent computes each passed pawn's Score and injects it.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO, cdiv
from datagen.tree.primitives.base import Factor

PassedRank = [S(0, 0), S(10, 28), S(17, 33), S(15, 41), S(62, 72), S(168, 177), S(276, 260), S(0, 0)]
PassedFile = S(11, 8)
SYMBOL = {chess.WHITE: "P", chess.BLACK: "p"}


def kprox(king_sq, s):
    return min(bb.dist(king_sq, s), 5)


def passed_pawn_score(board: chess.Board, ctx: Context, us: chess.Color, s: int) -> Score:
    them = not us
    up = 8 if us == chess.WHITE else -8
    A = ctx.att
    rq = int(board.rooks | board.queens)
    kus, kthem = board.king(us), board.king(them)
    r = bb.rel_rank(us, s)
    bonus = PassedRank[r]
    if r > 2:                          # r > RANK_3 (RANK_3 == 2)
        w = 5 * r - 13
        block = s + up
        bonus += S(0, (cdiv(kprox(kthem, block) * 19, 4) - kprox(kus, block) * 2) * w)
        if r != 6:
            bonus -= S(0, kprox(kus, block + up) * w)
        if board.piece_at(block) is None:
            squares_to_queen = bb.forward_file(us, s)
            unsafe = bb.passed_pawn_span(us, s)
            bbf = bb.forward_file(them, s) & rq
            if not (board.occupied_co[them] & bbf):
                unsafe &= A[them]["ALL"]
            if not unsafe:
                k = 35
            elif not (unsafe & squares_to_queen):
                k = 20
            elif not (unsafe & chess.BB_SQUARES[block]):
                k = 9
            else:
                k = 0
            if (board.occupied_co[us] & bbf) or (A[us]["ALL"] & chess.BB_SQUARES[block]):
                k += 5
            bonus += S(k * w, k * w)
    if (not ctx.pawn_passed(us, s + up)) or (board.pawns & chess.BB_SQUARES[s + up]):
        bonus = bonus.cdiv(2)
    return bonus - PassedFile * min(s & 7, 7 - (s & 7))



class Passed(Factor):
    tag = "passed"
    KIND = "whole"
    ACTIVITY = ("passed pawns",)

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        total = SCORE_ZERO
        for color in (chess.WHITE, chess.BLACK):
            for s in chess.scan_forward(ctx.pawn[color].passed_pawns):
                sc = passed_pawn_score(board, ctx, color, s)
                total = total + (sc if color == chess.WHITE else -sc)
        return total, {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        us = chess.WHITE if mover == "w" else chess.BLACK
        return {self.tag: v_passed(before, after, us, self.whole_direction(mover, before_eval, after_eval)) or None}




# ============================================================================== verbalization primitives
from datagen.tree.primitives import passed as passed_mod
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, derive_move, pcsym)


def passed_set(fen):
    """Passed pawns for verbalization — SF's set, MINUS any pawn the side-to-move can just
    capture with a pawn (attacked by an enemy pawn with that enemy to move): not really passed."""
    b = chess.Board(fen)
    ctx = get_context(b)
    out = set()
    for col in (chess.WHITE, chess.BLACK):
        for s in chess.scan_forward(ctx.pawn[col].passed_pawns):
            if b.attackers(not col, s) & b.pawns:
                continue                                # attacked by an enemy pawn — not a clean passer
            out.add((s, col))
    return out


def passed_score(fen):
    b = chess.Board(fen)
    ctx = get_context(b)
    out = {}
    for color in (chess.WHITE, chess.BLACK):
        for s in chess.scan_forward(ctx.pawn[color].passed_pawns):
            sc = passed_mod.passed_pawn_score(b, ctx, color, s)
            out[(color, s & 7)] = (sc.mg, sc.eg)
    return out


def elu_passed(parent_fen, child_fen, mover, sign):
    P, C = passed_set(parent_fen), passed_set(child_fen)
    items = []
    for s, col in C - P:                              # newly passed, or a passer that advanced
        adv = any(c2 == col and (s2 & 7) == (s & 7) for s2, c2 in P - C)
        pfx = "" if col == mover else "opp "
        if adv:
            items.append((col == mover, f"{pfx}{chess.FILE_NAMES[s & 7]}-passer to {chess.square_name(s)}"))
        else:
            items.append((col == mover, f"{pfx}{chess.square_name(s)} passed"))
    mv = derive_move(parent_fen, child_fen)
    for s, col in P - C:
        if s == CAPSQ[0]:
            continue                                  # the passer was captured — obvious, not "stopped"
        if mv is not None and mv.promotion and s == mv.from_square:
            continue                                  # the passer PROMOTED — not "stopped"
        if any(c2 == col and (s2 & 7) == (s & 7) for s2, c2 in C - P):
            continue                                  # already reported as an advance
        pfx = "" if col == mover else "opp "
        items.append((col != mover, f"{pfx}{chess.square_name(s)} no longer passed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    if items:
        return ", ".join(t for _, t in items[:2])
    # value change on a persisting passer (king shepherding, rook behind, blockade)
    Ps, Cs = passed_score(parent_fen), passed_score(child_fen)
    cands = []
    for key in set(Ps) & set(Cs):
        color, f = key
        (pmg, peg), (cmg, ceg) = Ps[key], Cs[key]
        if (pmg, peg) == (cmg, ceg):
            continue                                   # geometric no-change (phase excluded)
        mb = (ceg - peg) if color == mover else (peg - ceg)
        cands.append((mb, color, f, ceg > peg))
    cands = [c for c in cands if (c[0] > 0) == (sign > 0)]
    if not cands:
        return ""
    cands.sort(key=lambda c: -abs(c[0]))
    _, color, f, stronger = cands[0]
    who = "own " if color == mover else "opp "         # own/opp tokens -> root-relative via fix_ownopp
    return f"{who}{chess.FILE_NAMES[f]}-passer {'stronger' if stronger else 'weaker'}"


def v_passed(pf, cf, mover, sign):
    d = elu_passed(pf, cf, mover, sign)
    if not d:
        return ""
    if "no longer passed" in d:
        return "stops a passed pawn"
    if "passer to" in d:                                # a passed pawn was actually advanced
        return "pushes a passed pawn"
    if " passed" in d:                                  # a pawn became passed (a blocker/rival removed)
        return "creates a passed pawn"
    return f"makes {d}"                                 # value change: "makes our b-passer stronger"



