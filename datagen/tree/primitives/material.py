"""MATERIAL term: piece value + piece-square table, summed over every piece.

Material  -> one MaterialOnSquareEvaluator per occupied square (the
atomic leaf). The piece-square weight tables live here (the parent); the leaf is
weight-agnostic and just returns its injected, signed Score.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import get_context
from datagen.tree.primitives.core.score import SCORE_ZERO, S, Score, PIECE_VALUE_MG, PIECE_VALUE_EG
from datagen.tree.primitives.base import Factor

# Bonus[piece type][rank 0..7][queenside file 0..3]; files A..D, mirrored E..H.
BONUS = {
    chess.KNIGHT: [
        [S(-175, -96), S(-92, -65), S(-74, -49), S(-73, -21)],
        [S(-77, -67), S(-41, -54), S(-27, -18), S(-15, 8)],
        [S(-61, -40), S(-17, -27), S(6, -8), S(12, 29)],
        [S(-35, -35), S(8, -2), S(40, 13), S(49, 28)],
        [S(-34, -45), S(13, -16), S(44, 9), S(51, 39)],
        [S(-9, -51), S(22, -44), S(58, -16), S(53, 17)],
        [S(-67, -69), S(-27, -50), S(4, -51), S(37, 12)],
        [S(-201, -100), S(-83, -88), S(-56, -56), S(-26, -17)],
    ],
    chess.BISHOP: [
        [S(-53, -57), S(-5, -30), S(-8, -37), S(-23, -12)],
        [S(-15, -37), S(8, -13), S(19, -17), S(4, 1)],
        [S(-7, -16), S(21, -1), S(-5, -2), S(17, 10)],
        [S(-5, -20), S(11, -6), S(25, 0), S(39, 17)],
        [S(-12, -17), S(29, -1), S(22, -14), S(31, 15)],
        [S(-16, -30), S(6, 6), S(1, 4), S(11, 6)],
        [S(-17, -31), S(-14, -20), S(5, -1), S(0, 1)],
        [S(-48, -46), S(1, -42), S(-14, -37), S(-23, -24)],
    ],
    chess.ROOK: [
        [S(-31, -9), S(-20, -13), S(-14, -10), S(-5, -9)],
        [S(-21, -12), S(-13, -9), S(-8, -1), S(6, -2)],
        [S(-25, 6), S(-11, -8), S(-1, -2), S(3, -6)],
        [S(-13, -6), S(-5, 1), S(-4, -9), S(-6, 7)],
        [S(-27, -5), S(-15, 8), S(-4, 7), S(3, -6)],
        [S(-22, 6), S(-2, 1), S(6, -7), S(12, 10)],
        [S(-2, 4), S(12, 5), S(16, 20), S(18, -5)],
        [S(-17, 18), S(-19, 0), S(-1, 19), S(9, 13)],
    ],
    chess.QUEEN: [
        [S(3, -69), S(-5, -57), S(-5, -47), S(4, -26)],
        [S(-3, -55), S(5, -31), S(8, -22), S(12, -4)],
        [S(-3, -39), S(6, -18), S(13, -9), S(7, 3)],
        [S(4, -23), S(5, -3), S(9, 13), S(8, 24)],
        [S(0, -29), S(14, -6), S(12, 9), S(5, 21)],
        [S(-4, -38), S(10, -18), S(6, -12), S(8, 1)],
        [S(-5, -50), S(6, -27), S(10, -24), S(8, -8)],
        [S(-2, -75), S(-2, -52), S(1, -43), S(-2, -36)],
    ],
    chess.KING: [
        [S(271, 1), S(327, 45), S(271, 85), S(198, 76)],
        [S(278, 53), S(303, 100), S(234, 133), S(179, 135)],
        [S(195, 88), S(258, 130), S(169, 169), S(120, 175)],
        [S(164, 103), S(190, 156), S(138, 172), S(98, 172)],
        [S(154, 96), S(179, 166), S(105, 199), S(70, 199)],
        [S(123, 92), S(145, 172), S(81, 184), S(31, 191)],
        [S(88, 47), S(120, 121), S(65, 116), S(33, 131)],
        [S(59, 11), S(89, 59), S(45, 73), S(-1, 78)],
    ],
}

# PBonus[rank 0..7][file 0..7] (asymmetric; ranks 1 and 8 unused -> zero).
PBONUS = [
    [S(0, 0)] * 8,
    [S(3, -10), S(3, -6), S(10, 10), S(19, 0), S(16, 14), S(19, 7), S(7, -5), S(-5, -19)],
    [S(-9, -10), S(-15, -10), S(11, -10), S(15, 4), S(32, 4), S(22, 3), S(5, -6), S(-22, -4)],
    [S(-8, 6), S(-23, -2), S(6, -8), S(20, -4), S(40, -13), S(17, -12), S(4, -10), S(-12, -9)],
    [S(13, 9), S(0, 4), S(-13, 3), S(1, -12), S(11, -12), S(-2, -6), S(-13, 13), S(5, 8)],
    [S(-5, 28), S(-12, 20), S(-7, 21), S(22, 28), S(-8, 30), S(-5, 7), S(-15, 6), S(-18, 13)],
    [S(-7, 0), S(7, -11), S(-3, 12), S(-13, 21), S(5, 25), S(-16, 19), S(10, 4), S(-8, 7)],
    [S(0, 0)] * 8,
]

SYMBOL = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
           chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


def psq_white(pt: int, sq: int) -> Score:
    """psq[white pc][sq] = piece value + piece-square bonus (psqt.cpp)."""
    base = S(PIECE_VALUE_MG[pt], PIECE_VALUE_EG[pt])
    r, f = sq >> 3, sq & 7
    bonus = PBONUS[r][f] if pt == chess.PAWN else BONUS[pt][r][min(f, 7 - f)]
    return base + bonus



class Material(Factor):
    tag = "material"
    KIND = "material"

    def score(self, board, ctx=None):
        total = SCORE_ZERO
        for sq, piece in board.piece_map().items():
            w = psq_white(piece.piece_type, sq if piece.color == chess.WHITE else sq ^ 56)
            total = total + (w if piece.color == chess.WHITE else -w)
        return total, {}

    def describe(self, before, after, move, mover, before_eval, after_eval):
        return {"material": vmaterial(before, move) or None}




# ============================================================================== verbalization primitives
from datagen.tree.primitives import material as material_mod
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, PREV_CAPSQ, derive_move, pcsym)


def placement_reason(pt, color, frm, to):
    """A short piece-square-table reason, only when the square actually improved
    (per the mg PSQT) so the wording never contradicts a negative material cp."""
    fr_w = frm if color == chess.WHITE else frm ^ 56
    to_w = to if color == chess.WHITE else to ^ 56
    if material_mod.psq_white(pt, to_w).mg <= material_mod.psq_white(pt, fr_w).mg:
        return ""                                     # not a better square: let the -cp speak
    rr_f = chess.square_rank(frm) if color == chess.WHITE else 7 - chess.square_rank(frm)
    rr_t = chess.square_rank(to) if color == chess.WHITE else 7 - chess.square_rank(to)
    cf_f = min(chess.square_file(frm), 7 - chess.square_file(frm))
    cf_t = min(chess.square_file(to), 7 - chess.square_file(to))
    if pt in (chess.KNIGHT, chess.BISHOP) and (cf_t > cf_f or (rr_t > rr_f and rr_t >= 3)):
        return "more central"
    if pt in (chess.ROOK, chess.QUEEN) and rr_t >= 6 > rr_f:
        return "to the 7th" if rr_t == 6 else "to the 8th"
    if pt == chess.PAWN and rr_t > rr_f:
        return "advances"
    if pt == chess.KING and abs(chess.square_file(to) - chess.square_file(frm)) == 2:
        return "castles"
    return ""


def castle_rook(board, move):
    rank = 0 if board.turn == chess.WHITE else 7
    if chess.square_file(move.to_square) == 6:                 # kingside: h->f
        return chess.square(7, rank), chess.square(5, rank)
    return chess.square(0, rank), chess.square(3, rank)        # queenside: a->d


def material_note(parent_fen, move):
    b = chess.Board(parent_fen)
    if b.is_en_passant(move):
        return " (captures a pawn e.p.)"
    if b.is_capture(move):
        cap = b.piece_at(move.to_square)
        return f" (captures a {chess.piece_name(cap.piece_type)})" if cap else ""
    if move.promotion:
        return f" (promotes to {chess.piece_name(move.promotion)})"
    # quiet move: the material change is exactly the moved piece's piece-square delta
    moved = []
    pc = b.piece_at(move.from_square)
    if pc:
        moved.append((pc, move.from_square, move.to_square))
    if b.is_castling(move):
        rf, rt = castle_rook(b, move)
        rk = b.piece_at(rf)
        if rk:
            moved.append((rk, rf, rt))
    labels = []
    for p, frm, to in moved:
        why = placement_reason(p.piece_type, p.color, frm, to)
        lab = f"{pcsym(p.piece_type)}{chess.square_name(frm)}→{chess.square_name(to)}"
        labels.append(lab + (f" {why}" if why else ""))
    return f" ({'; '.join(labels)})" if labels else " (piece placement)"


# --- per-sub-factor elucidators (name the concrete geometry behind a factor) ---


def vmaterial(pf, move):
    b = chess.Board(pf)
    if b.is_en_passant(move):
        ep = piece_name(chess.PAWN, not b.turn)
        return pick(move.uci(), [f"wins a {ep} (en passant)", f"snaps off a {ep} en passant"])
    if b.is_capture(move):
        cap = b.piece_at(move.to_square)
        if cap:
            if PREV_CAPSQ[0] == move.to_square:       # taking back on the square just captured on
                return f"recaptures the {piece_name(cap.piece_type, cap.color)} on {square_name(move.to_square)}"
            return pick(move.uci(), [f"wins a {piece_name(cap.piece_type, cap.color)}", f"captures a {piece_name(cap.piece_type, cap.color)}",
                                      f"picks up a {piece_name(cap.piece_type, cap.color)}", f"grabs a {piece_name(cap.piece_type, cap.color)}"])
        return ""
    if move.promotion:
        return f"promotes to a {piece_name(move.promotion, b.turn)}"
    pc = b.piece_at(move.from_square)
    if not pc:
        return ""
    reason = placement_reason(pc.piece_type, pc.color, move.from_square, move.to_square)
    name, to = piece_name(pc.piece_type, pc.color), square_name(move.to_square)
    if reason == "more central":
        return pick(move.uci(), [f"centralizes the {name}", f"brings the {name} to a more active square on {to}",
                                  f"improves the {name}, planting it on {to}"])
    if reason == "to the 7th":
        return pick(move.uci(), [f"swings the {name} to the seventh rank", f"activates the {name} on {to}"])
    if reason == "to the 8th":
        return pick(move.uci(), [f"swings the {name} to the opponent's back rank", f"activates the {name} on {to}"])
    if reason == "castles":
        return pick(move.uci(), [f"castles the {name} into safety", f"tucks the {name} away by castling"])
    if pc.piece_type == chess.PAWN:
        return ""                                       # the move name already says where the pawn went
    return ""                                           # plain "repositions X to sq" only restates the move


# ---- threats family (two-sided, sign-aligned) ----



