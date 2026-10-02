"""Bitboard helpers (square index a1=0, matching Stockfish and python-chess).

These are weight-free geometric primitives shared by the structural Context and
by individual evaluators. They mirror the corresponding bitboard.h / evaluate.cpp
helpers exactly.
"""
from __future__ import annotations

import chess

MASK64 = (1 << 64) - 1
NOT_FILE_A = MASK64 ^ chess.BB_FILE_A
NOT_FILE_H = MASK64 ^ chess.BB_FILE_H
CENTER_FILES = chess.BB_FILES[2] | chess.BB_FILES[3] | chess.BB_FILES[4] | chess.BB_FILES[5]
CENTER = (chess.BB_FILES[3] | chess.BB_FILES[4]) & (chess.BB_RANKS[3] | chess.BB_RANKS[4])
QUEENSIDE = chess.BB_FILES[0] | chess.BB_FILES[1] | chess.BB_FILES[2] | chess.BB_FILES[3]
KINGSIDE = chess.BB_FILES[4] | chess.BB_FILES[5] | chess.BB_FILES[6] | chess.BB_FILES[7]


def bb_not(bb: int) -> int:
    return MASK64 ^ (bb & MASK64)


def popcount(bb: int) -> int:
    return bin(bb & MASK64).count("1")


def rel_rank(color: chess.Color, sq: int) -> int:
    r = sq >> 3
    return r if color == chess.WHITE else 7 - r


def shift_up(color: chess.Color, bb: int) -> int:
    return ((bb << 8) & MASK64) if color == chess.WHITE else (bb >> 8)


def shift_down(color: chess.Color, bb: int) -> int:
    return (bb >> 8) if color == chess.WHITE else ((bb << 8) & MASK64)


def forward_ranks(color: chess.Color, sq: int) -> int:
    """All squares on ranks strictly in front of sq, relative to `color`."""
    r = sq >> 3
    rng = range(r + 1, 8) if color == chess.WHITE else range(0, r)
    bb = 0
    for rr in rng:
        bb |= chess.BB_RANKS[rr]
    return bb


def adjacent_files(sq: int) -> int:
    f = sq & 7
    bb = 0
    if f > 0:
        bb |= chess.BB_FILES[f - 1]
    if f < 7:
        bb |= chess.BB_FILES[f + 1]
    return bb


def forward_file(color: chess.Color, sq: int) -> int:
    return forward_ranks(color, sq) & chess.BB_FILES[sq & 7]


def pawn_attack_span(color: chess.Color, sq: int) -> int:
    return forward_ranks(color, sq) & adjacent_files(sq)


def passed_pawn_span(color: chess.Color, sq: int) -> int:
    return forward_ranks(color, sq) & (chess.BB_FILES[sq & 7] | adjacent_files(sq))


def pawn_attacks_bb(color: chess.Color, pawns: int) -> int:
    if color == chess.WHITE:
        return (((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9)) & MASK64
    return ((pawns & NOT_FILE_A) >> 9) | ((pawns & NOT_FILE_H) >> 7)


def pawn_double_attacks_bb(color: chess.Color, pawns: int) -> int:
    if color == chess.WHITE:
        return (((pawns & NOT_FILE_A) << 7) & ((pawns & NOT_FILE_H) << 9)) & MASK64
    return ((pawns & NOT_FILE_A) >> 9) & ((pawns & NOT_FILE_H) >> 7)


def bishop_attacks(sq: int, occ: int) -> int:
    return chess.BB_DIAG_ATTACKS[sq][occ & chess.BB_DIAG_MASKS[sq]]


def rook_attacks(sq: int, occ: int) -> int:
    return (chess.BB_RANK_ATTACKS[sq][occ & chess.BB_RANK_MASKS[sq]]
            | chess.BB_FILE_ATTACKS[sq][occ & chess.BB_FILE_MASKS[sq]])


def dist(a: int, b: int) -> int:
    return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3)))


def frontmost(color: chess.Color, bb: int) -> int:
    """frontmost_sq(c, bb): msb for White, lsb for Black."""
    if color == chess.WHITE:
        return bb.bit_length() - 1
    return (bb & -bb).bit_length() - 1


def blockers_for_king(board: chess.Board, c: chess.Color) -> int:
    """Pieces (any colour) standing alone between c's king and an enemy slider."""
    ksq = board.king(c)
    them = not c
    occ = int(board.occupied)
    blockers = 0
    rq = (board.rooks | board.queens) & board.occupied_co[them]
    bq = (board.bishops | board.queens) & board.occupied_co[them]
    kr, kf = ksq >> 3, ksq & 7
    snipers = 0
    for sq in chess.scan_forward(rq):
        if (sq >> 3) == kr or (sq & 7) == kf:
            snipers |= chess.BB_SQUARES[sq]
    for sq in chess.scan_forward(bq):
        if abs((sq >> 3) - kr) == abs((sq & 7) - kf):
            snipers |= chess.BB_SQUARES[sq]
    for sq in chess.scan_forward(snipers):
        b = chess.between(ksq, sq) & occ
        if b and (b & (b - 1)) == 0:           # exactly one piece between
            blockers |= b
    return blockers
