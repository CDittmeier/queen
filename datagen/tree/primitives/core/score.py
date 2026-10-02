"""Packed (mg, eg) score with Stockfish's integer semantics.

Stockfish stores a Score as two int16 halves packed in one int32; every operation
on it (add, sub, *int, /int) is component-wise -- SF even asserts this -- so a
plain (mg, eg) pair with C-style truncating division is bit-identical over the
value range the evaluation uses.
"""
from __future__ import annotations

from dataclasses import dataclass

PHASE_MIDGAME = 128
PAWN_VALUE_EG = 213          # to_cp divides by this (evaluate.cpp to_cp)
TEMPO = 28


def cdiv(a: int, b: int) -> int:
    """C/C++ integer division: truncate toward zero (not Python floor)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


@dataclass(frozen=True)
class Score:
    mg: int = 0
    eg: int = 0

    def __add__(self, o: "Score") -> "Score": return Score(self.mg + o.mg, self.eg + o.eg)
    def __sub__(self, o: "Score") -> "Score": return Score(self.mg - o.mg, self.eg - o.eg)
    def __neg__(self) -> "Score": return Score(-self.mg, -self.eg)
    def __mul__(self, k: int) -> "Score": return Score(self.mg * k, self.eg * k)
    __rmul__ = __mul__
    def cdiv(self, k: int) -> "Score": return Score(cdiv(self.mg, k), cdiv(self.eg, k))


def S(mg: int, eg: int) -> Score:
    return Score(mg, eg)


SCORE_ZERO = S(0, 0)

# Canonical piece values (types.h). These are the fundamental material units; the
# full material term (value + piece-square table) lives in material.py, but the
# raw MG/EG values are also used structurally (game phase, non-pawn material,
# space threshold, scale factor), so they live in core.
import chess as chess

PIECE_VALUE_MG = {chess.PAWN: 128, chess.KNIGHT: 781, chess.BISHOP: 825,
                  chess.ROOK: 1276, chess.QUEEN: 2538, chess.KING: 0}
PIECE_VALUE_EG = {chess.PAWN: 213, chess.KNIGHT: 854, chess.BISHOP: 915,
                  chess.ROOK: 1380, chess.QUEEN: 2682, chess.KING: 0}


def non_pawn_material(board: "chess.Board", color: "chess.Color") -> int:
    return (len(board.pieces(chess.KNIGHT, color)) * PIECE_VALUE_MG[chess.KNIGHT]
            + len(board.pieces(chess.BISHOP, color)) * PIECE_VALUE_MG[chess.BISHOP]
            + len(board.pieces(chess.ROOK, color)) * PIECE_VALUE_MG[chess.ROOK]
            + len(board.pieces(chess.QUEEN, color)) * PIECE_VALUE_MG[chess.QUEEN])


def to_cp(v: int) -> float:
    """Convert an internal value to pawns, as the eval trace prints (v / 213)."""
    return v / PAWN_VALUE_EG
