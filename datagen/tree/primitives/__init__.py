"""Hierarchical, modular port of Stockfish 11's hand-crafted static evaluation.

Public entry points:
  * Evaluator tree:  TopLevelEvaluator() (and the per-term evaluators).
  * Flat term accessors (below): used to validate term-by-term against the binary,
    and convenient for callers who just want a single term's Score.

Each accessor returns a `Score(mg, eg)`. Per-term ("percolor") accessors return the
given colour's OWN score (positive = good for that colour), matching SF's eval-trace
columns; whole-board accessors are White-POV net.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import get_context
from datagen.tree.primitives.core.score import Score, SCORE_ZERO, to_cp
from datagen.tree.primitives.material import Material
from datagen.tree.primitives.imbalance import Imbalance
from datagen.tree.primitives import pawns as pawns_mod
from datagen.tree.primitives import pieces as pieces_mod
from datagen.tree.primitives import mobility as mobility_mod
from datagen.tree.primitives import king as king_mod
from datagen.tree.primitives import threats as threats_mod
from datagen.tree.primitives import passed as passed_mod
from datagen.tree.primitives import space as space_mod
from datagen.tree.primitives.initiative import Initiative
from datagen.tree.primitives import toplevel as toplevel


def psq_score(board) -> Score:
    return Material().get_score(board)


def imbalance(board) -> Score:
    return Imbalance().get_score(board)


def pawn_score(board, color) -> Score:
    ctx = get_context(board)
    s = SCORE_ZERO
    for rec in ctx.pawn[color].pawns:
        s = s + pawns_mod.pawn_score(rec)[0]
    return s


def pieces_score(board, color, pt) -> Score:
    ctx = get_context(board)
    s = SCORE_ZERO
    for rec in ctx.piece_recs(color, pt):
        s = s + pieces_mod.piece_score(board, ctx, rec)[0]
    return s


def mobility_score(board, color) -> Score:
    return mobility_mod.mobility_score(get_context(board), color)


def king_score(board, color) -> Score:
    return king_mod.king_term(board, get_context(board), color)[0]


def threats_score(board, color) -> Score:
    return threats_mod.threats_term(board, get_context(board), color)[0]


def passed_score(board, color) -> Score:
    ctx = get_context(board)
    s = SCORE_ZERO
    for sq in chess.scan_forward(ctx.pawn[color].passed_pawns):
        s = s + passed_mod.passed_pawn_score(board, ctx, color, sq)
    return s


def space_score(board, color) -> Score:
    return space_mod.space_term(board, get_context(board), color)


def initiative(board) -> Score:
    return Initiative().get_score(board)


def total_score(board) -> Score:
    return toplevel.total_score(board)


def value_white(board) -> int:
    return toplevel.value_white(board)


def value(board) -> int:
    return toplevel.value(board)
