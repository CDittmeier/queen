"""Top-level assembly: the static term sum, the (non-additive) scaling/interpolation
node, and the final value.

  TopLevelEvaluator
   |- StaticEvaluator        sum of all terms (Material .. Initiative), an (mg,eg)
   |   |- Material .. Initiative
   |- ScalingEvaluator       phase interpolation + endgame scale factor + tempo

The static term sum is a normal additive Score. The final value applies a phase
interpolation and an endgame scale factor (linear in (mg,eg), so each term's felt
contribution is its own Score through this same transform) plus tempo; that step is
exposed as its own node, and the integer-truncation residual is owned by it.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import (PHASE_MIDGAME, TEMPO, PAWN_VALUE_EG, S, Score, SCORE_ZERO,
                         cdiv, to_cp, non_pawn_material)
from datagen.tree.primitives.base import Factor
from datagen.tree.primitives.material import Material
from datagen.tree.primitives.imbalance import Imbalance
from datagen.tree.primitives.pawns import Pawns
from datagen.tree.primitives.pieces import Knights, Bishops, Rooks, Queens
from datagen.tree.primitives.mobility import Mobility
from datagen.tree.primitives.king import King
from datagen.tree.primitives.threats import Threats
from datagen.tree.primitives.passed import Passed
from datagen.tree.primitives.space import Space
from datagen.tree.primitives.initiative import Initiative

ENDGAME_LIMIT, MIDGAME_LIMIT = 3915, 15258


def terms_pre_initiative():
    return [Material(), Imbalance(), Pawns(),
            Knights(), Bishops(), Rooks(), Queens(),
            Mobility(), King(), Threats(),
            Passed(), Space()]


def pre_initiative_sum(board: chess.Board, ctx: Context) -> Score:
    s = SCORE_ZERO
    for ev in terms_pre_initiative():
        s = s + ev.score(board, ctx)[0]
    return s


def total_score(board: chess.Board, ctx: Context | None = None) -> Score:
    ctx = ctx or get_context(board)
    return pre_initiative_sum(board, ctx) + Initiative().score(board, ctx)[0]


# ---- final-value machinery (game phase, endgame scale factor) ---------------
def game_phase(board: chess.Board) -> int:
    npm = non_pawn_material(board, chess.WHITE) + non_pawn_material(board, chess.BLACK)
    npm = min(max(npm, ENDGAME_LIMIT), MIDGAME_LIMIT)
    return cdiv((npm - ENDGAME_LIMIT) * PHASE_MIDGAME, MIDGAME_LIMIT - ENDGAME_LIMIT)


def opposite_bishops(board: chess.Board) -> bool:
    bw, bbk = board.pieces(chess.BISHOP, chess.WHITE), board.pieces(chess.BISHOP, chess.BLACK)
    if len(bw) != 1 or len(bbk) != 1:
        return False
    sw, sb = next(iter(bw)), next(iter(bbk))
    return (((sw >> 3) + (sw & 7)) & 1) != (((sb >> 3) + (sb & 7)) & 1)


def material_factor(board: chess.Board, strong: chess.Color) -> int:
    npm_w = non_pawn_material(board, chess.WHITE)
    npm_b = non_pawn_material(board, chess.BLACK)
    pw = len(board.pieces(chess.PAWN, chess.WHITE))
    pb = len(board.pieces(chess.PAWN, chess.BLACK))
    fw = fb = 64
    if pw == 0 and npm_w - npm_b <= 825:
        fw = 0 if npm_w < 1276 else (4 if npm_b <= 825 else 14)
    if pb == 0 and npm_b - npm_w <= 825:
        fb = 0 if npm_b < 1276 else (4 if npm_w <= 825 else 14)
    return fw if strong == chess.WHITE else fb


def scale_factor(board: chess.Board, eg: int) -> int:
    strong = chess.WHITE if eg > 0 else chess.BLACK
    sf = material_factor(board, strong)
    if sf == 64:
        npm = non_pawn_material(board, chess.WHITE) + non_pawn_material(board, chess.BLACK)
        if opposite_bishops(board) and npm == 2 * 825:
            sf = 22
        else:
            sf = min(sf, 36 + (2 if opposite_bishops(board) else 7) * len(board.pieces(chess.PAWN, strong)))
        sf = max(0, sf - cdiv(board.halfmove_clock - 12, 4))
    return sf


def position_weigher(board: chess.Board, ctx: Context | None = None):
    """Returns weigh(Score) -> internal White-POV value (pre-tempo), the same
    interpolation+scale transform the final value uses. Linear, so it distributes
    over the term sum -- each node's blame magnitude is weigh(node.score)."""
    ctx = ctx or get_context(board)
    total = total_score(board, ctx)
    gp = game_phase(board)
    sf = scale_factor(board, total.eg)

    def weigh(s: Score) -> int:
        return cdiv(s.mg * gp + cdiv(s.eg * (PHASE_MIDGAME - gp) * sf, 64), PHASE_MIDGAME)
    return weigh


def value_white(board: chess.Board, ctx: Context | None = None) -> int:
    ctx = ctx or get_context(board)
    v = position_weigher(board, ctx)(total_score(board, ctx))
    return v + (TEMPO if board.turn == chess.WHITE else -TEMPO)


def value(board: chess.Board, ctx: Context | None = None) -> int:
    """Final eval in internal units, side-to-move POV (incl. tempo)."""
    vw = value_white(board, ctx)
    return vw if board.turn == chess.WHITE else -vw


TERMS = terms_pre_initiative() + [Initiative()]
EVAL_CACHE: dict[str, dict] = {}


def eval_full(fen):
    """Per-term White-POV cp plus each term's immediate sub-factors (by tag).
    The scoring view the move-effect verbalizers rank and describe."""
    if fen not in EVAL_CACHE:
        b = chess.Board(fen)
        ctx = get_context(b)
        weigh = position_weigher(b, ctx)
        out = {}
        for term in TERMS:
            total, breakdown = term.score(b, ctx)
            sub = {tag: weigh(sc) * 100 / 213 for tag, sc in breakdown.items()}
            out[term.tag] = (weigh(total) * 100 / 213, sub)
        EVAL_CACHE[fen] = out
    return EVAL_CACHE[fen]
