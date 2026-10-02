"""Position-activity detectors and their frozen percentile cutoffs.

Eleven static positional features (pawn structure .. passed pawns), each scored
as a "felt" value: the feature's HCE score pushed through the position's own
phase/scale weigher, White-POV pawns; 0.0 means the feature does not fire. A
feature is ACTIVE when |felt| clears its chosen corpus percentile — measured
over firing positions in an eval-balanced pool and frozen as absolute pawn
values in thresholds/position_features.json. `is_active_position` answers that
question; `dominant_feature` additionally requires the feature to be the
position's largest, |felt| normalized by each feature's own threshold.

The move-side twin lives in thresholds/move_deltas.json: per term and
sub-factor, the corpus percentiles of the per-ply |delta cp|. `move_bar` is
the lookup; Factor.is_active_move applies it to an eval_full pair.

Every Factor subclass lists the features it owns in ACTIVITY, so the cutoffs
are reachable from the feature classes themselves (Factor.is_active_position).
"""
from __future__ import annotations

import json
from pathlib import Path

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import SCORE_ZERO, to_cp
from datagen.tree.primitives.mobility import MOBILITY_BONUS
from datagen.tree.primitives.pawns import Pawns, pawn_score
from datagen.tree.primitives.passed import passed_pawn_score
from datagen.tree.primitives.pieces import (BishopPawns, LongDiagonalBishop,
                                            RookOnFile, piece_score)
from datagen.tree.primitives.space import Space, space_term
from datagen.tree.primitives.toplevel import position_weigher

_THRESHOLDS = Path(__file__).parent / "thresholds"

# felt floor: below this a feature does not fire at all (pawns)
FELT_MIN = 0.05

# chosen percentile per feature (None = toggle: the threshold is FELT_MIN)
CHOSEN = {"pawn structure": "0.95", "knight outpost": None,
          "active bishop pair": "0.8", "active lone bishop": "0.95",
          "bad bishop": "0.6", "bad knight": "0.75", "bad rook": "0.9",
          "active rook": "0.95", "colour complexion": "0.9", "space": "0.9",
          "passed pawns": "0.5"}

_BAD_BISHOP_MAX_MOB = 3
_BAD_KNIGHT_MAX_MOB = 2
_BAD_ROOK_MAX_MOB = 3
_ROOK_HOME = {chess.WHITE: (chess.A1, chess.H1), chess.BLACK: (chess.A8, chess.H8)}


def felt(board: chess.Board, ctx: Context, score) -> float:
    """White-POV pawns after the position's own phase/scale weighing."""
    return to_cp(position_weigher(board, ctx)(score))


# --------------------------------------------------------------- detectors
# Each returns White-POV felt pawns; 0.0 = does not fire.

def pawn_structure(board, ctx) -> float:
    return felt(board, ctx, Pawns().score(board, ctx)[0])


PAWN_FACTOR_ORDER = ("connected", "isolated", "backward", "doubled", "weak lever")


def pawn_structure_sides(board, ctx) -> dict:
    """{(is_white, factor): felt} per-side pawn-structure aggregates, keys in
    factor-major order (the prose lists factors in this canonical order)."""
    sums = {}
    for color in (chess.WHITE, chess.BLACK):
        sgn = 1 if color == chess.WHITE else -1
        for rec in ctx.pawn[color].pawns:
            for name, s in pawn_score(rec)[1]:
                key = (color == chess.WHITE, name)
                sums[key] = sums.get(key, 0.0) + felt(board, ctx, s) * sgn
    return {(w, f): sums[(w, f)] for f in PAWN_FACTOR_ORDER
            for w in (True, False) if (w, f) in sums}


def _outpost_keep(sq: int, white: bool) -> bool:
    """Rim outposts (a/h file) only count on the sixth relative rank."""
    if chess.square_file(sq) not in (0, 7):
        return True
    return chess.square_rank(sq) == (5 if white else 2)


def outpost_knights(board, ctx):
    """[(color, sq, Score)] — knights on qualifying outposts."""
    out = []
    for color in (chess.WHITE, chess.BLACK):
        for r in ctx.piece_recs(color, chess.KNIGHT):
            for name, s in piece_score(board, ctx, r)[1]:
                if name == "outpost" and _outpost_keep(r.sq, color == chess.WHITE):
                    out.append((color, r.sq, s))
    return out


def knight_outpost(board, ctx) -> float:
    return sum(felt(board, ctx, s) * (1 if color == chess.WHITE else -1)
               for color, _sq, s in outpost_knights(board, ctx))


def bishop_pair(ctx, col):
    """The side's bishop recs iff it has bishops on both colour complexes."""
    recs = list(ctx.piece_recs(col, chess.BISHOP))
    if len(recs) < 2:
        return None
    light = [bool(chess.BB_SQUARES[r.sq] & chess.BB_LIGHT_SQUARES) for r in recs]
    return recs if (any(light) and not all(light)) else None


def active_bishop_pair(board, ctx) -> float:
    w, b = bishop_pair(ctx, chess.WHITE), bishop_pair(ctx, chess.BLACK)
    if (w is None) == (b is None):          # neither side, or both, has the pair
        return 0.0
    recs, sgn = (w, 1) if w is not None else (b, -1)
    total = SCORE_ZERO
    for r in recs:
        total = total + MOBILITY_BONUS[chess.BISHOP][r.mob_count]
    return sgn * max(0.0, felt(board, ctx, total))


def lone_bishop_activity(board, ctx, col) -> float:
    recs = list(ctx.piece_recs(col, chess.BISHOP))
    if len(recs) != 1:
        return 0.0
    r = recs[0]
    total = MOBILITY_BONUS[chess.BISHOP][r.mob_count]
    if bb.popcount(bb.bishop_attacks(r.sq, int(board.pawns)) & bb.CENTER) > 1:
        total = total + LongDiagonalBishop
    return max(0.0, felt(board, ctx, total))


def active_lone_bishop(board, ctx) -> float:
    return (lone_bishop_activity(board, ctx, chess.WHITE)
            - lone_bishop_activity(board, ctx, chess.BLACK))


def bad_bishops(board, ctx, col):
    """[(sq, own-pawns-on-colour count, penalty Score)] for `col`'s bad bishops."""
    own_pawns = int(board.pawns & board.occupied_co[col])
    blocked = own_pawns & bb.shift_down(col, ctx.occ)
    out = []
    for r in ctx.piece_recs(col, chess.BISHOP):
        if r.mob_count > _BAD_BISHOP_MAX_MOB:
            continue
        same = (chess.BB_LIGHT_SQUARES
                if (chess.BB_SQUARES[r.sq] & chess.BB_LIGHT_SQUARES)
                else chess.BB_DARK_SQUARES)
        k = bb.popcount(own_pawns & same)
        pen = BishopPawns * k * (1 + bb.popcount(blocked & bb.CENTER_FILES))
        out.append((r.sq, k, pen))
    return out


def bishop_badness(board, ctx, col) -> float:
    return sum(felt(board, ctx, pen) for _sq, _k, pen in bad_bishops(board, ctx, col))


def bad_bishop(board, ctx) -> float:
    return bishop_badness(board, ctx, chess.BLACK) - bishop_badness(board, ctx, chess.WHITE)


def bad_knights(board, ctx, col):
    """[(sq, mob_count)] — very low mobility, not pinned (the pin is the story)."""
    return [(r.sq, r.mob_count) for r in ctx.piece_recs(col, chess.KNIGHT)
            if r.mob_count <= _BAD_KNIGHT_MAX_MOB and not board.is_pinned(col, r.sq)]


def knight_badness(board, ctx, col) -> float:
    return sum(max(0.0, -felt(board, ctx, MOBILITY_BONUS[chess.KNIGHT][mob]))
               for _sq, mob in bad_knights(board, ctx, col))


def bad_knight(board, ctx) -> float:
    return knight_badness(board, ctx, chess.BLACK) - knight_badness(board, ctx, chess.WHITE)


def _counts_rook(board, col, sq) -> bool:
    """Home-corner rooks (a development lag, not a bad rook) and pinned rooks
    are disregarded."""
    return sq not in _ROOK_HOME[col] and not board.is_pinned(col, sq)


def bad_rooks(board, ctx, col):
    """[(sq, mob_count)] for `col`'s counting low-mobility rooks."""
    return [(r.sq, r.mob_count) for r in ctx.piece_recs(col, chess.ROOK)
            if r.mob_count <= _BAD_ROOK_MAX_MOB and _counts_rook(board, col, r.sq)]


def rook_badness(board, ctx, col) -> float:
    total = 0.0
    for _sq, mob in bad_rooks(board, ctx, col):
        total += max(0.0, -felt(board, ctx, MOBILITY_BONUS[chess.ROOK][mob]))
    for r in ctx.piece_recs(col, chess.ROOK):
        if not _counts_rook(board, col, r.sq):
            continue
        for name, s in piece_score(board, ctx, r)[1]:
            if name == "trapped rook":
                total += max(0.0, -felt(board, ctx, s))
    return total


def bad_rook(board, ctx) -> float:
    return rook_badness(board, ctx, chess.BLACK) - rook_badness(board, ctx, chess.WHITE)


def rook_file_bonus(board, col, sq):
    """The rook-on-open/semi-open-file Score, or None off such a file."""
    own_pawns = int(board.pawns & board.occupied_co[col])
    enemy_pawns = int(board.pawns & board.occupied_co[not col])
    if own_pawns & chess.BB_FILES[sq & 7]:
        return None
    return RookOnFile[0 if (enemy_pawns & chess.BB_FILES[sq & 7]) else 1]


def rook_activity(board, ctx, col) -> float:
    total = 0.0
    for r in ctx.piece_recs(col, chess.ROOK):
        act = max(0.0, felt(board, ctx, MOBILITY_BONUS[chess.ROOK][r.mob_count]))
        fb = rook_file_bonus(board, col, r.sq)
        if fb is not None:
            act += felt(board, ctx, fb)
        total += act
    return total


def active_rook(board, ctx) -> float:
    return rook_activity(board, ctx, chess.WHITE) - rook_activity(board, ctx, chess.BLACK)


def colour_complexion(board, ctx) -> float:
    """The bishop-pawns term (own pawns fixed on the bishop's colour), netted.
    Summed as a Score and weighed once (the frozen thresholds assume that)."""
    total = SCORE_ZERO
    for color in (chess.WHITE, chess.BLACK):
        for r in ctx.piece_recs(color, chess.BISHOP):
            for name, s in piece_score(board, ctx, r)[1]:
                if name == "bishop pawns":
                    total = total + (s if color == chess.WHITE else -s)
    return felt(board, ctx, total)


def weak_colour(board, ctx, col):
    """`col`'s strongest colour weakness: True=light squares, or None. Pawns
    fixed on the bishop's colour leave the OPPOSITE colour uncoverable."""
    best, mag = None, 0
    own_pawns = int(board.pawns & board.occupied_co[col])
    for r in ctx.piece_recs(col, chess.BISHOP):
        light = bool(chess.BB_SQUARES[r.sq] & chess.BB_LIGHT_SQUARES)
        same = chess.BB_LIGHT_SQUARES if light else chess.BB_DARK_SQUARES
        n = bb.popcount(own_pawns & same)
        if n > mag:
            best, mag = not light, n
    return best


def space(board, ctx) -> float:
    return felt(board, ctx, Space().score(board, ctx)[0])


SPACE_REGIONS = [
    ("on the queenside", chess.BB_FILES[0] | chess.BB_FILES[1] | chess.BB_FILES[2]),
    ("in the center", chess.BB_FILES[3] | chess.BB_FILES[4]),
    ("on the kingside", chess.BB_FILES[5] | chess.BB_FILES[6] | chess.BB_FILES[7])]


def space_region_count(board, ctx, us, files):
    """Safe-square count as in space_term, restricted to a file region."""
    them = not us
    own_pawns = int(board.pawns & board.occupied_co[us])
    ranks = (1, 2, 3) if us == chess.WHITE else (6, 5, 4)
    mask = files & (chess.BB_RANKS[ranks[0]] | chess.BB_RANKS[ranks[1]]
                    | chess.BB_RANKS[ranks[2]])
    safe = mask & bb.bb_not(own_pawns) & bb.bb_not(ctx.att[them][chess.PAWN])
    behind = own_pawns
    behind |= bb.shift_down(us, behind)
    behind |= bb.shift_down(us, bb.shift_down(us, behind))
    return (bb.popcount(safe)
            + bb.popcount(behind & safe & bb.bb_not(ctx.att[them]["ALL"])))


def passed_squares(board, ctx, col):
    """Squares of `col`'s passed pawns, by the ORIGINAL SF11 definition.

    The tree pipeline's Context deliberately deviates from SF11 here (a pawn
    whose stoppers currently lever it is not passed there); the frozen activity
    thresholds were measured with SF11's `(stoppers ^ lever) == 0`, so this
    feature keeps SF11's set rather than the Context flag."""
    them = not col
    up = 8 if col == chess.WHITE else -8
    our = int(board.pawns & board.occupied_co[col])
    their = int(board.pawns & board.occupied_co[them])
    dbl_them = bb.pawn_double_attacks_bb(them, their)
    out = []
    for s in chess.scan_forward(our):
        r = bb.rel_rank(col, s)
        blocked = their & chess.BB_SQUARES[s + up]
        stoppers = their & bb.passed_pawn_span(col, s)
        lever = their & chess.BB_PAWN_ATTACKS[col][s]
        lever_push = their & chess.BB_PAWN_ATTACKS[col][s + up]
        neighbours = our & bb.adjacent_files(s)
        phalanx = neighbours & chess.BB_RANKS[s >> 3]
        support = neighbours & chess.BB_RANKS[(s - up) >> 3]
        if ((stoppers ^ lever) == 0
                or ((stoppers ^ lever_push) == 0
                    and bb.popcount(phalanx) >= bb.popcount(lever_push))
                or (stoppers == blocked and r >= 4
                    and (bb.shift_up(col, support) & bb.bb_not(their | dbl_them)))):
            out.append(s)
    return out


def passed_pawns(board, ctx) -> float:
    total = SCORE_ZERO
    for col in (chess.WHITE, chess.BLACK):
        for s in passed_squares(board, ctx, col):
            sc = passed_pawn_score(board, ctx, col, s)
            total = total + (sc if col == chess.WHITE else -sc)
    return felt(board, ctx, total)


FEATURES = {
    "pawn structure": pawn_structure,
    "knight outpost": knight_outpost,
    "active bishop pair": active_bishop_pair,
    "active lone bishop": active_lone_bishop,
    "bad bishop": bad_bishop,
    "bad knight": bad_knight,
    "bad rook": bad_rook,
    "active rook": active_rook,
    "colour complexion": colour_complexion,
    "space": space,
    "passed pawns": passed_pawns,
}


def measure(board: chess.Board, ctx: Context | None = None) -> dict:
    """{feature: White-POV felt pawns} for all eleven features."""
    ctx = ctx or get_context(board)
    return {name: fn(board, ctx) for name, fn in FEATURES.items()}


# --------------------------------------------------- frozen percentile cutoffs

_POS_THR = None
_MOVE_THR = None


def position_thresholds() -> dict:
    """{feature: felt threshold (pawns)} at each feature's chosen percentile."""
    global _POS_THR
    if _POS_THR is None:
        pct = json.load(open(_THRESHOLDS / "position_features.json"))["percentiles"]
        _POS_THR = {n: FELT_MIN if q is None else pct[n][q]
                    for n, q in CHOSEN.items()}
    return _POS_THR


def move_bar(channel: str, pct: int, fallback: float | None = None):
    """The |delta cp| a term ("bishops") or sub-factor ("bishops/outpost")
    channel must move to clear its corpus percentile. A channel absent from the
    table returns `fallback`."""
    global _MOVE_THR
    if _MOVE_THR is None:
        _MOVE_THR = json.load(open(_THRESHOLDS / "move_deltas.json"))
    d = _MOVE_THR.get(channel)
    return fallback if d is None else d[f"p{pct}"]


def is_active_position(board: chess.Board, ctx: Context | None = None,
                       felt_map: dict | None = None) -> dict:
    """{feature: felt} for the features whose |felt| clears their frozen
    threshold. In-check positions get no felt measurement: {}."""
    if felt_map is None:
        if board.is_check():
            return {}
        felt_map = measure(board, ctx)
    thr = position_thresholds()
    return {n: felt_map[n] for n in thr if abs(felt_map[n]) >= thr[n]}


def dominant_feature(board: chess.Board, ctx: Context | None = None,
                     felt_map: dict | None = None):
    """(feature, felt) for the position's dominant active feature — largest
    |felt| normalized by each feature's own threshold — or None if nothing
    clears its threshold (or the side to move is in check)."""
    if felt_map is None:
        if board.is_check():
            return None
        felt_map = measure(board, ctx)
    thr = position_thresholds()
    norms = {n: abs(felt_map[n]) / thr[n] for n in thr}
    best = max(norms, key=norms.get)
    return (best, felt_map[best]) if norms[best] >= 1.0 else None
