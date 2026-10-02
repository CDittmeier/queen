"""PAWNS term (pawns.cpp evaluate<>): pawn-structure score.

Decomposed all the way down: Pawns -> one evaluator class per pawn-structure
factor (connected / isolated / backward / doubled / weak-lever) -> one atomic leaf per
pawn that has that factor (called out by square). The penalty/bonus weights live in
the factor classes; the leaf is a weight-agnostic carrier.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO, cdiv
from datagen.tree.primitives.base import Factor

Backward      = S(9, 24)
Doubled       = S(11, 56)
Isolated      = S(5, 15)
WeakLever     = S(0, 56)
WeakUnopposed = S(13, 27)
Connected = [0, 7, 8, 12, 29, 48, 86, 0]   # by relative rank 0..7

SYMBOL = {chess.WHITE: "P", chess.BLACK: "p"}


def pawn_score(rec) -> tuple[Score, list]:
    """(score, [(factor, score)]) for one pawn, from its structural record.
    The factors partition the pawn's score, so they sum back to it."""
    score = SCORE_ZERO
    factors = []
    r = rec.r
    if rec.support or rec.phalanx:
        v = Connected[r] * (2 + (1 if rec.phalanx else 0) - (1 if rec.opposed else 0)) \
            + 21 * rec.support_count
        c = S(v, cdiv(v * (r - 2), 4))
        score += c
        factors.append(("connected", c))
    elif not rec.neighbours:
        c = -(Isolated + WeakUnopposed * (0 if rec.opposed else 1))
        score += c
        factors.append(("isolated", c))
    elif rec.backward:
        c = -(Backward + WeakUnopposed * (0 if rec.opposed else 1))
        score += c
        factors.append(("backward", c))
    if not rec.support:
        if rec.doubled:
            c = -(Doubled)
            score += c
            factors.append(("doubled", c))
        if rec.lever_more_than_one:
            c = -(WeakLever)
            score += c
            factors.append(("weak lever", c))
    return score, factors



class Pawns(Factor):
    tag = "pawns"
    ACTIVITY = ("pawn structure",)

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        total, breakdown = SCORE_ZERO, {}
        for color in (chess.WHITE, chess.BLACK):
            for rec in ctx.pawn[color].pawns:
                for name, s in pawn_score(rec)[1]:
                    signed = s if color == chess.WHITE else -s
                    total = total + signed
                    breakdown[name] = breakdown.get(name, SCORE_ZERO) + signed
        return total, breakdown

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the pawn structure changed across the move."""
        return self.run_phrasers({
            "connected": (v_connected, elu_connected),
            "backward": (v_pawn_struct("backward", "leaves a backward pawn", "frees the backward pawn"), elu_backward),
            "isolated": (v_pawn_struct("isolated", "leaves an isolated pawn", "resolves the isolated pawn"), elu_isolated),
            "doubled": (v_doubled, elu_doubled),
            "weak lever": (None, elu_weak_lever),
        }, before, after, mover, before_eval, after_eval)




# ============================================================================== verbalization primitives
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives import pawns as pawns_mod
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ)


def pawn_flags(fen, color):
    conn, back, iso, dbl, lev = set(), set(), set(), set(), set()
    for rec in get_context(chess.Board(fen)).pawn[color].pawns:
        if rec.support or rec.phalanx:
            conn.add(rec.sq)
        elif not rec.neighbours:
            iso.add(rec.sq)
        elif rec.backward:
            back.add(rec.sq)
        if not rec.support:                            # doubled/weak-lever apply to unsupported pawns
            if rec.doubled:
                dbl.add(rec.sq)
            if rec.lever_more_than_one:
                lev.add(rec.sq)
    return conn, back, iso, dbl, lev


def pawn_delta(parent_fen, child_fen, which, mover):
    idx = {"connected": 0, "backward": 1, "isolated": 2, "doubled": 3, "weak lever": 4}[which]
    for color, tag in ((mover, ""), (not mover, "opp ")):
        present = set(chess.Board(child_fen).pieces(chess.PAWN, color))  # pawns still on board
        pb = pawn_flags(parent_fen, color)[idx]
        pa = pawn_flags(child_fen, color)[idx]
        gained = [f"{tag}{chess.square_name(s)} now" for s in sorted(pa - pb)]
        # only count a pawn that STAYED and lost the property (not one that just moved away)
        lost = [f"{tag}{chess.square_name(s)} gone" for s in sorted((pb - pa) & present)]
        out = gained + lost
        if out:
            return ", ".join(out[:3])
    return ""


def connected_mg(fen):
    """{(color, file): (connected mg, relative rank)} -- mg is phase-independent, so
    diffing it isolates real structural change from capture-driven phase wiggle."""
    out = {}
    for color in (chess.WHITE, chess.BLACK):
        for rec in get_context(chess.Board(fen)).pawn[color].pawns:
            mg = 0
            for name, s in pawns_mod.pawn_score(rec)[1]:
                if name == "connected":
                    mg = s.mg
            key = (color, rec.sq & 7)
            if key not in out or mg > out[key][0]:     # keep the most-connected pawn on the file
                out[key] = (mg, rec.r)
    return out


def doubled_files(fen):
    out = {}                                          # (color, file) -> # of penalised doubled pawns
    for color in (chess.WHITE, chess.BLACK):
        for rec in get_context(chess.Board(fen)).pawn[color].pawns:
            if not rec.support and rec.doubled:
                out[(color, rec.sq & 7)] = out.get((color, rec.sq & 7), 0) + 1
    return out


def elu_connected(parent_fen, child_fen, mover, sign):
    d = pawn_delta(parent_fen, child_fen, "connected", mover)
    if d:
        return d
    P, C = connected_mg(parent_fen), connected_mg(child_fen)   # value change (advance / support)
    cands = []
    for key in set(P) & set(C):
        color, f = key
        pmg, pr = P[key]
        cmg, cr = C[key]
        if pmg == cmg and pr == cr:
            continue                                   # no geometric change (phase only)
        mb = (cmg - pmg) if color == mover else (pmg - cmg)
        if mb == 0:
            mb = (cr - pr) * (1 if color == mover else -1)
        cands.append((mb, color, f, pr, cr, pmg, cmg))
    cands = [c for c in cands if (c[0] > 0) == (sign > 0)]
    if not cands:
        return ""
    cands.sort(key=lambda c: -abs(c[0]))
    _, color, f, pr, cr, pmg, cmg = cands[0]
    who = "" if color == mover else "opp "
    fn = chess.FILE_NAMES[f]
    if cr != pr:
        return f"{who}{fn}-pawn {'advances' if cr > pr else 'retreats'}"
    return f"{who}{fn}-pawn {'more' if cmg > pmg else 'less'} connected"


def elu_backward(parent_fen, child_fen, mover, sign):
    return pawn_delta(parent_fen, child_fen, "backward", mover)


# ---- mobility (top-level term): which piece got more/less active ----


def elu_isolated(parent_fen, child_fen, mover, sign):
    return pawn_delta(parent_fen, child_fen, "isolated", mover)


def elu_doubled(parent_fen, child_fen, mover, sign):
    P, C = doubled_files(parent_fen), doubled_files(child_fen)
    items = []
    for key in set(P) | set(C):
        col, f = key
        d = C.get(key, 0) - P.get(key, 0)
        if d == 0:
            continue
        who = "" if col == mover else "opp "
        opp_good = col != mover                        # doubled is a penalty: opp doubling helps mover
        if d > 0:
            items.append((opp_good, f"{who}{chess.FILE_NAMES[f]}-file doubled"))
        else:
            items.append((not opp_good, f"{who}{chess.FILE_NAMES[f]}-file undoubled"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_weak_lever(parent_fen, child_fen, mover, sign):
    return pawn_delta(parent_fen, child_fen, "weak lever", mover)


# ---- more piece / king / threat primitives ----


def v_pawn_struct(which, noun_now, noun_gone):
    def f(pf, cf, mover, sign):
        d = pawn_delta(pf, cf, which, mover)
        if d:
            gained = [x[:-4].strip() for x in d.split(", ") if x.endswith(" now")]
            gone = [x[:-5].strip() for x in d.split(", ") if x.endswith(" gone")]
            outs = []
            if gained:
                outs.append(f"{noun_now} at {join(gained)}")
            if gone:
                outs.append(f"{noun_gone} at {join(gone)}")
            return join(outs)
        return ""
    return f


def connected_partner(board, s, color):
    """A friendly pawn that connects to the one on `s` — a diagonal supporter behind it,
    else a phalanx neighbour on the same rank."""
    us = board.pieces(chess.PAWN, color)
    back = -8 if color == chess.WHITE else 8
    for d in (back - 1, back + 1):                       # supporter one rank behind
        p = s + d
        if 0 <= p < 64 and p in us and abs((p & 7) - (s & 7)) == 1:
            return p
    for d in (-1, 1):                                    # phalanx neighbour
        p = s + d
        if 0 <= p < 64 and p in us and (p >> 3) == (s >> 3):
            return p
    return None


def v_connected(pf, cf, mover, sign):
    cb = chess.Board(cf)
    connects, breaks, partners = [], [], set()
    for us in (mover, not mover):
        before, after = set(pawn_flags(pf, us)[0]), set(pawn_flags(cf, us)[0])
        for s in sorted(after - before):                 # newly connected -> name BOTH pawns
            partner = connected_partner(cb, s, us)
            if partner is not None:
                partners |= {s, partner}
                connects.append(f"connects the pawns on {square_name(min(s, partner))} "
                                f"and {square_name(max(s, partner))}")
        for s in sorted((before - after) & set(cb.pieces(chess.PAWN, us))):
            if s not in partners:                        # not just the trailing half of a NEW connection
                breaks.append(f"breaks the pawn chain at {square_name(s)}")
    outs = connects or breaks                            # prefer naming the connection that was made
    if outs:
        return join(outs[:2])
    d = elu_connected(pf, cf, mover, sign)               # else a value change on an existing chain
    return f"strengthens the pawn chain ({d})" if d else ""


def v_doubled(pf, cf, mover, sign):
    d = elu_doubled(pf, cf, mover, sign)
    if not d:
        return ""
    if "undoubled" in d:
        return pick(d, [f"repairs the doubled pawns ({d})", f"undoubles the pawns ({d})"])
    return pick(d, [f"saddles a side with doubled pawns ({d})", f"doubles the pawns ({d})"])


# ---- whole terms ----





