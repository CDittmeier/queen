"""THREATS term (evaluate.cpp): threats by minor/rook/king, hanging and restricted
pieces, safe pawn pushes/attacks, and pressure on the enemy queen.

Threats -> one node per attacking side (White/Black, signed); colour lives
at that node. Each side breaks down additively into the nine threat sub-factors,
whose weight tables live here.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO
from datagen.tree.primitives.base import Factor

# Indexed by victim piece type 1..5 (pawn..queen). SF declares these size-8 and
# zero-fills the rest; the KING slot (index 6) is only ever reached for an in-check
# position (which SF never evaluates) -- kept at S(0,0) to match and avoid a crash.
ThreatByMinor = [S(0, 0), S(6, 32), S(59, 41), S(79, 56), S(90, 119), S(79, 161), S(0, 0)]
ThreatByRook = [S(0, 0), S(3, 44), S(38, 71), S(38, 61), S(0, 38), S(51, 38), S(0, 0)]
ThreatByKing = S(24, 89)
Hanging = S(69, 36)
RestrictedPiece = S(7, 7)
ThreatBySafePawn = S(173, 94)
ThreatByPawnPush = S(48, 39)
KnightOnQueen = S(16, 12)
SliderOnQueen = S(59, 18)


def threats_term(board: chess.Board, ctx: Context, us: chess.Color) -> tuple[Score, list]:
    them = not us
    occ = ctx.occ
    A, A2 = ctx.att, ctx.att2
    nonpawn_enemies = board.occupied_co[them] & bb.bb_not(int(board.pawns))
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    defended = nonpawn_enemies & strongly
    weak = board.occupied_co[them] & bb.bb_not(strongly) & A[us]["ALL"]
    score = SCORE_ZERO
    factors = []

    def add(label, c):
        nonlocal score
        if c != SCORE_ZERO:
            score = score + c
            factors.append((label, c))

    if defended | weak:
        c = SCORE_ZERO
        for sq in chess.scan_forward((defended | weak) & (A[us][chess.KNIGHT] | A[us][chess.BISHOP])):
            c = c + ThreatByMinor[board.piece_type_at(sq)]
        add("threat by minor", c)
        c = SCORE_ZERO
        for sq in chess.scan_forward(weak & A[us][chess.ROOK]):
            c = c + ThreatByRook[board.piece_type_at(sq)]
        add("threat by rook", c)
        if weak & A[us][chess.KING]:
            add("threat by king", ThreatByKing)
        b = bb.bb_not(A[them]["ALL"]) | (nonpawn_enemies & A2[us])
        add("hanging", Hanging * bb.popcount(weak & b))

    b = A[them]["ALL"] & bb.bb_not(strongly) & A[us]["ALL"]
    add("restricted", RestrictedPiece * bb.popcount(b))

    safe = bb.bb_not(A[them]["ALL"]) | A[us]["ALL"]
    our_pawns = int(board.pawns & board.occupied_co[us])
    b = bb.pawn_attacks_bb(us, our_pawns & safe) & nonpawn_enemies
    add("safe pawn threat", ThreatBySafePawn * bb.popcount(b))

    trank3 = chess.BB_RANKS[2] if us == chess.WHITE else chess.BB_RANKS[5]
    b = bb.shift_up(us, our_pawns) & bb.bb_not(occ)
    b |= bb.shift_up(us, b & trank3) & bb.bb_not(occ)
    b &= bb.bb_not(A[them][chess.PAWN]) & safe
    b = bb.pawn_attacks_bb(us, b) & nonpawn_enemies
    add("pawn push threat", ThreatByPawnPush * bb.popcount(b))

    enemy_queens = board.pieces(chess.QUEEN, them)
    if len(enemy_queens) == 1:
        s = next(iter(enemy_queens))
        safe2 = ctx.mobility_area[us] & bb.bb_not(strongly)
        b = A[us][chess.KNIGHT] & chess.BB_KNIGHT_ATTACKS[s]
        add("knight on queen", KnightOnQueen * bb.popcount(b & safe2))
        b = (A[us][chess.BISHOP] & bb.bishop_attacks(s, occ)) | (A[us][chess.ROOK] & bb.rook_attacks(s, occ))
        add("slider on queen", SliderOnQueen * bb.popcount(b & safe2 & A2[us]))

    return score, factors



class Threats(Factor):
    tag = "threats"

    def score(self, board, ctx=None):
        ctx = ctx or get_context(board)
        total, breakdown = SCORE_ZERO, {}
        for color in (chess.WHITE, chess.BLACK):
            for name, s in threats_term(board, ctx, color)[1]:
                signed = s if color == chess.WHITE else -s
                total = total + signed
                breakdown[name] = breakdown.get(name, SCORE_ZERO) + signed
        return total, breakdown

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the threats changed across the move."""
        return self.run_phrasers({
            "safe pawn threat": (v_safe_pawn_threat, elu_safe_pawn_threat),
            "pawn push threat": (v_pawn_push_threat, elu_pawn_push_threat),
            "threat by minor": (v_piece_threat(minor_threats, "minor"), elu_threat_by_minor),
            "threat by rook": (v_piece_threat(rook_threats, "rook"), elu_threat_by_rook),
            "threat by king": (v_threat_by_king, elu_threat_by_king),
            "hanging": (v_hanging, elu_hanging),
            "restricted": (v_restricted, elu_restricted),
            "knight on queen": (v_queen_pressure(knight_on_queen_hops, "knight"), elu_knight_on_queen),
            "slider on queen": (v_queen_pressure(slider_on_queen_hops, "slider"), elu_slider_on_queen),
        }, before, after, mover, before_eval, after_eval)




# ============================================================================== verbalization primitives
from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, MOVED_FROM, derive_move, pcsym)


def safe_pawn_threats(fen, us):
    """{enemy-piece square: (attacking safe-pawn square, piece)} for side `us`,
    matching SF11: enemy non-pawn pieces attacked by our safe pawns."""
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A = ctx.att
    npe = b.occupied_co[them] & bb.bb_not(int(b.pawns))
    safe = bb.bb_not(A[them]["ALL"]) | A[us]["ALL"]
    sp = int(b.pawns & b.occupied_co[us]) & safe
    out = {}
    for sq in chess.scan_forward(bb.pawn_attacks_bb(us, sp) & npe):
        psq = next(iter(chess.scan_forward(sp & bb.pawn_attacks_bb(them, 1 << sq))), None)
        out[sq] = (psq, b.piece_at(sq))
    return out


def hanging_pieces(fen, us):
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, A2 = ctx.att, ctx.att2
    nonpawn = b.occupied_co[them] & bb.bb_not(int(b.pawns))
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    weak = b.occupied_co[them] & bb.bb_not(strongly) & A[us]["ALL"]
    b2 = bb.bb_not(A[them]["ALL"]) | (nonpawn & A2[us])
    out = {}
    for sq in chess.scan_forward(weak & b2):
        undef = not (A[them]["ALL"] & chess.BB_SQUARES[sq])
        out[sq] = (b.piece_type_at(sq), "undefended" if undef else "hit twice")
    return out


def minor_threats(fen, us):
    """{victim square: (victim pt, attacking-minor square)} -- enemy pieces a
    knight/bishop of `us` threatens (SF's defended|weak set)."""
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, A2 = ctx.att, ctx.att2
    nonpawn = b.occupied_co[them] & bb.bb_not(int(b.pawns))
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    defended = nonpawn & strongly
    weak = b.occupied_co[them] & bb.bb_not(strongly) & A[us]["ALL"]
    minors = A[us][chess.KNIGHT] | A[us][chess.BISHOP]
    out = {}
    for sq in chess.scan_forward((defended | weak) & minors):
        atk = int(b.attackers(us, sq)) & (int(b.knights) | int(b.bishops))
        asq = next(iter(chess.scan_forward(atk)), None)
        out[sq] = (b.piece_type_at(sq), asq)
    return out


def pawn_push_threats(fen, us):
    """{victim square: (push-destination sq, victim pt)} -- enemy non-pawn pieces a
    safe pawn push of `us` would attack (SF's pawn-push threat set)."""
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, occ = ctx.att, ctx.occ
    npe = b.occupied_co[them] & bb.bb_not(int(b.pawns))
    safe = bb.bb_not(A[them]["ALL"]) | A[us]["ALL"]
    our = int(b.pawns & b.occupied_co[us])
    up = 8 if us == chess.WHITE else -8
    trank3 = chess.BB_RANKS[2] if us == chess.WHITE else chess.BB_RANKS[5]
    p1 = bb.shift_up(us, our) & bb.bb_not(occ)
    p2 = bb.shift_up(us, p1 & trank3) & bb.bb_not(occ)
    pushes = (p1 | p2) & bb.bb_not(A[them][chess.PAWN]) & safe
    out = {}
    for v in chess.scan_forward(bb.pawn_attacks_bb(us, pushes) & npe):
        dsq = next(iter(chess.scan_forward(pushes & bb.pawn_attacks_bb(them, 1 << v))), None)
        if dsq is not None:
            origin = dsq - up if (our & (1 << (dsq - up))) else dsq - 2 * up
            out[v] = (dsq, origin, b.piece_type_at(v))
    return out


def restricted_sqs(fen, us):
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, A2 = ctx.att, ctx.att2
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    return set(chess.scan_forward(A[them]["ALL"] & bb.bb_not(strongly) & A[us]["ALL"]))


def rook_threats(fen, us):
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, A2 = ctx.att, ctx.att2
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    weak = b.occupied_co[them] & bb.bb_not(strongly) & A[us]["ALL"]
    out = {}
    for sq in chess.scan_forward(weak & A[us][chess.ROOK]):
        atk = next(iter(chess.scan_forward(int(b.attackers(us, sq)) & int(b.rooks))), None)
        out[sq] = (b.piece_type_at(sq), atk)
    return out


def knight_on_queen_hops(fen, us):
    """(enemy queen sq, {landing square: our knight that can jump there})."""
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    qs = list(b.pieces(chess.QUEEN, them))
    if len(qs) != 1:
        return None, {}
    q = qs[0]
    strongly = ctx.att[them][chess.PAWN] | (ctx.att2[them] & bb.bb_not(ctx.att2[us]))
    safe2 = ctx.mobility_area[us] & bb.bb_not(strongly)
    our_knights = int(b.knights & b.occupied_co[us])
    hops = ctx.att[us][chess.KNIGHT] & chess.BB_KNIGHT_ATTACKS[q] & safe2
    return q, {L: next(iter(chess.scan_forward(chess.BB_KNIGHT_ATTACKS[L] & our_knights)), None)
               for L in chess.scan_forward(hops)}


def slider_on_queen_hops(fen, us):
    """(enemy queen sq, {landing square: our bishop/rook that covers it})."""
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    occ = ctx.occ
    qs = list(b.pieces(chess.QUEEN, them))
    if len(qs) != 1:
        return None, {}
    q = qs[0]
    strongly = ctx.att[them][chess.PAWN] | (ctx.att2[them] & bb.bb_not(ctx.att2[us]))
    safe2 = ctx.mobility_area[us] & bb.bb_not(strongly)
    own = b.occupied_co[us]
    out = {}
    # a landing square on the queen's ROOK-lines needs a ROOK to hit her from there; a square on
    # her BISHOP-lines needs a BISHOP — match the piece TYPE to the line (a rook on a bishop
    # diagonal does not attack the queen).
    for pt, line in ((chess.ROOK, bb.rook_attacks(q, occ)), (chess.BISHOP, bb.bishop_attacks(q, occ))):
        movers = (int(b.rooks) if pt == chess.ROOK else int(b.bishops)) & own
        for L in chess.scan_forward(ctx.att[us][pt] & line & safe2 & ctx.att2[us]):
            atk = next(iter(chess.scan_forward(int(b.attackers(us, L)) & movers)), None)
            if atk is not None:
                out[L] = atk
    return q, out


def king_threats(fen, us):
    b = chess.Board(fen)
    ctx = get_context(b)
    them = not us
    A, A2 = ctx.att, ctx.att2
    strongly = A[them][chess.PAWN] | (A2[them] & bb.bb_not(A2[us]))
    weak = b.occupied_co[them] & bb.bb_not(strongly) & A[us]["ALL"]
    return {sq: b.piece_type_at(sq) for sq in chess.scan_forward(weak & A[us][chess.KING])}


def elu_safe_pawn_threat(parent_fen, child_fen, mover, sign):
    items = []                                        # (good_for_mover, text) -- two-sided
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        P, C = safe_pawn_threats(parent_fen, us), safe_pawn_threats(child_fen, us)
        for sq in set(C) - set(P):                     # a new safe-pawn threat (diagonal capture)
            psq, pc = C[sq]
            if psq is not None:
                items.append((new_good, f"{pfx}{chess.square_name(psq)}×{pcsym(pc.piece_type)}{chess.square_name(sq)}"))
        for sq in set(P) - set(C):                     # a safe-pawn threat lifted
            items.append((not new_good, f"{pfx}{pcsym(P[sq][1].piece_type)}{chess.square_name(sq)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)           # sign-aligned first
    return ", ".join(t for _, t in items[:2])


def elu_hanging(parent_fen, child_fen, mover, sign):
    pb, pa = hanging_pieces(parent_fen, mover), hanging_pieces(child_fen, mover)
    ob, oa = hanging_pieces(parent_fen, not mover), hanging_pieces(child_fen, not mover)
    mv = derive_move(parent_fen, child_fen)
    cap_sq = mv.to_square if mv is not None else None
    items = []                                        # (good_for_mover, text)
    for sq in set(pa) - set(pb):                      # enemy piece the move leaves hanging (good)
        pt, why = pa[sq]
        items.append((True, f"{pcsym(pt)}{chess.square_name(sq)} ({why})"))
    for sq in set(pb) - set(pa):                      # enemy hanging piece resolved (bad)
        word = "captured" if sq == cap_sq else "defended"
        items.append((False, f"{pcsym(pb[sq][0])}{chess.square_name(sq)} {word}"))
    for sq in set(oa) - set(ob):                      # own piece now hanging (bad)
        pt, why = oa[sq]
        items.append((False, f"own {pcsym(pt)}{chess.square_name(sq)} ({why})"))
    for sq in set(ob) - set(oa):                      # own piece no longer hanging (good)
        items.append((True, f"own {pcsym(ob[sq][0])}{chess.square_name(sq)} now safe"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)          # sign-aligned first
    return ", ".join(t for _, t in items[:2])


# ---- threat by minor: which minor threatens which enemy piece ----


def elu_threat_by_minor(parent_fen, child_fen, mover, sign):
    cb = chess.Board(child_fen)
    items = []                                        # (good_for_mover, text)
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        P, C = minor_threats(parent_fen, us), minor_threats(child_fen, us)
        for sq in set(C) - set(P):                    # a new minor threat
            vt, asq = C[sq]
            atk = (f"{pcsym(cb.piece_type_at(asq))}{chess.square_name(asq)}"
                   if asq is not None else "minor")
            items.append((new_good, f"{pfx}{atk}→{pcsym(vt)}{chess.square_name(sq)}"))
        for sq in set(P) - set(C):                    # a minor threat lifted
            vt, _ = P[sq]
            items.append((not new_good, f"{pfx}{pcsym(vt)}{chess.square_name(sq)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)          # sign-aligned first
    return ", ".join(t for _, t in items[:2])


# ---- pawn push threat: which enemy piece a pawn push would hit ----


def elu_pawn_push_threat(parent_fen, child_fen, mover, sign):
    items = []                                        # (good_for_mover, text)
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        P, C = pawn_push_threats(parent_fen, us), pawn_push_threats(child_fen, us)
        for v in set(C) - set(P):                     # a new push threat: pawn pushes FORWARD then hits v
            dsq, origin, pt = C[v]
            items.append((new_good, f"{pfx}{chess.square_name(origin)}-{chess.square_name(dsq)}→{pcsym(pt)}{chess.square_name(v)}"))
        for v in set(P) - set(C):                     # a push threat lifted
            items.append((not new_good, f"{pfx}{pcsym(P[v][2])}{chess.square_name(v)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)          # sign-aligned first
    return ", ".join(t for _, t in items[:2])


# ---- restricted: enemy-controlled squares we contest (not strongly held) ----


def elu_restricted(parent_fen, child_fen, mover, sign):
    good, bad = [], []                                # squares helping / hurting the mover's balance
    for us, mine in ((mover, True), (not mover, False)):
        P, C = restricted_sqs(parent_fen, us), restricted_sqs(child_fen, us)
        for s in C - P:                               # `us` restricts a new square
            (good if mine else bad).append(chess.square_name(s))
        for s in P - C:                               # `us` stops restricting a square
            (bad if mine else good).append(chess.square_name(s))
    picks = good if sign > 0 else bad
    if not picks:
        return ""
    return ("contests " if sign > 0 else "yields ") + ", ".join(sorted(set(picks))[:3])


# ---- pawn structure: which pawns changed connected / backward ----


def elu_threat_by_rook(parent_fen, child_fen, mover, sign):
    cb = chess.Board(child_fen)
    items = []
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        P, C = rook_threats(parent_fen, us), rook_threats(child_fen, us)
        for sq in set(C) - set(P):
            vt, asq = C[sq]
            atk = f"R{chess.square_name(asq)}" if asq is not None else "rook"
            items.append((new_good, f"{pfx}{atk}→{pcsym(vt)}{chess.square_name(sq)}"))
        for sq in set(P) - set(C):
            items.append((not new_good, f"{pfx}{pcsym(P[sq][0])}{chess.square_name(sq)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_queen_pressure(parent_fen, child_fen, mover, sign, fn, noun):
    """A latent threat: our {noun} can move to a safe square and attack the enemy queen."""
    cb = chess.Board(child_fen)
    items = []
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        qp, hp = fn(parent_fen, us)
        qc, hc = fn(child_fen, us)
        gained = set(hc) - set(hp)
        if gained and qc is not None:
            L = sorted(gained)[0]
            atk = hc[L]
            src = (f"{pcsym(cb.piece_type_at(atk))}{chess.square_name(atk)}→{chess.square_name(L)}"
                   if atk is not None else f"a {noun} to {chess.square_name(L)}")
            items.append((new_good, f"{pfx}{src} would hit Q{chess.square_name(qc)}"))
        elif set(hp) - set(hc) and qp is not None:
            items.append((not new_good, f"{pfx}Q{chess.square_name(qp)} out of the {noun}'s reach"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_knight_on_queen(parent_fen, child_fen, mover, sign):
    return elu_queen_pressure(parent_fen, child_fen, mover, sign, knight_on_queen_hops, "knight")


def elu_slider_on_queen(parent_fen, child_fen, mover, sign):
    return elu_queen_pressure(parent_fen, child_fen, mover, sign, slider_on_queen_hops, "slider")


# ---- passed pawns + space (whole terms) ----


def elu_threat_by_king(parent_fen, child_fen, mover, sign):
    cb = chess.Board(child_fen)
    items = []
    for us, pfx, new_good in ((mover, "", True), (not mover, "opp ", False)):
        P, C = king_threats(parent_fen, us), king_threats(child_fen, us)
        ksq = cb.king(us)
        for sq in set(C) - set(P):
            items.append((new_good, f"{pfx}K{chess.square_name(ksq)}×{pcsym(C[sq])}{chess.square_name(sq)}"))
        for sq in set(P) - set(C):
            items.append((not new_good, f"{pfx}{pcsym(P[sq])}{chess.square_name(sq)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def v_safe_pawn_threat(pf, cf, mover, sign):
    parts = []
    for us, opp in ((mover, False), (not mover, True)):
        P, C = safe_pawn_threats(pf, us), safe_pawn_threats(cf, us)
        for sq in set(C) - set(P):
            psq, pc = C[sq]
            if pc is None:
                continue
            v = f"the {piece_name(pc.piece_type, pc.color)} on {square_name(sq)}"
            pawn = f"the {piece_name(chess.PAWN, us)} on {square_name(psq)}" if psq is not None else f"a {piece_name(chess.PAWN, us)}"
            parts.append((not opp, f"lets {pawn} threaten {v}" if opp
                          else pick((sq, us), [f"threatens {v} with {pawn}", f"attacks {v} with {pawn}"])))
        for sq in set(P) - set(C):
            if sq in (CAPSQ[0], MOVED_FROM[0]):   # piece captured or simply moved away — obvious
                continue
            _, pc = P[sq]
            parts.append((opp, f"lifts the threat against the {piece_name(pc.piece_type, pc.color)} on {square_name(sq)}"))
    return sign_pick(parts, sign)


def v_pawn_push_threat(pf, cf, mover, sign):
    parts = []
    for us, opp in ((mover, False), (not mover, True)):
        P, C = pawn_push_threats(pf, us), pawn_push_threats(cf, us)
        for v in set(C) - set(P):
            dsq, origin, pt = C[v]
            tgt = f"the {piece_name(pt, not us)} on {square_name(v)}"
            pawn = f"the {piece_name(chess.PAWN, us)} on {square_name(origin)}" if origin is not None else f"a {piece_name(chess.PAWN, us)}"
            parts.append((not opp, f"lets {pawn} push to {square_name(dsq)} and hit {tgt}" if opp
                          else pick((v, us), [f"prepares to push {pawn} to {square_name(dsq)}, hitting {tgt}",
                                               f"threatens to advance {pawn} to {square_name(dsq)} and attack {tgt}"])))
        for v in set(P) - set(C):
            if v in (CAPSQ[0], MOVED_FROM[0]):   # piece captured or moved away — the threat is moot
                continue
            dsq, origin, pt = P[v]
            pawn = f"the {piece_name(chess.PAWN, us)} on {square_name(origin)}" if origin is not None else f"a {piece_name(chess.PAWN, us)}"
            parts.append((opp, f"removes the idea of pushing {pawn} to {square_name(dsq)}, "
                               f"threatening the {piece_name(pt, not us)} on {square_name(v)}"))
    return sign_pick(parts, sign)


def v_piece_threat(detector, atk_name):
    def f(pf, cf, mover, sign):
        cb, pb = chess.Board(cf), chess.Board(pf)
        parts = []
        for us, opp in ((mover, False), (not mover, True)):
            P, C = detector(pf, us), detector(cf, us)
            for sq in set(C) - set(P):
                vt, asq = C[sq]
                a = f"the {piece_name(cb.piece_type_at(asq), cb.color_at(asq))} on {square_name(asq)}" if asq is not None else f"a {atk_name}"
                v = f"the {piece_name(vt, cb.color_at(sq))} on {square_name(sq)}"
                parts.append((not opp, f"lets {a} attack {v}" if opp
                              else pick((sq, us), [f"attacks {v} with {a}", f"trains {a} on {v}", f"hits {v} with {a}"])))
            for sq in set(P) - set(C):
                vt, asq = P[sq]
                if sq in (CAPSQ[0], MOVED_FROM[0]):   # the threatened piece was captured or moved away
                    continue
                if asq is not None and asq == CAPSQ[0]:   # the attacking piece itself was captured (from that square)
                    continue
                atk_sq = asq if (asq is not None and pb.piece_at(asq)) else next(iter(pb.attackers(us, sq)), None)
                if atk_sq is None or not pb.piece_at(atk_sq) or atk_sq == CAPSQ[0]:
                    continue                              # can't name the attacker (or it was captured) — skip
                atk = f"the {piece_name(pb.piece_type_at(atk_sq), us)}"   # name the actual piece, not "minor"
                parts.append((opp, f"relieves {atk}'s pressure on the {piece_name(vt, not us)} on {square_name(sq)}"))
        return sign_pick(parts, sign)
    return f


def v_threat_by_king(pf, cf, mover, sign):
    cb = chess.Board(cf)
    parts = []
    for us, opp in ((mover, False), (not mover, True)):
        P, C = king_threats(pf, us), king_threats(cf, us)
        kp = f"{piece_name(chess.KING, us)} on {square_name(cb.king(us))}"   # token in token mode; "king on e1" in human
        for sq in set(C) - set(P):
            v = f"the {piece_name(C[sq], cb.color_at(sq))} on {square_name(sq)}"
            parts.append((not opp, f"lets {kp} attack {v}" if opp
                          else f"brings {kp} to bear on {v}"))
        # a king THREAT being lifted isn't clearly good or bad to a reader — don't state it.
    return sign_pick(parts, sign)


def v_hanging(pf, cf, mover, sign):
    pb, pa = hanging_pieces(pf, mover), hanging_pieces(cf, mover)
    ob, oa = hanging_pieces(pf, not mover), hanging_pieces(cf, not mover)
    parts = []
    for sq in set(pa) - set(pb):
        pt, why = pa[sq]
        parts.append((True, f"leaves {our(not mover)} {piece_name(pt, not mover)} on {square_name(sq)} hanging ({why})"))
    for sq in set(oa) - set(ob):
        pt, why = oa[sq]
        parts.append((False, f"leaves {our(mover)} {piece_name(pt, mover)} on {square_name(sq)} hanging ({why})"))
    for sq in set(ob) - set(oa):
        pt = ob[sq][0]
        if sq == MOVED_FROM[0]:                          # the piece moved off the hanging square
            parts.append((True, f"tucks {our(mover)} {piece_name(pt, mover)} back to safety"))
        else:                                            # it stayed put and got defended
            parts.append((True, f"defends {our(mover)} {piece_name(pt, mover)} on {square_name(sq)}"))
    return sign_pick(parts, sign)


def v_restricted(pf, cf, mover, sign):
    cbd = chess.Board(cf)
    good, bad = [], []
    for us, mine in ((mover, True), (not mover, False)):
        P, C = restricted_sqs(pf, us), restricted_sqs(cf, us)
        for s in C - P:                      # `us` now contests s -> space gained by `us`
            (good if mine else bad).append(square_name(s))
        for s in P - C:                      # `us` no longer contests s
            if cbd.is_attacked_by(us, s):    # still attacks it -> the enemy left (e.g. we took the
                continue                     # piece that was there); not `us` conceding space
            (bad if mine else good).append(square_name(s))
    picks = sorted(set(good if sign > 0 else bad))[:3]
    if len(picks) < 2:                                 # a lone square is noise, not "space"
        return ""
    where = join(picks)
    return f"gains space on {where}" if sign > 0 else f"gives up space on {where}"


def v_queen_pressure(fn, noun):
    def f(pf, cf, mover, sign):
        cb, pb = chess.Board(cf), chess.Board(pf)
        parts = []
        for us, opp in ((mover, False), (not mover, True)):
            qp, hp = fn(pf, us)
            qc, hc = fn(cf, us)
            gained = set(hc) - set(hp)
            if gained and qc is not None:
                L, atk = sorted(gained)[0], hc[sorted(gained)[0]]
                if atk is not None:
                    val = PV[cb.piece_type_at(atk)]        # moot if a cheaper enemy piece guards the landing
                    if any(PV[cb.piece_type_at(d)] < val for d in cb.attackers(not us, L)):
                        continue
                src = f"the {piece_name(cb.piece_type_at(atk), cb.color_at(atk))} on {square_name(atk)}" if atk is not None else f"a {noun}"
                q = f"{our(not us)} queen on {square_name(qc)}"
                parts.append((not opp, f"lets {src} swing to {square_name(L)} to hit {q}" if opp
                              else f"eyes {q}: {src} could swing to {square_name(L)} to hit it"))
            elif set(hp) - set(hc) and qp is not None:
                if qp == MOVED_FROM[0]:
                    continue                             # the queen moved off the pressured square — obvious
                lost = [hp[k] for k in set(hp) - set(hc) if hp[k] is not None]
                piece = (f"the {piece_name(pb.piece_type_at(lost[0]), us)} on {square_name(lost[0])}"
                         if lost and pb.piece_at(lost[0]) else f"the {noun}")
                parts.append((opp, f"relieves {piece}'s pressure on {our(not us)} queen on {square_name(qp)}"))
        return sign_pick(parts, sign)
    return f


# ---- pawn structure ----





