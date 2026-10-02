"""Per-piece positional terms (evaluate.cpp pieces<>), excluding mobility.

Decomposed all the way down: each piece-type term (knights / bishops / rooks / queens)
-> one evaluator class per positional factor (outpost, king-protector, rook-on-open-file,
weak-queen, ...) -> one atomic leaf per piece that has that factor (called out by square).
The positional weights live in the factor classes. Mobility is a separate term.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO
from datagen.tree.primitives.base import Factor

RookOnFile = [S(21, 4), S(47, 25)]
BishopPawns = S(3, 7)
KingProtector = S(7, 8)
LongDiagonalBishop = S(45, 0)
MinorBehindPawn = S(18, 3)
Outpost = S(30, 21)
ReachableOutpost = S(32, 10)
RookOnQueenFile = S(7, 6)
TrappedRook = S(52, 10)
WeakQueen = S(49, 15)

OUTPOST_RANKS = {
    chess.WHITE: chess.BB_RANKS[3] | chess.BB_RANKS[4] | chess.BB_RANKS[5],
    chess.BLACK: chess.BB_RANKS[2] | chess.BB_RANKS[3] | chess.BB_RANKS[4],
}
SYMBOL = {chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q"}


def piece_label(rec) -> str:
    sym = SYMBOL[rec.pt]
    return (sym if rec.color == chess.WHITE else sym.lower()) + chess.square_name(rec.sq)


def piece_score(board: chess.Board, ctx: Context, rec) -> tuple[Score, list]:
    """(score, [(factor, score)]) for one piece; factors partition the score."""
    us, pt, s, b = rec.color, rec.pt, rec.sq, rec.attacks
    them = not us
    ksq = board.king(us)
    occ = ctx.occ
    queens = int(board.queens)
    own_occ = board.occupied_co[us]
    all_pawns = int(board.pawns)
    own_pawns = int(board.pawns & board.occupied_co[us])
    enemy_pawns = int(board.pawns & board.occupied_co[them])
    score = SCORE_ZERO
    factors = []

    def add(label, c):
        nonlocal score
        if c != SCORE_ZERO:
            score = score + c
            factors.append((label, c))

    if pt in (chess.BISHOP, chess.KNIGHT):
        out_bb = OUTPOST_RANKS[us] & ctx.att[us][chess.PAWN] & bb.bb_not(ctx.pawn[them].pawn_attacks_span)
        if out_bb & chess.BB_SQUARES[s]:
            add("outpost", Outpost * (2 if pt == chess.KNIGHT else 1))
        elif pt == chess.KNIGHT and (out_bb & b & bb.bb_not(own_occ)):
            add("reachable outpost", ReachableOutpost)
        if bb.shift_down(us, all_pawns) & chess.BB_SQUARES[s]:
            add("minor behind pawn", MinorBehindPawn)
        add("king protector", -(KingProtector * bb.dist(s, ksq)))
        if pt == chess.BISHOP:
            blocked = own_pawns & bb.shift_down(us, occ)
            same = chess.BB_LIGHT_SQUARES if (chess.BB_SQUARES[s] & chess.BB_LIGHT_SQUARES) else chess.BB_DARK_SQUARES
            add("bishop pawns", -(BishopPawns * bb.popcount(own_pawns & same)
                                  * (1 + bb.popcount(blocked & bb.CENTER_FILES))))
            if bb.popcount(bb.bishop_attacks(s, all_pawns) & bb.CENTER) > 1:
                add("long diagonal", LongDiagonalBishop)

    if pt == chess.ROOK:
        if chess.BB_FILES[s & 7] & queens:
            add("rook on queen file", RookOnQueenFile)
        if not (own_pawns & chess.BB_FILES[s & 7]):
            add("rook on open/semi-open file",
                RookOnFile[0 if (enemy_pawns & chess.BB_FILES[s & 7]) else 1])
        elif rec.mob_count <= 3:
            kf = ksq & 7
            if (kf < 4) == ((s & 7) < kf):
                add("trapped rook", -(TrappedRook * (1 + (0 if board.has_castling_rights(us) else 1))))

    if pt == chess.QUEEN and ctx.weak_queen(us, s):
        add("weak queen", -WeakQueen)

    return score, factors



class PieceType(Factor):
    pt = chess.KNIGHT

    def score(self, board, ctx=None):
        """(total Score, {factor: Score}) over this piece type's pieces, netted White-POV."""
        ctx = ctx or get_context(board)
        total, breakdown = SCORE_ZERO, {}
        for rec in ctx.piece_recs_ordered(self.pt):
            for name, s in piece_score(board, ctx, rec)[1]:
                signed = s if rec.color == chess.WHITE else -s
                total = total + signed
                breakdown[name] = breakdown.get(name, SCORE_ZERO) + signed
        return total, breakdown





class Knights(PieceType):
    tag = "knights"
    ACTIVITY = ("knight outpost", "bad knight")

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the knights changed across the move."""
        return self.run_phrasers({
            "outpost": (None, elu_outpost),
            "reachable outpost": (None, elu_reachable_outpost),
            "minor behind pawn": (v_minor_behind_pawn, elu_minor_behind_pawn),
            "king protector": (v_king_protector, elu_king_protector),
        }, before, after, mover, before_eval, after_eval)
    pt = chess.KNIGHT


class Bishops(PieceType):
    tag = "bishops"
    ACTIVITY = ("active bishop pair", "active lone bishop", "bad bishop",
                "colour complexion")

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the bishops changed across the move."""
        return self.run_phrasers({
            "outpost": (None, elu_outpost),
            "minor behind pawn": (v_minor_behind_pawn, elu_minor_behind_pawn),
            "king protector": (v_king_protector, elu_king_protector),
            "bishop pawns": (v_bishop_pawns, elu_bishop_pawns),
            "long diagonal": (v_long_diagonal, elu_long_diagonal),
        }, before, after, mover, before_eval, after_eval)
    pt = chess.BISHOP


class Rooks(PieceType):
    tag = "rooks"
    ACTIVITY = ("bad rook", "active rook")

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the rooks changed across the move."""
        return self.run_phrasers({
            "rook on open/semi-open file": (v_open_file, elu_open_file),
            "rook on queen file": (None, elu_rook_on_queen_file),
            "trapped rook": (v_trapped_rook, elu_trapped_rook),
        }, before, after, mover, before_eval, after_eval)
    pt = chess.ROOK


class Queens(PieceType):
    tag = "queens"

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how the queen changed across the move."""
        return self.run_phrasers({
            "weak queen": (v_weak_queen, elu_weak_queen),
        }, before, after, mover, before_eval, after_eval)
    pt = chess.QUEEN


# ============================================================================== verbalization primitives
from datagen.tree.primitives import pieces as pieces_mod
from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, derive_move, pcsym)


def bishop_pawn_buckets(fen):
    """(side, is_light) -> (own pawns on that colour, blocked central pawns, has such bishop)."""
    b = chess.Board(fen)
    occ = get_context(b).occ
    out = {}
    for color in (chess.WHITE, chess.BLACK):
        own = int(b.pawns & b.occupied_co[color])
        nb = bb.popcount(own & bb.shift_down(color, occ) & bb.CENTER_FILES)
        for is_light in (True, False):
            same = chess.BB_LIGHT_SQUARES if is_light else chess.BB_DARK_SQUARES
            has = any((1 << s) & same for s in b.pieces(chess.BISHOP, color))
            out[(color, is_light)] = (bb.popcount(own & same), nb, has)
    return out


def minor_behind_pawn(fen):
    b = chess.Board(fen)
    all_pawns = int(b.pawns)
    out = set()
    for col in (chess.WHITE, chess.BLACK):
        shadow = bb.shift_down(col, all_pawns)        # squares with a pawn directly ahead
        for pt in (chess.KNIGHT, chess.BISHOP):
            for s in b.pieces(pt, col):
                if shadow & (1 << s):
                    out.add((s, col, pt))
    return out


def outpost_pieces(fen):
    """(on, reach): minors ON an outpost, and knights that can REACH one."""
    b = chess.Board(fen)
    ctx = get_context(b)
    on, reach = set(), set()
    for rec in ctx.pieces:
        if rec.pt not in (chess.KNIGHT, chess.BISHOP):
            continue
        us, them = rec.color, not rec.color
        outbb = (pieces_mod.OUTPOST_RANKS[us] & ctx.att[us][chess.PAWN]
                 & bb.bb_not(ctx.pawn[them].pawn_attacks_span))
        if outbb & (1 << rec.sq):
            on.add((rec.sq, us, rec.pt))
        elif rec.pt == chess.KNIGHT:
            land = outbb & rec.attacks & bb.bb_not(b.occupied_co[us])
            if land:
                reach.add((rec.sq, us, next(iter(chess.scan_forward(land)))))
    return on, reach


def trapped_rooks(fen):
    b = chess.Board(fen)
    out = set()
    for rec in get_context(b).pieces:
        if rec.pt != chess.ROOK:
            continue
        us, s = rec.color, rec.sq
        own_pawns = int(b.pawns & b.occupied_co[us])
        if (own_pawns & chess.BB_FILES[s & 7]) and rec.mob_count <= 3:
            kf = b.king(us) & 7
            if (kf < 4) == ((s & 7) < kf):
                out.add((s, us))
    return out


def open_file_rooks(fen):
    b = chess.Board(fen)
    out = {}
    for col in (chess.WHITE, chess.BLACK):
        own = int(b.pawns & b.occupied_co[col])
        enemy = int(b.pawns & b.occupied_co[not col])
        for s in b.pieces(chess.ROOK, col):
            if not (own & chess.BB_FILES[s & 7]):
                out[(s, col)] = "open" if not (enemy & chess.BB_FILES[s & 7]) else "semi-open"
    return out


def weak_queens(fen):
    b = chess.Board(fen)
    ctx = get_context(b)
    return {(s, col) for col in (chess.WHITE, chess.BLACK)
            for s in b.pieces(chess.QUEEN, col) if ctx.weak_queen(col, s)}


def weak_queen_detail(fen, color, qsq):
    """The enemy rook/bishop x-raying the queen through exactly one blocker, if any."""
    b = chess.Board(fen)
    them = not color
    occ = int(b.occupied)
    qr, qf = qsq >> 3, qsq & 7
    snipers = 0
    for sq in b.pieces(chess.ROOK, them):
        if (sq >> 3) == qr or (sq & 7) == qf:
            snipers |= 1 << sq
    for sq in b.pieces(chess.BISHOP, them):
        if abs((sq >> 3) - qr) == abs((sq & 7) - qf):
            snipers |= 1 << sq
    occupancy = occ ^ snipers
    for sq in chess.scan_forward(snipers):
        bt = int(chess.between(qsq, sq)) & occupancy
        if bt and (bt & (bt - 1)) == 0:
            return sq, next(iter(chess.scan_forward(bt)))
    return None, None


def long_diag_bishops(fen):
    b = chess.Board(fen)
    all_pawns = int(b.pawns)
    out = set()
    for col in (chess.WHITE, chess.BLACK):
        for s in b.pieces(chess.BISHOP, col):
            if bb.popcount(bb.bishop_attacks(s, all_pawns) & bb.CENTER) > 1:
                out.add((s, col))
    return out


def rook_queen_file(fen):
    b = chess.Board(fen)
    queens = int(b.queens)
    out = {}
    for col in (chess.WHITE, chess.BLACK):
        for s in b.pieces(chess.ROOK, col):
            qf = chess.BB_FILES[s & 7] & queens
            if qf:
                out[(s, col)] = next(iter(chess.scan_forward(qf)))
    return out


def elu_bishop_pawns(parent_fen, child_fen, mover, sign):
    P, C = bishop_pawn_buckets(parent_fen), bishop_pawn_buckets(child_fen)
    best = None                                       # (mover_benefit, key, ns_p, ns_c, nb_p, nb_c)
    for key in P:
        pc = key[0]
        ns_p, nb_p, has_p = P[key]
        ns_c, nb_c, has_c = C.get(key, (0, 0, False))
        if not (has_p and has_c):
            continue
        d = ns_c * (1 + nb_c) - ns_p * (1 + nb_p)     # rise in penalty magnitude (more pawns = worse)
        if d == 0:
            continue
        mb = -d if pc == mover else d                 # mover benefits when the OPP bishop is hampered
        if best is None or abs(mb) > abs(best[0]):
            best = (mb, key, ns_p, ns_c, nb_p, nb_c)
    if best is None or (best[0] > 0) != (sign > 0):
        return ""
    _, (pc, is_light), ns_p, ns_c, nb_p, nb_c = best
    who = "White's" if pc == chess.WHITE else "Black's"
    colr = "light" if is_light else "dark"
    if ns_c != ns_p:
        return f"{who} {colr}-square bishop: {'more' if ns_c > ns_p else 'fewer'} pawns on {colr}"
    return f"{who} {colr}-square bishop: central pawns {'blocked' if nb_c > nb_p else 'freed'}"


def elu_minor_behind_pawn(parent_fen, child_fen, mover, sign):
    P, C = minor_behind_pawn(parent_fen), minor_behind_pawn(child_fen)
    items = []
    for s, col, pt in C - P:
        pawn = chess.square_name(s + (8 if col == chess.WHITE else -8))
        items.append((col == mover, f"{'' if col == mover else 'opp '}{pcsym(pt)}{chess.square_name(s)} behind pawn {pawn}"))
    for s, col, pt in P - C:
        pawn = chess.square_name(s + (8 if col == chess.WHITE else -8))
        items.append((col != mover, f"{'' if col == mover else 'opp '}{pcsym(pt)}{chess.square_name(s)} lost its {pawn} shield"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_outpost(parent_fen, child_fen, mover, sign):
    Pon, _ = outpost_pieces(parent_fen)
    Con, _ = outpost_pieces(child_fen)
    items = []
    for s, col, pt in Con - Pon:
        items.append((col == mover, f"{'' if col == mover else 'opp '}{pcsym(pt)}{chess.square_name(s)} on an outpost"))
    for s, col, pt in Pon - Con:
        items.append((col != mover, f"{'' if col == mover else 'opp '}{pcsym(pt)}{chess.square_name(s)} off its outpost"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_reachable_outpost(parent_fen, child_fen, mover, sign):
    _, Pr = outpost_pieces(parent_fen)
    _, Cr = outpost_pieces(child_fen)
    Pd = {(s, col): land for s, col, land in Pr}
    Cd = {(s, col): land for s, col, land in Cr}
    items = []
    for k in set(Cd) - set(Pd):
        s, col = k
        items.append((col == mover, f"{'' if col == mover else 'opp '}N{chess.square_name(s)}→{chess.square_name(Cd[k])} outpost"))
    for k in set(Pd) - set(Cd):
        s, col = k
        items.append((col != mover, f"{'' if col == mover else 'opp '}N{chess.square_name(s)} loses the {chess.square_name(Pd[k])} outpost"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_trapped_rook(parent_fen, child_fen, mover, sign):
    P, C = trapped_rooks(parent_fen), trapped_rooks(child_fen)
    items = []
    for s, col in C - P:
        items.append((col != mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} trapped"))
    for s, col in P - C:
        items.append((col == mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} freed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_open_file(parent_fen, child_fen, mover, sign):
    P, C = open_file_rooks(parent_fen), open_file_rooks(child_fen)
    items = []
    for k in set(C) - set(P):
        s, col = k
        items.append((col == mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} on the {C[k]} {chess.FILE_NAMES[s & 7]}-file"))
    for k in set(P) - set(C):
        s, col = k
        items.append((col != mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} leaves the {chess.FILE_NAMES[s & 7]}-file"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def elu_weak_queen(parent_fen, child_fen, mover, sign):
    P, C = weak_queens(parent_fen), weak_queens(child_fen)
    items = []
    for s, col in C - P:
        who = "own " if col == mover else "opp "
        sn, blk = weak_queen_detail(child_fen, col, s)
        if sn is not None:
            cb = chess.Board(child_fen)
            desc = (f"exposed to {pcsym(cb.piece_type_at(sn))}{chess.square_name(sn)}"
                    f" (only {pcsym(cb.piece_type_at(blk))}{chess.square_name(blk)} between)")
        else:
            desc = "exposed to an enemy slider"
        items.append((col != mover, f"{who}Q{chess.square_name(s)} {desc}"))
    for s, col in P - C:
        who = "own " if col == mover else "opp "
        items.append((col == mover, f"{who}Q{chess.square_name(s)} no longer x-rayed"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


# ---- threat by rook + knight/slider pressure on the enemy queen ----


def elu_long_diagonal(parent_fen, child_fen, mover, sign):
    P, C = long_diag_bishops(parent_fen), long_diag_bishops(child_fen)
    items = []
    for s, col in C - P:
        items.append((col == mover, f"{'' if col == mover else 'opp '}B{chess.square_name(s)} takes a long central diagonal"))
    for s, col in P - C:
        items.append((col != mover, f"{'' if col == mover else 'opp '}B{chess.square_name(s)} off the long diagonal"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


def long_diagonal_name(s):
    """The long central diagonal a bishop on `s` sits on, named by its two corner
    squares (datagen line convention: two endpoints; tok_free maps them to tokens)."""
    a, b = ("a1", "h8") if chess.square_file(s) == chess.square_rank(s) else ("h1", "a8")
    return f"{a}-{b}"


def v_long_diagonal(pf, cf, mover, sign):
    P, C = long_diag_bishops(pf), long_diag_bishops(cf)
    parts = []
    for s, col in C - P:                   # a bishop swings onto its long central diagonal
        parts.append((col == mover, f"opens up the {piece_name(chess.BISHOP, col)} ({square_name(s)})'s "
                                     f"long diagonal ({long_diagonal_name(s)})"))
    for s, col in P - C:                   # a bishop is shut off its long diagonal
        parts.append((col != mover, f"shuts the {piece_name(chess.BISHOP, col)} ({square_name(s)}) off its "
                                     f"long diagonal ({long_diagonal_name(s)})"))
    return sign_pick(parts, sign)


def elu_rook_on_queen_file(parent_fen, child_fen, mover, sign):
    P, C = rook_queen_file(parent_fen), rook_queen_file(child_fen)
    items = []
    for k in set(C) - set(P):
        s, col = k
        items.append((col == mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} lines up with Q{chess.square_name(C[k])}"))
    for k in set(P) - set(C):
        s, col = k
        items.append((col != mover, f"{'' if col == mover else 'opp '}R{chess.square_name(s)} off the queen's file"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])





def elu_king_protector(parent_fen, child_fen, mover, sign):
    """Penalty ~ a minor's distance to its own king: name the minor that moved
    nearer/farther, or the king walking toward/away from its minors."""
    m = derive_move(parent_fen, child_fen)
    if m is None:
        return ""
    pb, cb = chess.Board(parent_fen), chess.Board(child_fen)
    pc = pb.piece_at(m.from_square)
    if pc is None:
        return ""
    if pc.piece_type in (chess.KNIGHT, chess.BISHOP):   # only the minor-moves case: a king move changes
        dp = bb.dist(m.from_square, pb.king(pc.color))  # these distances only incidentally, not usefully
        dc = bb.dist(m.to_square, cb.king(pc.color))
        if dc != dp:
            return f"{pcsym(pc.piece_type)}{chess.square_name(m.to_square)} {'nearer' if dc < dp else 'farther from'} its king"
    return ""




# ---- polished phrasers (direct clauses, no "verb the READABLE (raw)" wrapper) ----
def v_trapped_rook(pf, cf, mover, sign):
    P, C = trapped_rooks(pf), trapped_rooks(cf)
    parts = []
    for s, col in C - P:
        parts.append((col != mover, f"traps the {piece_name(chess.ROOK, col)} on {square_name(s)}"))
    for s, col in P - C:
        parts.append((col == mover, f"frees the {piece_name(chess.ROOK, col)} on {square_name(s)}"))
    return sign_pick(parts, sign)


def v_king_protector(pf, cf, mover, sign):
    m = derive_move(pf, cf)
    if m is None:
        return ""
    pb, cb = chess.Board(pf), chess.Board(cf)
    pc = pb.piece_at(m.from_square)
    if pc is None:
        return ""
    if pc.piece_type in (chess.KNIGHT, chess.BISHOP):   # only the minor-moves case (a king move changes
        dp, dc = bb.dist(m.from_square, pb.king(pc.color)), bb.dist(m.to_square, cb.king(pc.color))
        if dc != dp:                                    # these distances only incidentally, not usefully)
            pn, sq = piece_name(pc.piece_type, pc.color), square_name(m.to_square)
            return f"brings the {pn} on {sq} nearer its king" if dc < dp else f"pulls the {pn} on {sq} away from its king"
    return ""


def v_bishop_pawns(pf, cf, mover, sign):
    P, C = bishop_pawn_buckets(pf), bishop_pawn_buckets(cf)
    best = None
    for key in P:
        pc = key[0]
        ns_p, nb_p, has_p = P[key]
        ns_c, nb_c, has_c = C.get(key, (0, 0, False))
        if not (has_p and has_c):
            continue
        dd = ns_c * (1 + nb_c) - ns_p * (1 + nb_p)
        if dd == 0:
            continue
        mb = -dd if pc == mover else dd
        if best is None or abs(mb) > abs(best[0]):
            best = (mb, key, ns_p, ns_c)
    if best is None or (best[0] > 0) != (sign > 0):
        return ""
    _, (pc, is_light), ns_p, ns_c = best
    if ns_c == ns_p:
        return ""                                        # only SF's blocked-pawns amplifier moved — too subtle
    who = "White's" if pc == chess.WHITE else "Black's"
    colr = "light" if is_light else "dark"
    if ns_c > ns_p:
        return f"hems in {who} {colr}-square bishop with more pawns on {colr} squares"
    return f"frees {who} {colr}-square bishop, fewer pawns on {colr} squares"


def v_weak_queen(pf, cf, mover, sign):
    P, C = weak_queens(pf), weak_queens(cf)
    cb = chess.Board(cf)
    parts = []
    for s, col in C - P:
        sn, blk = weak_queen_detail(cf, col, s)
        q = f"{our(col)} {piece_name(chess.QUEEN, col)} on {square_name(s)}"
        if sn is not None:
            sniper = f"the {piece_name(cb.piece_type_at(sn), cb.color_at(sn))} on {square_name(sn)}"
            blkr = f"the {piece_name(cb.piece_type_at(blk), cb.color_at(blk))} on {square_name(blk)}"
            parts.append((col != mover, f"leaves {q} exposed to {sniper} with only {blkr} between"))
        else:
            parts.append((col != mover, f"leaves {q} exposed to an enemy slider"))
    for s, col in P - C:
        q = f"{our(col)} {piece_name(chess.QUEEN, col)} on {square_name(s)}"
        parts.append((col == mover, f"shelters {q} from the x-ray"))
    return sign_pick(parts, sign)


def v_open_file(pf, cf, mover, sign):
    P, C = open_file_rooks(pf), open_file_rooks(cf)
    parts = []
    for k in set(C) - set(P):
        s, col = k
        parts.append((col == mover, f"puts the {piece_name(chess.ROOK, col)} on {square_name(s)} "
                                    f"on the {C[k]} {chess.FILE_NAMES[s & 7]}-file"))
    for k in set(P) - set(C):
        s, col = k
        parts.append((col != mover, f"takes the {piece_name(chess.ROOK, col)} on {square_name(s)} "
                                    f"off the {chess.FILE_NAMES[s & 7]}-file"))
    return sign_pick(parts, sign)


def v_minor_behind_pawn(pf, cf, mover, sign):
    # A minor sheltered behind a pawn is a subtle SF bonus that neither attacks nor restricts
    # anything the reader can see — too confusing to state, so we don't.
    return ""
