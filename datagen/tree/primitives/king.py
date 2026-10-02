"""KING (safety) term: pawn shelter/storm minus the king-danger transform,
pawnless-flank, and flank-attack penalties (pawns.cpp shelter + evaluate.cpp king).

King -> one node per king (White/Black, signed). Color sits at that node
(a king is intrinsically one colour); the term logic is otherwise color-agnostic.
Each king node breaks down (additively, in cp) into shelter, king-danger, pawnless
flank, and flank attacks; the king-danger integer composition is attached as detail.
All weight tables live here.
"""
from __future__ import annotations

import chess
import re

from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import S, Score, SCORE_ZERO, cdiv
from datagen.tree.primitives.base import Factor
from datagen.tree.primitives.mobility import mobility_score

# Shelter / storm (pawns.cpp).
BlockedStorm = S(82, 82)
ShelterStrength = [
    [-6, 81, 93, 58, 39, 18, 25, 0],
    [-43, 61, 35, -49, -29, -11, -63, 0],
    [-10, 75, 23, -2, 32, 3, -45, 0],
    [-39, -13, -29, -52, -48, -67, -166, 0],
]
UnblockedStorm = [
    [85, -289, -166, 97, 50, 45, 50, 0],
    [46, -25, 122, 45, 37, -10, 20, 0],
    [-6, 51, 168, 34, -2, -22, -14, 0],
    [-15, -11, 101, 4, 11, -15, -29, 0],
]

# King danger (evaluate.cpp).
KING_ATTACK_WEIGHTS = {chess.KNIGHT: 81, chess.BISHOP: 52, chess.ROOK: 44, chess.QUEEN: 10}
ROOK_SAFE_CHECK, QUEEN_SAFE_CHECK, BISHOP_SAFE_CHECK, KNIGHT_SAFE_CHECK = 1080, 780, 635, 790
PawnlessFlank = S(17, 95)
FlankAttacks = S(8, 0)
CAMP = {
    chess.WHITE: bb.MASK64 ^ chess.BB_RANKS[5] ^ chess.BB_RANKS[6] ^ chess.BB_RANKS[7],
    chess.BLACK: bb.MASK64 ^ chess.BB_RANKS[0] ^ chess.BB_RANKS[1] ^ chess.BB_RANKS[2],
}
KING_FLANK = [
    bb.QUEENSIDE ^ chess.BB_FILES[3], bb.QUEENSIDE, bb.QUEENSIDE,
    bb.CENTER_FILES, bb.CENTER_FILES,
    bb.KINGSIDE, bb.KINGSIDE, bb.KINGSIDE ^ chess.BB_FILES[4],
]


def evaluate_shelter(board: chess.Board, color: chess.Color, ksq: int) -> Score:
    them = not color
    all_pawns = int(board.pawns)
    b = all_pawns & bb.bb_not(bb.forward_ranks(them, ksq))
    our_pawns = b & board.occupied_co[color]
    their_pawns = b & board.occupied_co[them]
    bonus = S(5, 5)
    center = min(max(ksq & 7, 1), 6)
    for f in (center - 1, center, center + 1):
        fb = chess.BB_FILES[f]
        ob = our_pawns & fb
        our_rank = bb.rel_rank(color, bb.frontmost(them, ob)) if ob else 0
        tb = their_pawns & fb
        their_rank = bb.rel_rank(color, bb.frontmost(them, tb)) if tb else 0
        d = min(f, 7 - f)
        bonus += S(ShelterStrength[d][our_rank], 0)
        if our_rank and our_rank == their_rank - 1:
            bonus -= BlockedStorm * (1 if their_rank == 2 else 0)
        else:
            bonus -= S(UnblockedStorm[d][their_rank], 0)
    return bonus


def king_safety_shelter(board: chess.Board, color: chess.Color) -> Score:
    ksq = board.king(color)
    shelter = evaluate_shelter(board, color, ksq)
    if board.has_kingside_castling_rights(color):
        cand = evaluate_shelter(board, color, 6 if color == chess.WHITE else 62)
        if cand.mg > shelter.mg:
            shelter = cand
    if board.has_queenside_castling_rights(color):
        cand = evaluate_shelter(board, color, 2 if color == chess.WHITE else 58)
        if cand.mg > shelter.mg:
            shelter = cand
    pawns = int(board.pawns & board.occupied_co[color])
    min_dist = 8 if pawns else 0
    if pawns & chess.BB_KING_ATTACKS[ksq]:
        min_dist = 1
    else:
        for ps in chess.scan_forward(pawns):
            min_dist = min(min_dist, bb.dist(ksq, ps))
    return shelter - S(0, 16 * min_dist)


def king_term(board: chess.Board, ctx: Context, us: chess.Color) -> tuple[Score, list, dict]:
    them = not us
    ksq = board.king(us)
    occ = ctx.occ
    own_queens = int(board.queens & board.occupied_co[us])
    A, A2, KR = ctx.att, ctx.att2, ctx.king_ring

    shelter = king_safety_shelter(board, us)
    score = shelter

    weak = (A[them]["ALL"] & bb.bb_not(A2[us])
            & (bb.bb_not(A[us]["ALL"]) | A[us][chess.KING] | A[us][chess.QUEEN]))
    safe = bb.bb_not(board.occupied_co[them])
    safe &= bb.bb_not(A[us]["ALL"]) | (weak & A2[them])

    b1 = bb.rook_attacks(ksq, occ ^ own_queens)
    b2 = bb.bishop_attacks(ksq, occ ^ own_queens)

    unsafe = 0
    safe_checks = 0
    rook_checks = b1 & safe & A[them][chess.ROOK]
    if rook_checks:
        safe_checks += ROOK_SAFE_CHECK
    else:
        unsafe |= b1 & A[them][chess.ROOK]
    queen_checks = (b1 | b2) & A[them][chess.QUEEN] & safe & bb.bb_not(A[us][chess.QUEEN]) & bb.bb_not(rook_checks)
    if queen_checks:
        safe_checks += QUEEN_SAFE_CHECK
    bishop_checks = b2 & A[them][chess.BISHOP] & safe & bb.bb_not(queen_checks)
    if bishop_checks:
        safe_checks += BISHOP_SAFE_CHECK
    else:
        unsafe |= b2 & A[them][chess.BISHOP]
    knight_checks = chess.BB_KNIGHT_ATTACKS[ksq] & A[them][chess.KNIGHT]
    if knight_checks & safe:
        safe_checks += KNIGHT_SAFE_CHECK
    else:
        unsafe |= knight_checks

    safe_check_sqs = {}
    if rook_checks:
        safe_check_sqs["R"] = int(rook_checks)
    if queen_checks:
        safe_check_sqs["Q"] = int(queen_checks)
    if bishop_checks:
        safe_check_sqs["B"] = int(bishop_checks)
    if knight_checks & safe:
        safe_check_sqs["N"] = int(knight_checks & safe)

    unsafe_check_sqs = {}  # unsafe checks that fed `unsafe` above (rook/bishop/knight only)
    if not rook_checks and (b1 & A[them][chess.ROOK]):
        unsafe_check_sqs["R"] = int(b1 & A[them][chess.ROOK])
    if not bishop_checks and (b2 & A[them][chess.BISHOP]):
        unsafe_check_sqs["B"] = int(b2 & A[them][chess.BISHOP])
    if not (knight_checks & safe) and knight_checks:
        unsafe_check_sqs["N"] = int(knight_checks)

    flank = KING_FLANK[ksq & 7]
    camp = CAMP[us]
    c1 = A[them]["ALL"] & flank & camp
    c2 = c1 & A2[them]
    c3 = A[us]["ALL"] & flank & camp
    kfa = bb.popcount(c1) + bb.popcount(c2)
    kfd = bb.popcount(c3)

    kaw_them = sum(KING_ATTACK_WEIGHTS[pt] for pt in ctx.king_attacker_pts[them])
    # king_danger as a sum of named contributions (order/grouping is exact — sum is unchanged)
    kd_components = {
        "safe checks": safe_checks,
        "attackers": ctx.kac[them] * kaw_them,
        "weak king-ring": 185 * bb.popcount(KR[us] & weak),
        "unsafe checks": 148 * bb.popcount(unsafe),
        "king pins": 98 * bb.popcount(bb.blockers_for_king(board, us)),
        "king-ring attacks": 69 * ctx.katt[them],
        "flank attack": cdiv(3 * kfa * kfa, 8),
        "mobility": (mobility_score(ctx, them) - mobility_score(ctx, us)).mg,
        "no enemy queen": -873 * (0 if board.pieces(chess.QUEEN, them) else 1),
        "knight defender": -100 * (1 if (A[us][chess.KNIGHT] & A[us][chess.KING]) else 0),
        "shelter bonus": -cdiv(6 * shelter.mg, 8),
        "flank defense": -4 * kfd,
        "tempo": 37,
    }
    king_danger = sum(kd_components.values())

    factors = [("shelter/storm", shelter)]
    danger_pen = SCORE_ZERO
    if king_danger > 100:
        danger_pen = S(cdiv(king_danger * king_danger, 4096), cdiv(king_danger, 16))
        score -= danger_pen
        factors.append(("king danger", -danger_pen))
    if not (board.pawns & flank):
        score -= PawnlessFlank
        factors.append(("pawnless flank", -PawnlessFlank))
    flank_pen = FlankAttacks * kfa
    if flank_pen != SCORE_ZERO:
        score -= flank_pen
        factors.append(("flank attacks", -flank_pen))

    detail = {"king_danger": king_danger, "kfa": kfa, "kfd": kfd,
              "unsafe_checks": bb.popcount(unsafe), "components": kd_components,
              "safe_check_sqs": safe_check_sqs, "unsafe_check_sqs": unsafe_check_sqs,
              "flank_atk_sqs": int(c1)}
    return score, factors, detail



class King(Factor):
    tag = "king safety"

    def score(self, board, ctx=None):
        """(total Score, {sub_factor: Score}) for king safety, both sides netted White-POV."""
        ctx = ctx or get_context(board)
        total, breakdown = SCORE_ZERO, {}
        for color in (chess.WHITE, chess.BLACK):
            for name, s in king_term(board, ctx, color)[1]:
                signed = s if color == chess.WHITE else -s
                total = total + signed
                breakdown[name] = breakdown.get(name, SCORE_ZERO) + signed
        return total, breakdown

    def describe(self, before, after, move, mover, before_eval, after_eval):
        """{sub_factor: description-or-None} — how king safety changed across the move."""
        return self.run_phrasers({
            "king danger": (v_king_danger, elu_king_danger),
            "flank attacks": (None, elu_flank_attacks),
            "shelter/storm": (None, elu_shelter),
            "pawnless flank": (None, elu_pawnless_flank),
        }, before, after, mover, before_eval, after_eval)




# ============================================================================== verbalization primitives
from datagen.tree.primitives import king as king_mod
from datagen.tree.primitives.core import bitboard as bb
from datagen.tree.primitives.core.context import get_context as get_context
from datagen.tree.primitives.common import (PV, PIECE, piece_name, square_name, file_letter, who, our, obj, fix_ownopp,
                     pick, join, join_contrast, sign_pick, captured_sq,
                     notation, PERSP, CAPSQ, derive_move, pcsym)


KD_TMPL = {
    "attackers":         ("more attackers on",       "fewer attackers on"),
    "weak king-ring":    ("more attacked squares around", "fewer attacked squares around"),
    "unsafe checks":     ("prepares a check against", "removes a check against"),
    "safe checks":       ("prepares a check against", "removes a check against"),
    "king pins":         ("a pin against",           "pin relieved on"),
    "king-ring attacks": ("more attacks around",     "fewer attacks around"),
    "flank attack":      ("a flank attack on",       "eased flank attack on"),
    "mobility":          ("more enemy activity vs",  "less enemy activity vs"),
    "no enemy queen":    ("attacking queen back vs", "queen traded off near"),
    "knight defender":   ("lost knight guard of",    "knight now guards"),
    "shelter bonus":     ("weakened shelter of",     "improved shelter of"),
    "flank defense":     ("less flank defense for",  "more flank defense for"),
}


def kd_pen_comp(fen, us):
    b = chess.Board(fen)
    _, _, det = king_mod.king_term(b, get_context(b), us)
    kd = det["king_danger"]
    return ((kd * kd) // 4096 if kd > 100 else 0), det["components"]


def king_ring_attackers(fen, us):
    b = chess.Board(fen)
    ring = get_context(b).king_ring[us]
    them = not us
    out = set()
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in b.pieces(pt, them):
            if int(b.attacks(sq)) & ring:
                out.add((chess.piece_symbol(pt).upper(), chess.square_name(sq)))
    return out


SC_NAMES = {"R": "rook", "Q": "queen", "B": "bishop", "N": "knight"}


SC_PT = {"R": chess.ROOK, "Q": chess.QUEEN, "B": chess.BISHOP, "N": chess.KNIGHT}


def flank_atk_sqs(fen, k):
    b = chess.Board(fen)
    return king_mod.king_term(b, get_context(b), k)[2]["flank_atk_sqs"]


def flank_change_pieces(parent_fen, child_fen, k, direction):
    """Enemy pieces responsible for the CHANGE in king k's flank attack: those
    covering the flank squares whose attacked-status flipped — not the static
    long-range pieces already covering the flank before the move."""
    pb, ca = flank_atk_sqs(parent_fen, k), flank_atk_sqs(child_fen, k)
    if direction > 0:
        changed, pos = ca & ~pb, child_fen           # squares newly attacked
    else:
        changed, pos = pb & ~ca, parent_fen           # squares no longer attacked
    b = chess.Board(pos)
    them = not k
    out = []
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        for sq in b.pieces(pt, them):
            if int(b.attacks(sq)) & changed:
                out.append(f"{chess.piece_symbol(pt).upper()}{chess.square_name(sq)}")
    return out


def flank_name(fen, k):
    f = chess.square_file(chess.Board(fen).king(k))
    return "queenside" if f <= 2 else "kingside" if f >= 5 else "centre"


def check_detail(pos, k, key, capturers=False):
    """Name the checking piece(s) as from→to (piece's square → the check square). For an
    UNSAFE check, also name the king's-side pieces that would capture the checker (the
    reason it's unsafe)."""
    b = chess.Board(pos)
    them = not k
    ksq = b.king(k)
    items = []
    for pc, bbv in king_mod.king_term(b, get_context(b), k)[2][key].items():
        csq = next(iter(chess.scan_forward(bbv)), None)  # square the check is delivered from
        if csq is None:
            continue
        frm = next((p for p in b.pieces(SC_PT[pc], them) if csq in b.attacks(p)), None)
        if frm is not None and ksq in b.attacks(frm):    # this piece ALREADY checks the king (the check
            continue                                     # square sits on its live check ray) -> not a new one
        to = chess.square_name(csq)
        d = f"{SC_NAMES[pc]} {chess.square_name(frm)}→{to}" if frm is not None else f"{SC_NAMES[pc]} to {to}"
        items.append(d)                                  # no nested "(met by ...)" — keep it flat
    return "; ".join(items[:2])


def kd_detail2(term, parent_fen, child_fen, k, dcomp):
    """Second-level geometry behind a king-danger component."""
    pos = child_fen if dcomp > 0 else parent_fen
    if term == "safe checks":
        return check_detail(pos, k, "safe_check_sqs")
    if term == "unsafe checks":
        return check_detail(pos, k, "unsafe_check_sqs", capturers=True)
    if term == "attackers":
        before, after = king_ring_attackers(parent_fen, k), king_ring_attackers(child_fen, k)
        changed = list((after - before) if dcomp > 0 else (before - after))
        return ", ".join(f"{pc}{sq}" for pc, sq in changed[:2])
    if term == "king pins":
        b = chess.Board(pos)
        pins = [chess.square_name(s) for s in chess.scan_forward(bb.blockers_for_king(b, k))
                if b.color_at(s) == k]
        return ", ".join(pins[:2])
    if term == "shelter bonus":
        bp, ch = chess.Board(parent_fen), chess.Board(child_fen)
        if bp.king(k) != ch.king(k):
            return f"K{chess.square_name(bp.king(k))}→{chess.square_name(ch.king(k))}"
        moved = ch.pieces(chess.PAWN, k) ^ bp.pieces(chess.PAWN, k)
        sqs = [chess.square_name(s) for s in moved]
        return "pawn " + "/".join(sqs[:2]) if sqs else ""
    if term == "flank attack":
        return ", ".join(flank_change_pieces(parent_fen, child_fen, k, dcomp)[:3])
    return ""


def shelter_ref_sq(board, color):
    """The square whose shelter SF actually uses -- the current king square, or a
    castling target if its (pre-penalty) shelter mg is higher (matches king.py)."""
    ksq = board.king(color)
    ref, best = ksq, king_mod.evaluate_shelter(board, color, ksq).mg
    for has, tgt in ((board.has_kingside_castling_rights(color), 6 if color == chess.WHITE else 62),
                     (board.has_queenside_castling_rights(color), 2 if color == chess.WHITE else 58)):
        if has and king_mod.evaluate_shelter(board, color, tgt).mg > best:
            ref, best = tgt, king_mod.evaluate_shelter(board, color, tgt).mg
    return ref


def shelter_files(ksq):
    kf = min(max(ksq & 7, 1), 6)
    return {kf - 1, kf, kf + 1}


def pawnless_flank_kings(fen):
    b = chess.Board(fen)
    pawns = int(b.pawns)
    return {col for col in (chess.WHITE, chess.BLACK)
            if not (pawns & king_mod.KING_FLANK[b.king(col) & 7])}


def elu_king_danger(parent_fen, child_fen, mover, sign):
    """Name the component that DROVE the king-danger change, on whichever king
    (ours/theirs) whose danger penalty moved most. The component must align with the
    net direction (so an offsetting drop isn't reported as the cause)."""
    best = None  # (abs_dpen, king, term, dcomp)
    for k in (mover, not mover):
        pb, cb = kd_pen_comp(parent_fen, k)
        pa, ca = kd_pen_comp(child_fen, k)
        dkd = sum(ca.values()) - sum(cb.values())            # net king-danger change
        cands = [(t, ca.get(t, 0) - cb.get(t, 0)) for t in set(cb) | set(ca) if t != "tempo"]
        if dkd > 0:
            cands = [c for c in cands if c[1] > 0]            # explain the rise
        elif dkd < 0:
            cands = [c for c in cands if c[1] < 0]            # explain the fall
        cands.sort(key=lambda kv: -abs(kv[1]))
        if not cands:
            continue
        cand = (abs(pa - pb), k, cands[0][0], cands[0][1])
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return ""
    _, k, term, dcomp = best
    tmpl = KD_TMPL.get(term)
    if not tmpl:
        return ""
    if term in ("safe checks", "unsafe checks") and dcomp < 0 and k == (not mover):
        mv = derive_move(parent_fen, child_fen)         # the move IS the check on k -> don't call it
        if mv is not None and chess.Board(parent_fen).gives_check(mv):   # "removes the possibility of a check"
            return ""
    who = "White's" if k == chess.WHITE else "Black's"  # name the king by colour, not our/their
    phrase = f"{tmpl[0] if dcomp > 0 else tmpl[1]} {who} king"
    d2 = kd_detail2(term, parent_fen, child_fen, k, dcomp)
    if term in ("safe checks", "unsafe checks") and dcomp < 0 and not d2:
        return ""                                       # the only checks "removed" were already-active ones
    return f"{phrase} [{d2}]" if d2 else phrase


def elu_flank_attacks(parent_fen, child_fen, mover, sign):
    """Name whose king flank the attack shifted on, and by which pieces."""
    best = None  # (abs_dkfa, k, dkfa)
    for k in (not mover, mover):                         # prefer the offensive framing (opponent's king) on ties
        kb = king_mod.king_term(chess.Board(parent_fen), get_context(chess.Board(parent_fen)), k)[2]["kfa"]
        ka = king_mod.king_term(chess.Board(child_fen), get_context(chess.Board(child_fen)), k)[2]["kfa"]
        dk = ka - kb
        if best is None or abs(dk) > best[0]:
            best = (abs(dk), k, dk)
    if best is None or best[2] == 0:
        return ""
    _, k, dk = best
    pos = child_fen if dk > 0 else parent_fen
    who = "White's" if k == chess.WHITE else "Black's"
    flank = flank_name(pos, k)
    return f"strengthens the attack on {who} {flank}" if dk > 0 else f"weakens the attack on {who} {flank}"


def elu_shelter(parent_fen, child_fen, mover, sign):
    """Explain a king's pawn-shelter change: the king moving, losing the castle
    option (which re-evaluates the castled-square shelter), or a pawn on the three
    shelter files of the king's reference square (own = cover, enemy = storm)."""
    pb, cb = chess.Board(parent_fen), chess.Board(child_fen)
    best_k, best_d = None, 0                            # king whose shelter mg moved most
    for color in (chess.WHITE, chess.BLACK):
        d = king_mod.king_safety_shelter(cb, color).mg - king_mod.king_safety_shelter(pb, color).mg
        if abs(d) > abs(best_d):
            best_k, best_d = color, d
    if best_k is None:
        return ""
    who = "White's" if best_k == chess.WHITE else "Black's"
    kp, kc = pb.king(best_k), cb.king(best_k)
    ref_before = shelter_ref_sq(pb, best_k)            # SF credits the BEST reachable shelter -- possibly a
    if ref_before != kp and best_d < 0:                # castle target, not the king's square; a move can forfeit it
        side = "kingside" if (ref_before & 7) >= 4 else "queenside"
        lead = "" if best_k == mover else f"{who} king "
        return f"{lead}gives up the safer {side} castling shelter"
    if kp != kc:
        return f"{who} king {chess.square_name(kp)}→{chess.square_name(kc)}"
    if (pb.has_kingside_castling_rights(best_k) != cb.has_kingside_castling_rights(best_k)
            or pb.has_queenside_castling_rights(best_k) != cb.has_queenside_castling_rights(best_k)):
        return f"{who} king gives up the castle option"
    files = shelter_files(shelter_ref_sq(cb, best_k))
    tag = lambda col: "cover" if col == best_k else "storm"
    sp = lambda board: {s: board.color_at(s) for s in chess.scan_forward(int(board.pawns))
                        if (s & 7) in files}
    P, C = sp(pb), sp(cb)
    for s in sorted(set(P) & set(C)):                  # a shelter pawn captured, or the king recaptures cover
        if P[s] != C[s]:
            if C[s] == best_k:
                return f"{who} shelter pawn back on {chess.square_name(s)}"
            return f"{who} {chess.square_name(s)} shelter pawn captured"
    for g in sorted(set(P) - set(C)):                  # a shelter/storm pawn advanced or left
        adv = [n for n in set(C) - set(P) if (n & 7) == (g & 7) and C[n] == P[g]]
        if adv:
            return f"{who} {tag(P[g])} {chess.FILE_NAMES[g & 7]}-pawn {chess.square_name(g)}→{chess.square_name(adv[0])}"
        return f"{who} {tag(P[g])} pawn {chess.square_name(g)} gone"
    for n in sorted(set(C) - set(P)):
        return f"{who} {tag(C[n])} pawn to {chess.square_name(n)}"
    return ""


def elu_pawnless_flank(parent_fen, child_fen, mover, sign):
    P, C = pawnless_flank_kings(parent_fen), pawnless_flank_kings(child_fen)
    items = []
    for col in C - P:
        who = "White's" if col == chess.WHITE else "Black's"
        items.append((col != mover, f"{who} king flank now pawnless"))
    for col in P - C:
        who = "White's" if col == chess.WHITE else "Black's"
        items.append((col == mover, f"{who} king flank gains a pawn"))
    want = sign > 0
    items.sort(key=lambda it: it[0] != want)
    return ", ".join(t for _, t in items[:2])


KD_VERB = {
    "more attackers on": "brings more attackers against", "fewer attackers on": "removes an attacker from",
    "more attacked squares around": "attacks more squares around", "fewer attacked squares around": "frees up squares around",
    "prepares a check against": "prepares a check against", "removes a check against": "removes the possibility of a check against",
    "a pin against": "sets up a pin against", "pin relieved on": "relieves a pin on",
    "more attacks around": "steps up the attacks around", "fewer attacks around": "lets up the attacks around",
    "a flank attack on": "launches a flank attack on", "eased flank attack on": "lets up the flank attack on",
    "more enemy activity vs": "gives the enemy more play against", "less enemy activity vs": "cuts the enemy's play against",
    "attacking queen back vs": "brings the attacking queen back against", "queen traded off near": "trades queens off near",
    "lost knight guard of": "weakens the knight's defence of", "knight now guards": "adds a knight to the defence of",
    "weakened shelter of": "weakens the shelter of", "improved shelter of": "improves the shelter of",
    "less flank defense for": "weakens the flank defence for", "more flank defense for": "strengthens the flank defence for",
}

# When the king whose danger ROSE is the mover's OWN king, the rise is a concession
# (our move EXPOSED our king) — the opponent mounts the attack, not us. The default
# offense-framed verbs above then read backwards ("attacks more squares around our
# king"), so these passive variants replace them for the own-king case.


KD_OWN = {
    "more attackers on":            "lets more attackers bear on",
    "more attacked squares around": "lets more squares be attacked around",
    "prepares a check against":      "allows a check against",
    "a pin against":                "allows a pin against",
    "more attacks around":          "lets the attack build around",
    "a flank attack on":            "allows a flank attack on",
    "attacking queen back vs":      "lets the attacking queen return against",
}


def v_king_danger(pf, cf, mover, sign):
    d = re.sub(r"\s+", " ", elu_king_danger(pf, cf, mover, sign).replace("[", "(").replace("]", ")")).strip()
    if not d:
        return ""
    for k in sorted(KD_VERB, key=len, reverse=True):
        if d.startswith(k):
            rest = d[len(k):].strip()
            verb = KD_VERB[k]
            # A danger RISE on the mover's OWN king is a concession (the move exposes
            # its own king); the offense-framed verb then reads backwards, so swap in
            # the passive variant.
            m = re.match(r"(White|Black)'s king", rest)
            if m and (m.group(1) == "White") == (mover == chess.WHITE) and k in KD_OWN:
                verb = KD_OWN[k]
            return f"{verb} {rest}"
    return d





