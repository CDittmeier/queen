"""moves.py — the cross-factor ORDERING + COMBINING layer over the primitives.

Every factor-specific phrasing lives in datagen/tree/primitives/<factor>.py (each Factor
exposes evaluate() and describe(); describe returns {sub_factor: description}). This module
owns only the logic that spans factors:
  * move_priority   — CCT move ordering.
  * elucidate(fen, move) — THE top-level move describer: rank the terms by |delta cp|, ask each
    salient term's Factor for its {sub_factor: description} dict, pick each term's most salient
    clause, and join the top few into one description of the move.
  * effect_map(fen, move) / diff_clause — the same per sub-factor, for the repeated-refuter diff.
tree.py imports the phrasing seam (notation/PERSP) + these from here.
"""
from __future__ import annotations

import chess

from datagen.tree.primitives import common as common
from datagen.tree.primitives.toplevel import eval_full, TERMS
from datagen.tree.primitives import threats as threats_mod     # threat detectors, for CCT move ordering

# phrasing seam re-exported for tree.py (the Notation renderer + root-mover perspective):
notation = common.notation
PERSP = common.PERSP
join_contrast = common.join_contrast
join = common.join
PV = common.PV

BY_TAG = {t.tag: t for t in TERMS}            # term tag -> its Factor


# ============================================================ CCT move ordering
def threat_targets(fen, mover):
    t = set()
    for sq in threats_mod.minor_threats(fen, mover):
        t.add(sq)
    for sq in threats_mod.rook_threats(fen, mover):
        t.add(sq)
    for sq in threats_mod.safe_pawn_threats(fen, mover):
        t.add(sq)
    for v in threats_mod.pawn_push_threats(fen, mover):
        t.add(v)
    for sq in threats_mod.hanging_pieces(fen, mover):
        t.add(sq)
    return t


def move_priority(board, move):
    """Sortable key: (tier, tier-local tiebreaks, uci). tier 0=check 1=capture
    2=threat 3=quiet. Lower sorts first. Fully deterministic (uci final key)."""
    child = board.copy(stack=False)
    child.push(move)
    uci = move.uci()
    is_cap = board.is_capture(move)
    if board.gives_check(move):
        cap_v = PV[board.piece_at(move.to_square).piece_type] if (is_cap and board.piece_at(move.to_square)) else 0
        return (0, (0 if len(child.checkers()) >= 2 else 1, 0 if is_cap else 1, -cap_v), uci)
    if is_cap:
        vic = board.piece_at(move.to_square)
        atk = board.piece_at(move.from_square)
        return (1, (-(PV[vic.piece_type] if vic else 1), PV[atk.piece_type] if atk else 9), uci)
    cf = child.fen()
    new_t = threat_targets(cf, board.turn) - threat_targets(board.fen(), board.turn)
    if new_t:
        tv = max((PV[child.piece_type_at(s)] for s in new_t if child.piece_type_at(s)), default=0)
        return (2, (-tv,), uci)
    pc = board.piece_at(move.from_square)
    cat = 4
    if board.is_castling(move):
        cat = 0
    elif pc and pc.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(move.from_square) in (0, 7):
        cat = 1
    elif pc and pc.piece_type == chess.PAWN and (move.to_square & 7) in (2, 3, 4, 5):
        cat = 2
    elif pc and pc.piece_type == chess.ROOK and not (int(child.pawns & child.occupied_co[pc.color]) & chess.BB_FILES[move.to_square & 7]):
        cat = 3
    return (3, (cat,), uci)


# ============================================================ combining primitive effects
def setup(fen, move):
    if isinstance(move, str):
        move = chess.Move.from_uci(move)
    board = chess.Board(fen)
    mover = "w" if board.turn == chess.WHITE else "b"
    us = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
    sign = 1 if mover == "w" else -1
    common.CAPSQ[0] = common.captured_sq(fen, move)
    common.MOVED_FROM[0] = move.from_square
    common.PREV_CAPSQ[0] = None            # default: no recapture context (elucidate sets it if known)
    board.push(move)
    after = board.fen()
    return move, after, mover, us, sign, eval_full(fen), eval_full(after)


def term_clause(term, d, elu, B, A, sign):
    """(clause, good) — the term's most salient clause from its {sub_factor: description} dict.
    Material/whole factors have a single clause; container factors pick the top sub-factor by
    |delta| that has a description."""
    if term.KIND == "material":
        return elu.get("material"), d > 0
    if term.KIND == "whole":
        return elu.get(term.tag), d > 0
    before_sub, after_sub = B[term.tag][1], A[term.tag][1]
    for sub in sorted(set(before_sub) | set(after_sub),
                      key=lambda s: -abs(sign * (after_sub.get(s, 0.0) - before_sub.get(s, 0.0)))):
        x = sign * (after_sub.get(sub, 0.0) - before_sub.get(sub, 0.0))
        if abs(x) < 6 or sub in common.GENERIC_SUB:
            continue
        if elu.get(sub):
            return elu[sub], x > 0
    return None, None


def elucidate(fen, move, topn=3, prev_capture_sq=None, sole_recapture=False):
    """Describe how `move` changes `fen`: rank terms by |delta cp|, collect each salient term's
    Factor.elucidate() dict, pick its top clause, and join the top `topn`.

    Tree context (from tree.py): `prev_capture_sq` is the square the previous move captured on
    (so a capture back on it phrases as "recaptures ..."); `sole_recapture` is True when this
    move's node has exactly one reply and it recaptures the moved piece (so the recapture is
    already shown next and needn't be flagged here)."""
    move, after, mover, us, sign, B, A = setup(fen, move)
    common.PREV_CAPSQ[0] = prev_capture_sq       # so material.vmaterial can phrase a recapture
    ranked = sorted(((t, sign * (A[t.tag][0] - B[t.tag][0])) for t in TERMS), key=lambda kv: -abs(kv[1]))
    ranked = [(t, d) for t, d in ranked if abs(d) >= 8] or ranked[:1]

    # A capture whose own piece the opponent can now take back is a *recapture* — a
    # conditional exchange, not an unconditional loss. The bare threat-on-the-moved-piece
    # clause is replaced: if the tree already shows the recapture as the sole reply, it's
    # obvious and dropped (`sole_recapture`); otherwise it becomes a "open to being
    # recaptured" concession, with the capture spoken first.
    board, ab = chess.Board(fen), chess.Board(after)
    recapturable = (board.is_capture(move) and not move.promotion
                    and ab.piece_at(move.to_square) is not None
                    and ab.is_attacked_by(not us, move.to_square))
    if recapturable:
        atk_val = min(PV[ab.piece_at(s).piece_type] for s in ab.attackers(not us, move.to_square))
        if any(PV[ab.piece_at(d).piece_type] < atk_val for d in ab.attackers(us, move.to_square)):
            recapturable = False           # a cheaper defender makes the "recapture" a bad trade — safe

    out = []
    for term, d in ranked:
        clause, good = term_clause(term, d, term.describe(fen, after, move, mover, B, A), B, A, sign)
        if clause:
            clause = notation[0].tok_free(common.fix_ownopp(clause, us), after, fen, default_color=us)
            if clause not in [row[0] for row in out]:
                out.append([clause, good, term.tag])
        if len(out) >= topn:
            break

    if recapturable:
        idx = next((i for i, r in enumerate(out) if r[2] == "threats" and not r[1]), None)
        if sole_recapture:                              # the recapture is the very next move -> obvious
            if idx is not None:
                out.pop(idx)
        else:
            pc = ab.piece_at(move.to_square)
            atk = min(ab.attackers(not us, move.to_square),   # the cheapest piece that can take it back
                      key=lambda s: PV[ab.piece_at(s).piece_type])
            ap = ab.piece_at(atk)
            rc = (f"leaves the {common.piece_name(pc.piece_type, pc.color)} on "
                  f"{common.square_name(move.to_square)} open to being recaptured by the "
                  f"{common.piece_name(ap.piece_type, ap.color)} on {common.square_name(atk)}")
            rc = notation[0].tok_free(common.fix_ownopp(rc, us), after, fen, default_color=us)
            if idx is not None:
                out[idx] = [rc, False, "recapture"]
            else:
                out.append([rc, False, "recapture"])
            goods = [c for c, g, _ in out if g]         # capture first, recapture as the concession
            bads = [c for c, g, _ in out if not g]
            head, tail = join(goods), join(bads)
            return f"{head}, but {tail}" if head and tail else (head or tail)

    return join_contrast([(c, g) for c, g, _ in out])


def effect_map(fen, move):
    """{factor: (clause, good)} over ALL salient sub-factors (for the repeated-refuter diff)."""
    move, after, mover, us, sign, B, A = setup(fen, move)
    out = {}
    for term in TERMS:
        d = sign * (A[term.tag][0] - B[term.tag][0])
        if abs(d) < 8:
            continue
        elu = term.describe(fen, after, move, mover, B, A)
        if term.KIND == "material":
            out["material"] = (elu.get("material") or "", d > 0)
        elif term.KIND == "whole":
            out[term.tag] = (elu.get(term.tag) or "", d > 0)
        else:
            before_sub, after_sub = B[term.tag][1], A[term.tag][1]
            for sub in set(before_sub) | set(after_sub):
                x = sign * (after_sub.get(sub, 0.0) - before_sub.get(sub, 0.0))
                if abs(x) < 6 or sub in common.GENERIC_SUB:
                    continue
                if elu.get(sub):
                    out[sub] = (elu[sub], x > 0)
    return {k: (notation[0].tok_free(common.fix_ownopp(c, us), after, fen, default_color=us), g)
            for k, (c, g) in out.items()}


def diff_clause(cur_factors, first_factors):
    """Describe how a repeated refuter's effects differ from its first occurrence."""
    added = [c for f, (c, g) in cur_factors.items() if f not in first_factors]
    removed = [c for f, (c, g) in first_factors.items() if f not in cur_factors]
    seg = []
    if removed:
        seg.append("no longer " + join(removed))
    if added:
        seg.append(("but now " if removed else "now ") + join(added))
    return (", which " + " ".join(seg)) if seg else ""
