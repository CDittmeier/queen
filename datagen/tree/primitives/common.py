"""primitives/common.py — the shared verbalization layer.

Low-level helpers every factor's verbalize() leans on: mode-aware naming (via the
injected Notation renderer notation), deterministic template choice (pick), list joining
and gain/concession contrast, and the sub-factor dispatch machinery. Each factor
module registers its polished (v_*) and raw (elu_*) sub-factor verbalizers into the
V / ELU / VTOP / ELU_TOP registries at import; top_subfactor_clause / subfactor_effects
then rank a term's sub-factors and dispatch to the owning factor's phraser.

PERSP[0] is the ROOT mover (so "our"/"the opponent's" stay consistent); CAPSQ[0] is
the square the current move captured on (set per clause).
"""
from __future__ import annotations

import hashlib
import re

import chess

PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
          chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

notation = [None]   # Notation renderer, set per narrative at the top of flatten()


def piece_name(pt, color=None):
    return notation[0].piece(pt, color)


def square_name(s):
    return notation[0].sq(s)


def file_letter(s):
    return chess.FILE_NAMES[s & 7]


def who(color):
    return "White" if color == chess.WHITE else "Black"


PERSP = [chess.WHITE]  # the ROOT mover; set per narrative so "our"/"the opponent's" stay consistent
CAPSQ = [None]  # square the current move captured on (set per clause); so a threat we EXECUTED
                 # (captured the threatened piece) isn't also reported as "threat lifted".
PREV_CAPSQ = [None]  # square the PREVIOUS move captured on (tree context); when the current move
                     # captures the same square, it's a recapture ("recaptures the <piece> on <sq>").
MOVED_FROM = [None]  # square the current move's piece vacated; a threat "lifted" on a piece that
                     # simply moved away is not worth stating (it obviously left).


def our(color):        # possessive for a piece, relative to the root mover (not the move's mover)
    return "our" if color == PERSP[0] else "the opponent's"


def obj(color):        # a side as subject/object ("us" / "the opponent")
    return "us" if color == PERSP[0] else "the opponent"


def fix_ownopp(text, mover):   # convert any mover-relative own/opp tokens to root-relative
    text = re.sub(r"\bown\b", our(mover), text)
    text = re.sub(r"\bopp\b", our(not mover), text)
    return text


def pick(key, opts):
    """Deterministic template choice (stable across runs; varies by input)."""
    return opts[hashlib.md5(str(key).encode()).digest()[0] % len(opts)]


def join(items):
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def join_contrast(pairs):
    """Join (clause, good_for_mover) effects, contrasting gains against concessions."""
    goods = [c for c, g in pairs if g]
    bads = [c for c, g in pairs if not g]
    if not goods:
        return join(bads)
    if not bads:
        return join(goods)
    g, b = join(goods), join(bads)
    return pick(g + b, [f"{g} but {b}", f"{g}, though it {b}", f"{g}, conceding that it {b}",
                         f"{b}, but in return {g}"])


# ============================================================ move ordering (CCT)


def sign_pick(parts, sign):
    want = sign > 0
    parts = [t for g, t in sorted(parts, key=lambda it: it[0] != want)]
    return parts[0] if parts else ""


def captured_sq(pf, move):
    """Square the move captures a piece on (the en-passant victim's square for ep),
    or None for a non-capture — used to suppress a 'threat lifted' clause on a piece
    we just took (we executed the threat, we didn't relieve it)."""
    b = chess.Board(pf)
    if b.is_en_passant(move):
        return move.to_square + (-8 if b.turn == chess.WHITE else 8)
    return move.to_square if b.is_capture(move) else None


READABLE = {
    "shelter/storm": "king's pawn shelter", "flank attacks": "flank attack", "pawnless flank": "king's flank",
    "king protector": "king's piece cover", "bishop pawns": "bishop's pawn structure", "minor behind pawn": "minor's pawn cover",
    "outpost": "outpost", "reachable outpost": "knight's outpost", "trapped rook": "rook's mobility",
    "rook on open/semi-open file": "rook's file", "rook on queen file": "rook and queen", "weak queen": "queen's safety",
    "long diagonal": "bishop's long diagonal", "weak lever": "pawn levers", "initiative": "initiative",
    "threat by king": "king's threat",
}
VERB_STARTS = {"attacks", "takes", "lines", "loses", "prepares", "removes", "builds",
                "controls", "gives", "connects", "gains", "lets", "strengthens", "weakens"}


# ---------------------------------------------------------------- sub-factor helpers
GENERIC_SUB = {"piece", "side", "king", "pawn", "passed pawn"}   # non-descriptive carriers


def mech_fallback(tag, elu, before, after, us, sign):
    """Phrase a sub-factor that has no polished phraser: wrap its raw elucidator's text with a
    directional verb. `elu` is the factor's own raw elucidator for `tag` (or None)."""
    d = elu(before, after, us, sign) if elu else ""
    if not d:
        return ""
    d = re.sub(r"\s+", " ", d.replace("[", "(").replace("]", ")")).strip()
    if d.split()[0].lower() in VERB_STARTS:               # already a verb phrase
        return d
    verb = pick(d, ["improves", "strengthens"]) if sign > 0 \
        else pick(d, ["weakens", "worsens", "compromises"])   # directional, not the vague "reshapes"
    return f"{verb} the {READABLE.get(tag, tag)} ({d})"


# ---------------------------------------------------------------- shared geometry utilities
# (not owned by a single factor — used across mobility/king/threats/pieces/...)
def derive_move(parent_fen, child_fen):
    pb = chess.Board(parent_fen)
    tgt = chess.Board(child_fen).board_fen()
    for m in pb.legal_moves:
        t = pb.copy(stack=False)
        t.push(m)
        if t.board_fen() == tgt:
            return m
    return None


def pcsym(pt):
    return chess.piece_symbol(pt).upper()
