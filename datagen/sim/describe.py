"""Verdict sentence builders: '{side} is {verdict} due to {prose}' in paired
human/machine (POV-token) form, for every verdict class.

The positional prose rides the activity detectors
(datagen.tree.primitives.activity); balance sentences follow the deployed
convention: feature names only, no squares, and the dry sentence says
"drawish" (the tree pipeline always applied that edit post-hoc, so the
training data now carries it directly).
"""
import chess

from datagen.tree.primitives import activity
from utils.translate_helpers import Translator

# verdict phrase from the White-POV eval (pawns), per the Stockfish convention
_VERDICTS = [(4.0, "winning"), (2.5, "much better-to-winning"),
             (1.5, "much better"), (1.0, "better-to-much better"),
             (0.6, "better"), (0.3, "slightly better-to-better"),
             (0.1, "slightly better")]


def verdict_head(d: Translator, eval_w: float, mate: bool = False) -> Translator:
    color = eval_w >= 0
    phrase = "winning" if mate else next(
        (p for t, p in _VERDICTS if abs(eval_w) >= t), None)
    assert phrase is not None
    return d.side(color).txt(f" is {phrase} due to ")


def _side_of(v: float) -> chess.Color:
    return chess.WHITE if v > 0 else chess.BLACK


def feature_prose(name: str, board, ctx, felt_v: float):
    """Prose fragment (Translator) for a dominant positional feature (favoured side
    per felt_v's sign), or None when the feature fails to render."""
    d = Translator(board.turn)
    col = _side_of(felt_v)

    if name == "pawn structure":
        agg = activity.pawn_structure_sides(board, ctx)
        good = [t for (w, t), v in agg.items()
                if w == col and v * (1 if col else -1) > 0.04]
        bad = [t for (w, t), v in agg.items()
               if w != col and v * (1 if col else -1) > 0.04]
        d.txt("the pawn structure: ").side(col).txt("'s ")
        d.txt(" and ".join(good or ["healthier"]) + " pawns")
        if bad:
            d.txt(" against ").side(not col).txt("'s ")
            d.txt(" and ".join(bad) + " pawns")
        return d

    if name == "knight outpost":
        sqs = [sq for color, sq, _s in activity.outpost_knights(board, ctx)
               if color == col]
        if not sqs:
            return None
        d.txt("the knight outpost" + ("s" if len(sqs) > 1 else "") + " on ")
        for i, s in enumerate(sqs):
            if i:
                d.txt(" and ")
            d.sq(s)
        return d

    if name == "active bishop pair":
        recs = activity.bishop_pair(ctx, col)
        if not recs:
            return None
        d.txt("the active bishop pair on ")
        return d.sq(recs[0].sq).txt(" and ").sq(recs[1].sq)

    if name == "active lone bishop":
        recs = list(ctx.piece_recs(col, chess.BISHOP))
        if len(recs) != 1:
            return None
        return d.txt("the active bishop on ").sq(recs[0].sq)

    if name == "bad bishop":
        bad_bs = activity.bad_bishops(board, ctx, not col)
        if not bad_bs:
            return None
        worst, k, _pen = max(bad_bs, key=lambda t: t[1])
        if k == 0:
            return None
        d.side(not col).txt("'s bad bishop on ").sq(worst)
        return d.txt(", hemmed in by pawns of its own color")

    if name == "bad knight":
        for sq, _mob in activity.bad_knights(board, ctx, not col):
            d.side(not col).txt("'s bad knight on ").sq(sq)
            return d.txt(", which has almost no squares")
        return None

    if name == "bad rook":
        for sq, _mob in activity.bad_rooks(board, ctx, not col):
            d.side(not col).txt("'s bad rook on ").sq(sq)
            return d.txt(", shut in with almost no moves")
        return None

    if name == "active rook":
        rs = []
        for r in ctx.piece_recs(col, chess.ROOK):
            fb = activity.rook_file_bonus(board, col, r.sq)
            rs.append((r.sq, fb is not None))
        if not rs:
            return None
        d.txt("the active rook" + ("s" if len(rs) > 1 else "") + " on ")
        for i, (s, _) in enumerate(rs):
            if i:
                d.txt(" and ")
            d.sq(s)
        if any(open_ for _, open_ in rs):
            d.txt(", commanding an open file")
        return d

    if name == "colour complexion":
        light = activity.weak_colour(board, ctx, not col)
        if light is None:
            return None
        d.side(not col).txt("'s weakness on the "
                            + ("light" if light else "dark") + " squares")
        return d

    if name == "space":
        sgn = 1 if col else -1
        phrase, _ = max(activity.SPACE_REGIONS, key=lambda reg: sgn * (
            activity.space_region_count(board, ctx, chess.WHITE, reg[1])
            - activity.space_region_count(board, ctx, chess.BLACK, reg[1])))
        return d.side(col).txt(f"'s space advantage {phrase}")

    if name == "passed pawns":
        sqs = activity.passed_squares(board, ctx, col)
        if not sqs:
            return None
        d.txt("the passed pawn" + ("s" if len(sqs) > 1 else "") + " on ")
        for i, s in enumerate(sqs):
            if i:
                d.txt(" and ")
            d.sq(s)
        return d

    return None


def material_prose(d: Translator, mat_w: int) -> Translator:
    n = abs(mat_w)
    if n >= 9:
        return d.txt("an overwhelming material advantage")
    return d.txt(f"a material advantage of roughly {n} "
                 + ("pawn" if n == 1 else "pawns"))


def fork_prose(board, move, targets) -> Translator:
    d = Translator(board.turn)
    b2 = board.copy(stack=False)
    b2.push(move)
    d.txt("the fork ").move(board, move).txt(", the ")
    d.piece(b2, move.to_square).txt(" attacking the ")
    for i, t in enumerate(targets):
        if i:
            d.txt(" and the ")
        d.piece(b2, t)
    return d


def balance_sentence(kind: str, pov: chess.Color, w_feats=None, b_feats=None):
    """kind: 'dynamic' | 'dry'; feats = feature-name lists favouring each side."""
    d = Translator(pov)
    if kind == "dry":
        return d.txt("The position is drawish: no notable feature favours "
                     "either side")
    d.txt("The position is dynamically balanced: the ")
    d.txt(", ".join(w_feats)).txt(" favouring ").side(chess.WHITE)
    d.txt(" is offset by the ").txt(", ".join(b_feats))
    d.txt(" favouring ").side(chess.BLACK)
    return d
