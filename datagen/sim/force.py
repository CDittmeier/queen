"""Guarantee the key variation is one of the plans.

When Stockfish says one move is clearly best -- its win% leads the second move
by GAP or more -- that move is the point of the position, and a plan list that
omits it cannot lead anywhere useful. The plan model surfaces it most of the
time but not always, so this closes the gap deterministically:

  promote  the model already opens some plan with that move -> that plan is
           moved to the front of the list, unchanged;
  inject   nothing opens with it -> a plan IS that move, extended in (opponent
           reply, our reply) pairs for as long as our reply is itself clearly
           best, so the whole forcing sequence is spelled out. To keep the list
           at `cap` the weakest existing plan is dropped, and the order is then
           shuffled so the injected plan is not identifiable by position.

Injected plans are built as ordinary plan entries -- the human prose is
written in the model's own compressed vocabulary and re-parsed with the same
move_seq/_own_tuples the model's plans go through, and a paired token-space
line (anchored, like the model's own generations, at the node's side to move)
rides along for the token-mode narrative -- so the search, the subset-drop,
the live-plan tracking and the flattener all treat them identically.

Forcing changes what alpha-beta expands, so it is applied at the root and then
follows the forced line: if the root had to be forced, the child down the
forced move is forced too, and so on for as long as forcing keeps firing.
"""
import chess

from datagen.sim.plan_parse import _own_tuples, move_seq
from datagen.tree.search import win_rate
from utils.utils import POV_SQUARE_TOKENS, _PIECE_TO_POV_TOKEN

GAP = 5.0            # win% the best move must lead the second by
MAX_PAIRS = 4        # (opponent reply, our reply) pairs appended to an injection

_NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def key_move(fen, judge):
    """(uci, win% gap) when one move is clearly best at `fen`, else None."""
    best, cp1, _pv, cp2 = judge.search2(fen)
    if best is None:
        return None
    if cp2 is None:                       # only one legal move: trivially forced
        return best, 100.0
    gap = win_rate(cp1) - win_rate(cp2)
    return (best, gap) if gap >= GAP else None


def forced_line(fen, judge, max_pairs=MAX_PAIRS):
    """[our best, opp reply, our best, ...] while every own move is clearly best."""
    b = chess.Board(fen)
    got = key_move(fen, judge)
    if got is None:
        return []
    line = [got[0]]
    b.push(chess.Move.from_uci(got[0]))
    for _ in range(max_pairs):
        if b.is_game_over():
            break
        reply, _, _ = judge.search(b.fen())          # opponent: plain best
        if reply is None:
            break
        line.append(reply)
        b.push(chess.Move.from_uci(reply))
        if b.is_game_over():
            break
        got = key_move(b.fen(), judge)               # ours: must be clearly best
        if got is None:
            break
        line.append(got[0])
        b.push(chess.Move.from_uci(got[0]))
    # end on OUR move: a trailing opponent reply would leave an "if ..." clause
    # with nothing after it
    return line[:len(line) - (1 - len(line) % 2)]


# ------------------------------------------------------------------ prose

def _sqtok(sq, pov_white):
    return POV_SQUARE_TOKENS[sq if pov_white else sq ^ 56]


def _ptok(color, ptype, pov_white):
    return _PIECE_TO_POV_TOKEN[(color == pov_white, ptype)]


def phrase_pair(board, mv, pov_white, third=False):
    """(human, machine) for one forced-line move; machine in POV tokens
    anchored at `pov_white` (the node's side to move)."""
    pc = board.piece_at(mv.from_square)
    frm, to = chess.square_name(mv.from_square), chess.square_name(mv.to_square)
    frm_t, to_t = _sqtok(mv.from_square, pov_white), _sqtok(mv.to_square, pov_white)
    if board.is_castling(mv):
        side = "kingside" if chess.square_file(mv.to_square) > 4 else "queenside"
        h = f"castles {side}" if third else f"castle {side}"
        return h, h
    pt = _ptok(pc.color, pc.piece_type, pov_white)
    if board.is_capture(mv):
        vic = board.piece_at(mv.to_square)
        vtype = vic.piece_type if vic else chess.PAWN         # en passant
        vcolor = vic.color if vic else not board.turn
        vname = _NAME[vtype]
        vcol = "white" if vcolor else "black"
        v = "takes" if third else "take"
        h = f"{v} {vcol} {vname} {to} with the {_NAME[pc.piece_type]} {frm}"
        m = (f"{v} {_ptok(vcolor, vtype, pov_white)}{to_t} "
             f"with the {pt}{frm_t}")
        if mv.promotion:
            h += f", promoting to {_NAME[mv.promotion]}"
            m += f", promoting to {_ptok(pc.color, mv.promotion, pov_white)}"
        return h, m
    if pc.piece_type == chess.PAWN:
        v = "pushes" if third else "push"
        return (f"{v} the pawn {frm} to {to}", f"{v} the {pt}{frm_t} to {to_t}")
    v = "moves" if third else "move"
    h = f"{v} the {_NAME[pc.piece_type]} {frm} to {to}"
    m = f"{v} the {pt}{frm_t} to {to_t}"
    if mv.promotion:
        h += f", promoting to {_NAME[mv.promotion]}"
        m += f", promoting to {_ptok(pc.color, mv.promotion, pov_white)}"
    return h, m


def line_prose(fen, ucis):
    """(human, machine) '- White can take ...; if Black moves ..., White can
    ...' for the line; machine uses <PLAYER>/<OPPONENT> + POV tokens anchored
    at `fen`'s side to move (the node anchor, like the model's own plans)."""
    b = chess.Board(fen)
    pov_white = b.turn
    mover = "White" if b.turn else "Black"
    other = "Black" if b.turn else "White"
    hparts, mparts = [], []
    for i, u in enumerate(ucis):
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        ph, pm = phrase_pair(b, mv, pov_white, third=i % 2 == 1)
        if i % 2 == 0:
            hparts.append(f"{mover} can {ph}")
            mparts.append(f"<PLAYER> can {pm}")
        else:
            hparts.append(f"if {other} {ph}")
            mparts.append(f"if <OPPONENT> {pm}")
        b.push(mv)
    if not hparts:
        return None, None
    # "A; if B, C; if D, E" -- the same shape the model uses for forcing lines
    def join(parts):
        out = parts[0]
        for i in range(1, len(parts), 2):
            out += "; " + parts[i]
            if i + 1 < len(parts):
                out += ", " + parts[i + 1]
        return "- " + out + "."
    return join(hparts), join(mparts)


def entry(fen, ucis):
    """An injected plan, shaped exactly like a plan_entries() entry, plus
    `machine_lines` (token-space) and `ucis` for the narrative."""
    text, machine = line_prose(fen, ucis)
    if text is None:
        return None
    board = chess.Board(fen)
    mover = "White" if board.turn else "Black"
    flat = move_seq(text, mover)
    own = [(v, p, (s[1] if len(s) > 1 else None) if v == "take"
            else (s[0] if len(s) > 1 else None),
            (s[0] if s else None) if v == "take" else (s[-1] if s else None))
           for a, v, p, s in flat if a == mover and (s or v == "castle")]
    return {"side": mover, "first": chess.Move.from_uci(ucis[0]),
            "lines": [text], "machine_lines": [machine], "ucis": list(ucis),
            "seqs": [[list(map(list, own)), 0]], "flat": flat,
            "injected": True}


# ------------------------------------------------------------------ apply

def apply(fen, entries, judge, rng, cap=4):
    """(entries, forced_line) with the key variation guaranteed to be present.

    `forced_line` is empty when nothing was clearly best, in which case the
    entries come back untouched."""
    got = key_move(fen, judge)
    if got is None:
        return entries, []
    best = got[0]
    hit = [e for e in entries if e["first"] is not None
           and e["first"].uci() == best]
    if hit:                       # promote: the model already had it
        rest = [e for e in entries if e not in hit]
        return hit + rest, [best]
    line = forced_line(fen, judge)
    made = entry(fen, line) if line else None
    if made is None:
        return entries, []
    kept = entries[:cap - 1] if len(entries) >= cap else entries
    out = kept + [made]
    rng.shuffle(out)              # do not let position give the injection away
    return out, line
