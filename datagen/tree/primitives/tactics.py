"""Tactical-motif detectors + prose for the search-tree narrative.

Vendored ports of the lichess-puzzler cook.py patterns (via lab/two_networks'
extractor mirrors) — fork, hanging piece, trapped piece, skewer, interference,
deflection, attraction, intermezzo, pin, discovered check/attack. Nothing is
imported from lab/ or lichess-puzzler; the ~60 lines of tagger/util.py helpers
the patterns lean on are inlined below.

Frame: a node position plus the judge's clear-best line from it. `detect()` gets
[setup (the move INTO the node, None at the root), best, reply, ...] and runs the
patterns with pov = the side to move at the node — the same shape as a lichess
puzzle (opponent's setup move, then our line), with pov moves at odd indices of
the mainline list. A fired motif therefore always explains the clear-best move,
possibly referring to the continuation behind it.

Detection returns mode-agnostic clause specs (plain tuples, picklable); render()
turns them into a verb phrase for the "It {eff}." / ", which {eff}." glue frames
under the notation seam (human words / chess-LM tokens) at flatten time.
"""
from __future__ import annotations

from itertools import combinations

import chess

from datagen.tree.primitives import common

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
KVAL = {**VAL, chess.KING: 99}
RAY = (chess.QUEEN, chess.ROOK, chess.BISHOP)


# ============================================================ tagger/util.py helpers (inlined)

def _is_defended(board, piece, square):
    if board.attackers(piece.color, square):
        return True
    for attacker in board.attackers(not piece.color, square):    # ray defense
        ap = board.piece_at(attacker)
        if ap.piece_type in RAY:
            bc = board.copy(stack=False)
            bc.remove_piece_at(attacker)
            if bc.attackers(piece.color, square):
                return True
    return False


def _is_hanging(board, piece, square):
    return not _is_defended(board, piece, square)


def _lower_taker(board, piece, square):
    for attacker_square in board.attackers(not piece.color, square):
        a = board.piece_at(attacker_square)
        if a.piece_type != chess.KING and VAL[a.piece_type] < VAL[piece.piece_type]:
            return True
    return False


def _is_in_bad_spot(board, square):
    piece = board.piece_at(square)
    return (bool(board.attackers(not piece.color, square))
            and (_is_hanging(board, piece, square) or _lower_taker(board, piece, square)))


def _is_trapped(board, square):
    """cook's is_trapped; mutates `board` (early return leaves it pushed) — pass a copy."""
    if board.is_check() or board.is_pinned(board.turn, square):
        return False
    piece = board.piece_at(square)
    if piece.piece_type in (chess.PAWN, chess.KING):
        return False
    if not _is_in_bad_spot(board, square):
        return False
    for escape in board.legal_moves:
        if escape.from_square == square:
            capturing = board.piece_at(escape.to_square)
            if capturing and VAL[capturing.piece_type] >= VAL[piece.piece_type]:
                return False
            board.push(escape)
            if not _is_in_bad_spot(board, escape.to_square):
                return False
            board.pop()
    return True


def _material_count(board, side):
    return sum(len(board.pieces(pt, side)) * v for pt, v in VAL.items())


def _material_diff(board, side):
    return _material_count(board, side) - _material_count(board, not side)


# ============================================================ the line frame

class _Ply:
    """One played move; mimics the chess.pgn.ChildNode surface the patterns use.
    board() is the position AFTER the move (a stored board — copy before mutating)."""
    __slots__ = ("move", "parent", "after", "next")

    def __init__(self, move, parent, after):
        self.move, self.parent, self.after, self.next = move, parent, after, None

    def board(self):
        return self.after

    def turn(self):
        return self.after.turn


def _moved_pt(node):
    pt = node.board().piece_type_at(node.move.to_square)
    assert pt
    return pt


def _frame(board, setup_move, setup_parent_board, line_moves):
    """[setup ply, m0 (pov), m1 (opp), ...] — setup.move is None at the root."""
    setup = _Ply(setup_move, None, board.copy(stack=False))
    if setup_move is not None and setup_parent_board is not None:
        setup.parent = _Ply(None, None, setup_parent_board)
    main, prev, b = [setup], setup, board
    for mv in line_moves:
        if mv not in b.legal_moves:
            break
        b = b.copy(stack=False)
        b.push(mv)
        ply = _Ply(mv, prev, b)
        prev.next = ply
        main.append(ply)
        prev = ply
    return main


# ============================================================ clause specs

class C:
    """Clause spec builder: mode-agnostic parts, rendered later under the notation
    seam. Templates own the articles ("the ") — .p() renders "{piece} on {square}"."""

    def __init__(self):
        self.parts = []

    def t(self, s):
        self.parts.append(("t", s)); return self

    def s(self, sq):
        self.parts.append(("s", sq)); return self

    def p(self, board, sq):
        pc = board.piece_at(sq)
        self.parts.append(("p", pc.piece_type, pc.color, sq)); return self

    def m(self, board, mv):
        self.parts.append(("m", board.fen(), mv.uci())); return self


def render(specs) -> str:
    """[(tag, parts), ...] -> one verb-phrase clause for the "It {eff}." frames."""
    NT = common.notation[0]
    out = []
    for _, parts in specs:
        bits = []
        for p in parts:
            if p[0] == "t":
                bits.append(p[1])
            elif p[0] == "s":
                bits.append(NT.sq(p[1]))
            elif p[0] == "p":
                bits.append(f"{NT.piece(p[1], p[2])} on {NT.sq(p[3])}")
            elif p[0] == "m":
                bits.append(NT.move_prose(chess.Board(p[1]), chess.Move.from_uci(p[2])))
        out.append("".join(bits))
    return " and ".join(out)


# ============================================================ move-level patterns

def _clustered(a, t1, t2):
    fs = [chess.square_file(s) for s in (a, t1, t2)]
    rs = [chess.square_rank(s) for s in (a, t1, t2)]
    return max(fs) - min(fs) <= 1 and max(rs) - min(rs) <= 1


def _counter_captures(pc_type, sq, p_type, t):
    """The target could simply capture the forker: same-type N/B/R/Q pairs (their
    attacks are mutual), a bishop the queen attacks along a diagonal, a rook the
    queen attacks along a rank or file."""
    if p_type == pc_type and p_type in (chess.KNIGHT, chess.BISHOP,
                                        chess.ROOK, chess.QUEEN):
        return True
    if pc_type == chess.QUEEN:
        df = abs((sq & 7) - (t & 7))
        dr = abs((sq >> 3) - (t >> 3))
        if p_type == chess.BISHOP and df == dr:
            return True
        if p_type == chess.ROOK and (df == 0 or dr == 0):
            return True
    return False


def fork_targets(board, move, raw=False):
    """Qualifying fork targets after `move` (>= 2, else []): enemy non-pawns worth
    more than the moved piece, or no more but undefended; a target that could
    itself capture the forker never qualifies. Unless raw, a fork whose attacker
    and every target pair sit inside one 2x2 square doesn't count."""
    us = board.turn
    b = board.copy(stack=False)
    b.push(move)
    sq = move.to_square
    pc = b.piece_at(sq)
    if pc is None:
        return []
    av = KVAL[pc.piece_type]
    targets = []
    for t in b.attacks(sq):
        p = b.piece_at(t)
        if p is None or p.color == us or p.piece_type == chess.PAWN:
            continue
        if _counter_captures(pc.piece_type, sq, p.piece_type, t):
            continue
        if KVAL[p.piece_type] > av or not b.attackers(p.color, t):
            targets.append(t)
    if len(targets) < 2:
        return []
    if not raw and not any(not _clustered(sq, t1, t2)
                           for t1, t2 in combinations(targets, 2)):
        return []
    return targets


def _fork(main, pov):
    node = main[1]
    targets = fork_targets(main[0].board(), node.move)
    if not targets:
        return None
    # confirmation from the judge line: the follow-up move must cash a target
    # (else the "fork" is e.g. a recaptured queen trade that happened to double-attack)
    if len(main) < 4 or main[3].move.to_square not in targets:
        return None
    b2 = node.board()
    c = C().t("creates a fork, the ").p(b2, node.move.to_square).t(" attacking the ")
    for i, t in enumerate(targets):
        if i:
            c.t(" and the ")
        c.p(b2, t)
    return c


def _hanging_piece(main, pov):
    board0 = main[0].board()
    to = main[1].move.to_square
    captured = board0.piece_at(to)
    if not captured or captured.piece_type == chess.PAWN:
        return None
    if not _is_hanging(board0, captured, to):
        return None
    op_move = main[0].move
    if op_move is not None and main[0].parent is not None:
        op_capture = main[0].parent.board().piece_at(op_move.to_square)
        if (op_capture and VAL[op_capture.piece_type] >= VAL[captured.piece_type]
                and op_move.to_square == to):
            return None                       # an even recapture, not a hanging win
    if len(main) >= 4 and _material_diff(main[3].board(), pov) < _material_diff(main[1].board(), pov):
        return None
    return C().t("wins the hanging ").p(board0, to)


# ============================================================ line-level patterns

def _trapped_piece(main, pov):
    for node in main[1::2][1:]:
        square = node.move.to_square
        prev = node.parent
        captured = prev.board().piece_at(square)
        if captured and captured.piece_type != chess.PAWN:
            tsq = prev.move.from_square if prev.move.to_square == square else square
            if _is_trapped(prev.parent.board().copy(stack=False), tsq):
                return (C().t("wins the trapped ").p(prev.parent.board(), tsq)
                        .t(" with ").m(node.parent.board(), node.move))
    return None


def _skewer(main, pov):
    for node in main[1::2][1:]:
        prev = node.parent
        capture = prev.board().piece_at(node.move.to_square)
        if (capture and _moved_pt(node) in RAY and not node.board().is_checkmate()):
            between = chess.SquareSet.between(node.move.from_square, node.move.to_square)
            op_move = prev.move
            if (op_move.to_square == node.move.to_square
                    or op_move.from_square not in between):
                continue
            if KVAL[_moved_pt(prev)] > KVAL[capture.piece_type] \
                    and _is_in_bad_spot(prev.board(), node.move.to_square):
                return (C().t("sets up a skewer: the ").p(prev.board(), node.move.to_square)
                        .t(" is won through the ").p(prev.board(), op_move.to_square)
                        .t(" with ").m(prev.board(), node.move))
    return None


def _interference(main, pov):
    for node in main[1::2][1:]:
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        if not capture or not _is_hanging(prev_board, capture, square):
            continue
        for interfering, init in (
                (node.parent, node.parent.parent),
                (node.parent.parent, getattr(node.parent.parent, "parent", None))):
            if init is None or interfering.move is None:
                continue
            if interfering is node.parent.parent and square == node.parent.move.to_square:
                continue
            init_board = init.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            dpiece = init_board.piece_at(defender) if defender is not None else None
            if (dpiece and dpiece.piece_type in RAY
                    and interfering.move.to_square in chess.SquareSet.between(square, defender)):
                return (C().t("sets up an interference: ")
                        .m(interfering.parent.board(), interfering.move)
                        .t(" cuts the ").p(init_board, defender)
                        .t("'s defense of the ").p(init_board, square)
                        .t(", which ").m(node.parent.board(), node.move).t(" wins"))
    return None


def _deflection(main, pov):
    for node in main[1::2][1:]:
        captured = node.parent.board().piece_at(node.move.to_square)
        if captured or node.move.promotion:
            if captured and KVAL[captured.piece_type] > KVAL[_moved_pt(node)]:
                continue
            square = node.move.to_square
            prev_op_move = node.parent.move
            grandpa = node.parent.parent
            prev_player_move = grandpa.move
            prev_player_capture = grandpa.parent.board().piece_at(prev_player_move.to_square)
            if (
                (not prev_player_capture
                 or VAL[prev_player_capture.piece_type] < _moved_pt(grandpa))
                and square != prev_op_move.to_square
                and square != prev_player_move.to_square
                and (prev_op_move.to_square == prev_player_move.to_square
                     or grandpa.board().is_check())
                and (square in grandpa.board().attacks(prev_op_move.from_square)
                     or node.move.promotion
                     and chess.square_file(node.move.to_square)
                     == chess.square_file(prev_op_move.from_square)
                     and node.move.from_square
                     in grandpa.board().attacks(prev_op_move.from_square))
                and (square not in node.parent.board().attacks(prev_op_move.to_square))
            ):
                c = C().t("works as a deflection: the ").p(
                    node.parent.board(), prev_op_move.to_square)
                c.t(" is pulled away from the defense of ")
                if captured:
                    c.t("the ").p(grandpa.board(), square).t(", which ")
                else:
                    c.s(square).t(", so ")
                return c.m(node.parent.board(), node.move).t(" follows")
    return None


def _attraction(main, pov):
    for node in main[1:]:
        if node.turn() == pov:
            continue
        first_move_to = node.move.to_square
        reply = node.next
        if reply and reply.move.to_square == first_move_to:
            attracted = _moved_pt(reply)
            if attracted in (chess.KING, chess.QUEEN, chess.ROOK):
                to_sq = reply.move.to_square
                nn = reply.next
                if nn and nn.move.to_square in nn.board().attackers(pov, to_sq):
                    n3 = nn.next.next if nn.next else None
                    if attracted == chess.KING or (n3 and n3.move.to_square == to_sq):
                        return (C().t("sets up the attraction ")
                                .m(node.parent.board(), node.move)
                                .t(", drawing the ")
                                .p(reply.parent.board(), reply.move.from_square)
                                .t(" onto ").s(to_sq))
    return None


def _intermezzo(main, pov):
    for node in main[1::2][1:]:
        if node.parent.board().is_capture(node.move):
            capture_move = node.move
            capture_square = node.move.to_square
            op_node = node.parent
            prev_pov_node = node.parent.parent
            if op_node.move.from_square not in prev_pov_node.board().attackers(
                    not pov, capture_square):
                if prev_pov_node.move.to_square != capture_square:
                    prev_op_node = prev_pov_node.parent
                    if (prev_op_node.move is not None
                            and prev_op_node.move.to_square == capture_square
                            and prev_op_node.parent is not None
                            and prev_op_node.parent.board().is_capture(prev_op_node.move)
                            and capture_move in prev_op_node.board().legal_moves):
                        c = C().t("throws in the intermezzo ").m(
                            prev_pov_node.parent.board(), prev_pov_node.move)
                        c.t(" before recapturing on ").s(capture_square)
                        why_board = prev_op_node.board()
                        defs = list(why_board.attackers(not pov, capture_square))
                        if defs:
                            ds = min(defs, key=lambda s: KVAL[why_board.piece_at(s).piece_type])
                            c.t("; a direct recapture would allow the ")
                            c.p(why_board, ds).t(" to take back")
                        return c
    return None


def _pin(main, pov):
    for node in main[1::2]:                          # pin prevents attack
        board = node.board()
        for square, piece in board.piece_map().items():
            if piece.color == pov:
                continue
            pin_dir = board.pin(piece.color, square)
            if pin_dir == chess.BB_ALL:
                continue
            for attack in board.attacks(square):
                attacked = board.piece_at(attack)
                if (attacked and attacked.color == pov
                        and attack not in pin_dir
                        and (VAL[attacked.piece_type] > VAL[piece.piece_type]
                             or _is_hanging(board, attacked, attack))):
                    return (C().t("exploits a pin: ").m(node.parent.board(), node.move)
                            .t(" is possible because the ").p(board, square)
                            .t(" is pinned and cannot take the ").p(board, attack))
    for node in main[1::2]:                          # pin prevents escape
        board = node.board()
        for pinned_square, pinned_piece in board.piece_map().items():
            if pinned_piece.color == pov:
                continue
            pin_dir = board.pin(pinned_piece.color, pinned_square)
            if pin_dir == chess.BB_ALL:
                continue
            for attacker_square in board.attackers(pov, pinned_square):
                attacker = board.piece_at(attacker_square)
                ok = VAL[pinned_piece.piece_type] > VAL[attacker.piece_type]
                if not ok:
                    ok = (_is_hanging(board, pinned_piece, pinned_square)
                          and pinned_square not in board.attackers(not pov, attacker_square)
                          and [m for m in board.pseudo_legal_moves
                               if m.from_square == pinned_square
                               and m.to_square not in pin_dir])
                if ok and attacker_square in pin_dir:
                    return (C().t("pins the ").p(board, pinned_square)
                            .t(" to the ").p(board, board.king(pinned_piece.color)))
    return None


def _discovered_check(main, pov):
    line = [n.move for n in main[1:]]
    boards = [n.board() for n in main]               # boards[i] = before line[i]
    for i, m in enumerate(line):
        if i % 2 == 0:
            ck = boards[i + 1].checkers()
            if ck and m.to_square not in ck:
                c = C()
                if i == 0:
                    c.t("discovers check: the ").p(boards[i], m.from_square)
                else:
                    c.t("sets up a discovered check: with ").m(boards[i], m)
                    c.t(", the ").p(boards[i], m.from_square)
                return (c.t(" steps aside, unmasking check from the ")
                        .p(boards[i + 1], next(iter(ck))))
    return None


def _discovered_attack(main, pov):
    line = [n.move for n in main[1:]]
    boards = [n.board() for n in main]
    for i, m in enumerate(line):                     # vacate-then-capture
        if i % 2 == 0 and i >= 2 and boards[i].is_capture(m):
            prev, parent = line[i - 2], line[i - 1]
            if parent.to_square == m.to_square:      # an immediate recapture disqualifies
                return None
            between = chess.SquareSet.between(m.from_square, m.to_square)
            if (prev.from_square in between and m.to_square != prev.to_square
                    and m.from_square != prev.to_square
                    and not boards[i - 2].is_castling(prev)):
                return (C().t("uncovers a discovered attack: the ")
                        .p(boards[i - 2], prev.from_square)
                        .t(" steps out of the line, and the ")
                        .p(boards[i], m.from_square).t(" then takes the ")
                        .p(boards[i], m.to_square))
    return None


# ============================================================ dispatch

EXTRACTORS = (("fork", _fork), ("hangingPiece", _hanging_piece),
              ("trappedPiece", _trapped_piece), ("skewer", _skewer),
              ("interference", _interference), ("deflection", _deflection),
              ("attraction", _attraction), ("intermezzo", _intermezzo),
              ("pin", _pin), ("discoveredCheck", _discovered_check),
              ("discoveredAttack", _discovered_attack))

NO_GAIN_OK = {"discoveredCheck", "pin"}   # motifs that don't claim a material win


def detect(fen, setup_uci, setup_parent_fen, line_ucis, max_motifs=1):
    """Run every pattern on the frame [setup, best, reply, ...].
    -> [(tag, parts), ...] (at most max_motifs, EXTRACTORS order), possibly []."""
    board = chess.Board(fen)
    setup = chess.Move.from_uci(setup_uci) if setup_uci else None
    parent_board = chess.Board(setup_parent_fen) if setup_parent_fen else None
    main = _frame(board, setup, parent_board,
                  [chess.Move.from_uci(u) for u in line_ucis])
    if len(main) < 2:
        return []
    pov = board.turn
    # lichess puzzles guarantee the line wins; our clear-best gate doesn't. Unless
    # the line mates or nets the mover material, "wins the ..." motifs are just
    # even trades wearing tactic shapes — only the non-material motifs may fire.
    end = main[-1].board()
    decisive = end.is_checkmate() or \
        _material_diff(end, pov) > _material_diff(main[0].board(), pov)
    fired = []
    for tag, fn in EXTRACTORS:
        if not decisive and tag not in NO_GAIN_OK:
            continue
        try:
            c = fn(main, pov)
        except Exception:
            c = None
        if c is not None:
            fired.append((tag, c.parts))
    # a raw fork supersedes deflection (the double attack is the truer story)
    if fork_targets(board, main[1].move, raw=True):
        fired = [f for f in fired if f[0] != "deflection"]
    return fired[:max_motifs]
