"""Tactic detection and acceptance for the verdict classes.

Common acceptance for every tactic example:
  (1) the tactic side is clearly better: >= +1.5 pawns (mover POV) at 10k nodes;
  (2) one clearly good move: the best move beats the engine's second line by
      >= 5% expected score (lichess win-rate formula);
  (3) the best line actually follows the tactic (cook tags on the shortest
      representative line, plus our own fork / discovered-attack detectors).

The tagger uses the lichess-puzzler primitives under datagen.tree.primitives
on a lichess-frame Puzzle: the game starts one ply before the tactic with the
opponent's actual setup move, then our line.
"""
import math
from itertools import combinations

import chess
import chess.pgn

from datagen.tree.primitives import lichess_cook as cook
from datagen.tree.primitives import lichess_util as util
from datagen.tree.primitives.lichess_model import Puzzle
from utils.translate_helpers import Translator

NODES = 10000
SCREEN_NODES = 1000     # cheap first-pass screen
EVAL_MIN = 150          # cp, mover POV
GAP_MIN = 0.05          # expected-score gap best vs second line
K = 0.00368208

# kept cook.py motifs; fork uses only our own detector, and a raw static fork
# on the best move supersedes deflection
COOK_TAGS = [
    "attraction", "deflection", "hangingPiece", "trappedPiece", "skewer",
    "interference", "intermezzo", "pin", "xRayAttack", "collinearMove",
    "mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "backRankMate",
]
# response priority when several motifs fire (rarest first)
TAC_PRIORITY = ["mate", "backRankMate", "collinearMove", "xRayAttack",
                "intermezzo", "interference", "skewer", "trappedPiece",
                "attraction", "deflection", "fork", "discovered attack",
                "pin", "hangingPiece"]

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
       chess.QUEEN: 9, chess.KING: 100}


def es(cp: float) -> float:
    """Lichess expected score from mover-POV centipawns."""
    # Stockfish mate scores live near +/-300_000 cp.  The direct logistic
    # expression overflows on large negative values even though its limiting
    # result is simply zero.  Evaluate the equivalent branch-stable form.
    x = K * cp
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ------------------------------------------------------------------ engines

class SfMulti:
    """MultiPV-2 facade over the shared ``SfJudge`` pool.

    Only the first move and the two scores are consumed below.  Keeping the
    historical list interface avoids duplicating the tactic filters while
    ensuring the short-lived ``Engines`` wrappers never own subprocesses.
    """

    def __init__(self, nodes=NODES):
        self.nodes = nodes
        from datagen.tree.search import SfJudge
        self.judge = SfJudge(nodes=nodes, multipv=2)

    def search2(self, fen):
        """-> [(move_uci, cp), ...] for multipv 1..2 (mover POV), [] if terminal."""
        best, cp1, _pv, cp2 = self.judge.search2(fen)
        if best is None:
            return []
        lines = [(best, cp1)]
        if cp2 is not None:
            lines.append((None, cp2))
        return lines


class Engines:
    """One process's engine set: MultiPV at 10k and 1k nodes, judge at 10k,
    rollout judge at 100. Built lazily on first use."""

    def __init__(self):
        self._m = self._m1 = self._j = self._jr = None

    @property
    def multi(self):
        if self._m is None:
            self._m = SfMulti()
        return self._m

    @property
    def screen(self):
        if self._m1 is None:
            self._m1 = SfMulti(nodes=SCREEN_NODES)
        return self._m1

    @property
    def judge(self):
        if self._j is None:
            from datagen.tree.search import SfJudge
            self._j = SfJudge(nodes=NODES)
        return self._j

    @property
    def rollout(self):
        if self._jr is None:
            from datagen.tree.search import SfJudge
            self._jr = SfJudge(nodes=100)
        return self._jr


def clear_best(eng: Engines, fen):
    """(best move uci, cp, gap) if the position has a clearly good move and the
    mover is clearly better, else None."""
    lines = eng.multi.search2(fen)
    if not lines:
        return None
    mv1, cp1 = lines[0]
    if cp1 < EVAL_MIN:
        return None
    gap = (es(cp1) - es(lines[1][1])) if len(lines) > 1 else 1.0
    if gap < GAP_MIN:
        return None
    return mv1, cp1, gap


def tactic_line(eng: Engines, board, mv1):
    """Shortest representative line: our side keeps playing only while it has a
    single clearly good move (same clear_best test at every pov position) --
    once it has several, the tactic is over and the line stops on our last
    move. Opponent replies are the 10k best. -> list of chess.Move."""
    b = board.copy()
    line, mv = [], mv1
    while len(line) < 12:
        m = chess.Move.from_uci(mv)
        line.append(m)
        b.push(m)                                   # pov move
        if b.is_game_over():
            break
        reply, _, _ = eng.judge.search(b.fen())
        if reply is None:
            break
        rm = chess.Move.from_uci(reply)
        b.push(rm)                                  # opponent's best reply
        if b.is_game_over():
            line.append(rm)
            break
        nxt = clear_best(eng, b.fen())
        if nxt is None:                             # several good moves: tactic over
            break
        line.append(rm)
        mv = nxt[0]
    return line


# ------------------------------------------------------------ fork detector

def _clustered(a: int, t1: int, t2: int) -> bool:
    """Attacker and both targets fit inside one 2x2 square."""
    fs = [chess.square_file(s) for s in (a, t1, t2)]
    rs = [chess.square_rank(s) for s in (a, t1, t2)]
    return max(fs) - min(fs) <= 1 and max(rs) - min(rs) <= 1


def fork_targets(board: chess.Board, move: chess.Move, raw: bool = False) -> list:
    """Qualifying fork-target squares after `move`, or [] if fewer than 2: the
    moved piece double-attacks enemy non-pawn pieces each either worth strictly
    more than it or undefended. Unless raw=True, a fork whose attacker and
    every target pair sit inside one 2x2 square does not count."""
    us = board.turn
    b = board.copy(stack=False)
    b.push(move)
    sq = move.to_square
    pc = b.piece_at(sq)
    if pc is None:
        return []
    av = VAL[pc.piece_type]
    targets = []
    for t in b.attacks(sq):
        p = b.piece_at(t)
        if p is None or p.color == us or p.piece_type == chess.PAWN:
            continue
        if VAL[p.piece_type] > av or not b.attackers(p.color, t):
            targets.append(t)
    if len(targets) < 2:
        return []
    if not raw and not any(not _clustered(sq, t1, t2)
                           for t1, t2 in combinations(targets, 2)):
        return []
    return targets


# ------------------------------------------- discovered-attack detector

def discovered_check(board: chess.Board, line: list) -> bool:
    b = board.copy(stack=False)
    for i, m in enumerate(line):
        b.push(m)
        if i % 2 == 0:
            ck = b.checkers()
            if ck and m.to_square not in ck:
                return True
    return False


def discovered_prose(board, line):
    """Translator for a discovered check / vacate-then-capture on the line, or None."""
    d = Translator(board.turn)
    b = board.copy(stack=False)
    boards = [b.copy(stack=False)]
    for m in line:
        b.push(m)
        boards.append(b.copy(stack=False))
    for i, m in enumerate(line):                      # discovered check
        if i % 2 == 0:
            ck = boards[i + 1].checkers()
            if ck and m.to_square not in ck:
                d.txt("the discovered check ").move(boards[i], m)
                d.txt(": the ").piece(boards[i], m.from_square)
                d.txt(" steps aside, unmasking check from the ")
                return d.piece(boards[i + 1], next(iter(ck)))
    for i, m in enumerate(line):                      # vacate-then-capture
        if i % 2 == 0 and i >= 2 and boards[i].is_capture(m):
            prev, parent = line[i - 2], line[i - 1]
            if parent.to_square == m.to_square:
                return None
            between = chess.SquareSet.between(m.from_square, m.to_square)
            if (prev.from_square in between and m.to_square != prev.to_square
                    and m.from_square != prev.to_square
                    and not boards[i - 2].is_castling(prev)):
                d.txt("the discovered attack ").move(boards[i - 2], prev)
                d.txt(": the ").piece(boards[i - 2], prev.from_square)
                d.txt(" steps out of the line, and the ")
                d.piece(boards[i], m.from_square).txt(" then takes the ")
                return d.piece(boards[i], m.to_square)
    return None


# ----------------------------------------------------------- tagger bridge

def build_puzzle(board_before: chess.Board, setup: chess.Move, line: list,
                 cp: int = 0) -> Puzzle:
    """Lichess-frame Puzzle: board_before = position before the opponent's
    setup move, line = our shortest representative line (pov moves first)."""
    game = chess.pgn.Game()
    game.setup(board_before)
    node = game.add_main_variation(setup)
    for m in line:
        node = node.add_main_variation(m)
    return Puzzle(id="scan", game=game, cp=cp)


# ------------------------------------- motif explanation payloads (Translator per tag)
# Each extractor mirrors the matching cook.py loop exactly but returns WHAT
# fired, taking the same Puzzle object cook.cook gets.

def attraction(puzzle):
    for node in puzzle.mainline[1:]:
        if node.turn() == puzzle.pov:
            continue
        first_move_to = node.move.to_square
        reply = util.next_node(node)
        if reply and reply.move.to_square == first_move_to:
            attracted = util.moved_piece_type(reply)
            if attracted in [chess.KING, chess.QUEEN, chess.ROOK]:
                to_sq = reply.move.to_square
                nn = util.next_node(reply)
                if nn and nn.move.to_square in nn.board().attackers(puzzle.pov, to_sq):
                    n3 = util.next_next_node(nn)
                    if attracted == chess.KING or (n3 and n3.move.to_square == to_sq):
                        d = Translator(puzzle.pov)
                        return (d.txt("the attraction ").node(node)
                                .txt(", drawing the ")
                                .piece(reply.parent.board(), reply.move.from_square)
                                .txt(" onto ").sq(to_sq)
                                .txt(", after which ").node(nn).txt(" follows"))
    return None


def deflection(puzzle):
    for node in puzzle.mainline[1::2][1:]:
        captured = node.parent.board().piece_at(node.move.to_square)
        if captured or node.move.promotion:
            if captured and util.king_values[captured.piece_type] \
                    > util.king_values[util.moved_piece_type(node)]:
                continue
            square = node.move.to_square
            prev_op_move = node.parent.move
            grandpa = node.parent.parent
            prev_player_move = grandpa.move
            prev_player_capture = grandpa.parent.board().piece_at(
                prev_player_move.to_square)
            if (
                (not prev_player_capture
                 or util.values[prev_player_capture.piece_type]
                 < util.moved_piece_type(grandpa))
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
                d = Translator(puzzle.pov)
                d.txt("the deflection: the ").piece(
                    node.parent.board(), prev_op_move.to_square)
                d.txt(" is pulled away from the defense of ")
                if captured:
                    d.piece(grandpa.board(), square).txt(", allowing ")
                else:
                    d.sq(square).txt(", allowing ")
                return d.node(node)
    return None


def hanging_piece(puzzle):
    to = puzzle.mainline[1].move.to_square
    board0 = puzzle.mainline[0].board()
    if board0.piece_at(to) is None:
        return None
    d = Translator(puzzle.pov)
    return (d.txt("the hanging ").piece(board0, to).txt(", captured with ")
            .node(puzzle.mainline[1]))


def trapped_piece(puzzle):
    for node in puzzle.mainline[1::2][1:]:
        square = node.move.to_square
        prev = node.parent
        captured = prev.board().piece_at(square)
        if captured and captured.piece_type != chess.PAWN:
            tsq = prev.move.from_square if prev.move.to_square == square else square
            if util.is_trapped(prev.parent.board(), tsq):
                d = Translator(puzzle.pov)
                return (d.txt("the trapped ").piece(prev.parent.board(), tsq)
                        .txt(", which ").node(node).txt(" wins"))
    return None


def skewer(puzzle):
    for node in puzzle.mainline[1::2][1:]:
        prev = node.parent
        capture = prev.board().piece_at(node.move.to_square)
        if (capture and util.moved_piece_type(node) in util.ray_piece_types
                and not node.board().is_checkmate()):
            between = chess.SquareSet.between(node.move.from_square,
                                              node.move.to_square)
            op_move = prev.move
            if (op_move.to_square == node.move.to_square
                    or op_move.from_square not in between):
                continue
            if util.king_values[util.moved_piece_type(prev)] > util.king_values[
                    capture.piece_type] and util.is_in_bad_spot(
                    prev.board(), node.move.to_square):
                d = Translator(puzzle.pov)
                return (d.txt("the skewer: the ")
                        .piece(prev.board(), node.move.to_square)
                        .txt(" is won through the ")
                        .piece(prev.board(), op_move.to_square)
                        .txt(" with ").node(node))
    return None


def interference(puzzle):
    for node in puzzle.mainline[1::2][1:]:
        prev_board = node.parent.board()
        square = node.move.to_square
        capture = prev_board.piece_at(square)
        if not capture or not util.is_hanging(prev_board, capture, square):
            continue
        for interfering, init in (
                (node.parent, node.parent.parent),
                (node.parent.parent, getattr(node.parent.parent, "parent", None))):
            if init is None or interfering.move is None:
                continue
            if interfering is node.parent.parent \
                    and square == node.parent.move.to_square:
                continue
            init_board = init.board()
            defenders = init_board.attackers(capture.color, square)
            defender = defenders.pop() if defenders else None
            dpiece = init_board.piece_at(defender) if defender is not None else None
            if (dpiece and dpiece.piece_type in util.ray_piece_types
                    and interfering.move.to_square
                    in chess.SquareSet.between(square, defender)):
                d = Translator(puzzle.pov)
                return (d.txt("the interference ").node(interfering)
                        .txt(", cutting the ").piece(init_board, defender)
                        .txt("'s defense of the ").piece(init_board, square)
                        .txt(", which ").node(node).txt(" wins"))
    return None


def intermezzo(puzzle):
    for node in puzzle.mainline[1::2][1:]:
        if util.is_capture(node):
            capture_move = node.move
            capture_square = node.move.to_square
            op_node = node.parent
            prev_pov_node = node.parent.parent
            if op_node.move.from_square not in prev_pov_node.board().attackers(
                    not puzzle.pov, capture_square):
                if prev_pov_node.move.to_square != capture_square:
                    prev_op_node = prev_pov_node.parent
                    if (prev_op_node.move.to_square == capture_square
                            and util.is_capture(prev_op_node)
                            and capture_move in prev_op_node.board().legal_moves):
                        d = Translator(puzzle.pov)
                        d.txt("the intermezzo ").node(prev_pov_node)
                        d.txt(", thrown in before recapturing on ")
                        d.sq(capture_square)
                        why_board = prev_op_node.board()
                        defs = list(why_board.attackers(not puzzle.pov,
                                                        capture_square))
                        if defs:
                            ds = min(defs, key=lambda s: VAL[
                                why_board.piece_at(s).piece_type])
                            d.txt("; a direct recapture would allow the ")
                            d.piece(why_board, ds).txt(" to take back")
                        return d
    return None


def pin(puzzle):
    for node in puzzle.mainline[1::2]:               # pin prevents attack
        board = node.board()
        for square, piece in board.piece_map().items():
            if piece.color == puzzle.pov:
                continue
            pin_dir = board.pin(piece.color, square)
            if pin_dir == chess.BB_ALL:
                continue
            for attack in board.attacks(square):
                attacked = board.piece_at(attack)
                if (attacked and attacked.color == puzzle.pov
                        and attack not in pin_dir
                        and (util.values[attacked.piece_type]
                             > util.values[piece.piece_type]
                             or util.is_hanging(board, attacked, attack))):
                    d = Translator(puzzle.pov)
                    return (d.txt("the pin: ").node(node)
                            .txt(" is possible because the ").piece(board, square)
                            .txt(" is pinned and cannot take the ")
                            .piece(board, attack))
    for node in puzzle.mainline[1::2]:               # pin prevents escape
        board = node.board()
        for pinned_square, pinned_piece in board.piece_map().items():
            if pinned_piece.color == puzzle.pov:
                continue
            pin_dir = board.pin(pinned_piece.color, pinned_square)
            if pin_dir == chess.BB_ALL:
                continue
            for attacker_square in board.attackers(puzzle.pov, pinned_square):
                attacker = board.piece_at(attacker_square)
                ok = util.values[pinned_piece.piece_type] > util.values[
                    attacker.piece_type]
                if not ok:
                    ok = (util.is_hanging(board, pinned_piece, pinned_square)
                          and pinned_square not in board.attackers(
                              not puzzle.pov, attacker_square)
                          and [m for m in board.pseudo_legal_moves
                               if m.from_square == pinned_square
                               and m.to_square not in pin_dir])
                if ok and attacker_square in pin_dir:
                    d = Translator(puzzle.pov)
                    return (d.txt("the pin: the ").piece(board, attacker_square)
                            .txt(" pins the ").piece(board, pinned_square)
                            .txt(" to the king"))
    return None


def x_ray(puzzle):
    """Exact geometry for the lichess-cook x-ray tag."""
    for node in puzzle.mainline[1::2][1:]:
        if not util.is_capture(node):
            continue
        screen = node.parent
        prior = screen.parent
        if (screen.move.to_square != node.move.to_square
                or util.moved_piece_type(screen) == chess.KING
                or prior.move.to_square != screen.move.to_square
                or screen.move.from_square not in chess.SquareSet.between(
                    node.move.from_square, node.move.to_square
                )):
            continue
        d = Translator(puzzle.pov)
        return (d.txt("the x-ray attack: the ")
                .piece(node.parent.board(), node.move.from_square)
                .txt(" attacks through the ")
                .piece(screen.parent.board(), screen.move.from_square)
                .txt(", and after ").node(screen).txt(", ")
                .node(node).txt(" wins it"))
    return None


def collinear_move(puzzle):
    """Exact pieces and line for the lichess-cook collinear-move tag."""
    from datagen.tree.primitives.lichess_util import squares_are_collinear

    for node in puzzle.mainline[1::2]:
        moving_type = util.moved_piece_type(node)
        if moving_type not in util.ray_piece_types or util.is_capture(node):
            continue
        board = node.parent.board()
        start, end = node.move.from_square, node.move.to_square
        for square in board.attacks(start):
            piece = board.piece_at(square)
            if (piece is None or piece.color == puzzle.pov
                    or piece.piece_type not in util.ray_piece_types
                    or not squares_are_collinear(start, square, end)
                    or chess.Move(start, square) not in board.legal_moves):
                continue
            d = Translator(puzzle.pov)
            return (d.txt("the collinear move ").node(node)
                    .txt(": the ").piece(board, start)
                    .txt(" remains aligned with the ").piece(board, square))
    return None


def back_rank_mate(puzzle):
    """Exact mating move and king for the lichess-cook back-rank-mate tag."""
    node = puzzle.game.end()
    board = node.board()
    if not board.is_checkmate():
        return None
    king_square = board.king(not puzzle.pov)
    if king_square is None:
        return None
    d = Translator(puzzle.pov)
    return (d.txt("the back-rank mate ").node(node)
            .txt(", checkmating the ").piece(board, king_square))


def mate_line(puzzle):
    """mate in N: the line is the justification."""
    if not puzzle.game.end().board().is_checkmate():
        return None
    n = len(puzzle.mainline[1:]) // 2 + len(puzzle.mainline[1:]) % 2
    d = Translator(puzzle.pov)
    d.txt(f"mate in {n}: ")
    for i, node in enumerate(puzzle.mainline[1:]):
        if i:
            d.txt(" ")
        d.node(node)
    return d


EXTRACTORS = {"attraction": attraction, "deflection": deflection,
              "hangingPiece": hanging_piece, "trappedPiece": trapped_piece,
              "skewer": skewer, "interference": interference,
              "intermezzo": intermezzo, "pin": pin,
              "xRayAttack": x_ray, "collinearMove": collinear_move,
              "backRankMate": back_rank_mate,
              "mateIn1": mate_line, "mateIn2": mate_line, "mateIn3": mate_line,
              "mateIn4": mate_line, "mateIn5": mate_line}


def duos(puzzle, tags) -> dict:
    """{tag: Translator} for the kept motifs present in tags."""
    out = {}
    for t in tags:
        fn = EXTRACTORS.get(t)
        if fn is None:
            continue
        try:
            duo = fn(puzzle)
        except Exception:
            duo = None
        if duo:
            out[t] = duo
    return out


def cook_tags(puzzle) -> list:
    try:
        return cook.cook(puzzle)
    except Exception:
        return []
