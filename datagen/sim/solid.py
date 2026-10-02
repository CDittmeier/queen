"""Board-derived "solid" conclusions for tree leaves.

The verdict model only describes the position in front of it, so a leaf that
is one forced recapture away from a settled position gets a static reason for
a dynamic fact. This walks the leaf forward along a single oracle line (both
sides play Stockfish best at ORACLE_NODES) for up to EXT_PLY plies, testing at
each position for the strongest provable conclusion — mate > tactic >
material — and returns the shortest one found:

  mate      the oracle scores a forced mate for the favoured side; the pv is
            spelled out move by move as the proof;
  tactic    only when the favoured side is to move: the corpus tactic
            acceptance (cheap screen with slack, clear-best at 10k, cook tags
            + own fork / discovered-attack detectors, TAC_PRIORITY prose);
  material  only when the position is quiet (not in check, and the oracle's
            own best move is neither a capture nor a check — a running
            exchange must finish before material is counted); >= MAT_MIN
            pawns for the favoured side, and with `require_new` the count
            must differ from the baseline (the material must be something the
            line produced).

A non-mate conclusion under 0.1 pawns (below the smallest verdict band) is
discarded. Sentences come back as (human, machine) pairs — "After {moves},
{side} is {verdict} due to {reason}." — with the machine side in POV tokens
anchored at the LEAF's side to move (re-anchor with
utils.translate_helpers.Translator when embedding in a root-anchored
narrative).
"""
import os

import chess

from datagen.sim import tactics
from datagen.sim.describe import fork_prose, material_prose, verdict_head
from datagen.tree.strong_primitives import material_w
from datagen.tree.search import MATE, SfJudge
from utils.translate_helpers import Translator

EXT_PLY = 8              # longest extension tried
ORACLE_NODES = int(os.environ.get("MT_ORACLE_NODES", 100_000))
MAT_MIN = 1.0            # pawns of new material that count as solid


def _tactic(eng, before, setup, board):
    """A tactic for the side to move at `board`, squares detector-derived."""
    fen = board.fen()
    lines1 = eng.screen.search2(fen)
    if not lines1:
        return None
    cp1k = lines1[0][1]
    if not (cp1k >= tactics.EVAL_MIN - 30
            and (len(lines1) == 1
                 or tactics.es(cp1k) - tactics.es(lines1[1][1])
                 >= tactics.GAP_MIN - 0.02)):
        return None
    cb = tactics.clear_best(eng, fen)
    if cb is None:
        return None
    mv1, cp10, _gap = cb
    line = tactics.tactic_line(eng, board, mv1)
    m1 = chess.Move.from_uci(mv1)
    puzzle = tactics.build_puzzle(before, setup, line, cp10)
    tags = tactics.cook_tags(puzzle)
    raw_fork = tactics.fork_targets(board, m1, raw=True)
    tags = [t for t in tags if t in tactics.COOK_TAGS
            and not (t == "deflection" and raw_fork)]
    dd = tactics.duos(puzzle, tags)
    targets = tactics.fork_targets(board, m1)
    our_fork = False
    if targets and len(line) >= 3:
        tracked = set(targets)
        if line[1].from_square in tracked:
            tracked.discard(line[1].from_square)
            tracked.add(line[1].to_square)
        b = board.copy()
        b.push(line[0])
        b.push(line[1])
        if b.is_capture(line[2]) and line[2].to_square in tracked \
                and line[2].from_square == m1.to_square:
            our_fork = True
    disc = tactics.discovered_prose(board, line)
    motifs = (["fork"] if our_fork else []) \
        + (["discovered attack"] if disc else []) + tags
    is_mate = any(t.startswith("mateIn") for t in tags)
    duo = None
    for t in tactics.TAC_PRIORITY:
        if t == "mate" and is_mate:
            duo = next((dd[x] for x in dd if x.startswith("mateIn")), None)
        elif t == "fork" and our_fork:
            duo = fork_prose(board, m1, targets)
        elif t == "discovered attack" and disc:
            duo = disc
        elif t in dd:
            duo = dd[t]
        if duo is not None:
            break
    if duo is None:
        return None
    ev = cp10 / 100.0
    return duo, (ev if board.turn else -ev), is_mate, motifs


def _mate(board, cp_w, sgn, pv):
    """A proven forced mate for the favoured side -- the strongest conclusion
    the board can offer. The mate distance alone is an assertion; the line is
    the proof, so the pv is spelled out move by move (through Translator, so the
    machine rendering stays valid). A pv that runs out early names what it
    has."""
    if cp_w * sgn < MATE - 1000:
        return None
    n = MATE - abs(cp_w)
    d = Translator(board.turn).txt(f"a forced mate in {n}")
    b, cnt = board.copy(stack=False), 0
    for u in pv:
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break
        if mv not in b.legal_moves:
            break
        d.txt(": " if not cnt else " ").move(b, mv)
        b.push(mv)
        cnt += 1
    return d, cp_w / 100.0, True, ["mate"]


def _material(board, eval_w, base_mat, sgn, require_new):
    mat = material_w(board)
    if mat * sgn < MAT_MIN:
        return None
    if require_new and mat == base_mat:
        return None
    return material_prose(Translator(board.turn), mat), eval_w, False, ["material"]


def _quiet(board, oracle):
    """Is the position settled -- nothing left to take and no check pending?

    Testing a single reply is not enough: an exchange can run several plies and
    the count slides the whole way down it. A material claim waits until the
    oracle's own best move is neither a capture nor a check."""
    if board.is_check():
        return False
    mv, _, _ = oracle.search(board.fen())
    if mv is None:
        return True
    m = chess.Move.from_uci(mv)
    return not (board.is_capture(m) or board.gives_check(m))


class SolidExtender:
    """Oracle and tactic engines held once; conclusion() is called per leaf."""

    def __init__(self, ply=EXT_PLY, nodes=ORACLE_NODES):
        self.ply = ply
        self.oracle = SfJudge(nodes=nodes)
        self.eng = tactics.Engines()

    def _eval(self, fen):
        """(White-POV centipawns, principal variation) at `fen`."""
        _, cp, pv = self.oracle.search(fen)
        return (cp if chess.Board(fen).turn else -cp), pv

    def conclusion(self, leaf_fen, parent_fen, move_uci, base_fen,
                   require_new=True):
        """((human, machine), motifs, sans) for the shortest solid extension,
        or None. `parent_fen`/`move_uci` are the move into the leaf (the 0-ply
        tactic test needs the position it was played from); `base_fen` is the
        material baseline. `require_new` demands the material be something the
        line produced -- true when explaining what a refuted move cost, false
        when simply asking why a position is winning. Machine text is anchored
        at the LEAF's side to move."""
        b = chess.Board(leaf_fen)
        anchor = b.turn                                    # POV anchor
        sgn = 1 if self._eval(leaf_fen)[0] > 0 else -1     # favoured side
        base_mat = material_w(chess.Board(base_fen))
        before, setup = chess.Board(parent_fen), chess.Move.from_uci(move_uci)
        sans = []
        walk = Translator(anchor)                                 # the "After ..." moves
        for k in range(self.ply + 1):
            if k:
                mv_uci, _, _ = self.oracle.search(b.fen())
                if mv_uci is None:
                    break
                mv = chess.Move.from_uci(mv_uci)
                before, setup = b.copy(stack=False), mv
                if sans:
                    walk.txt(" ")
                walk.move(b, mv)
                sans.append(b.san(mv))
                b.push(mv)
            cp_w, pv = self._eval(b.fen())
            eval_w = cp_w / 100.0
            hit = _mate(b, cp_w, sgn, pv)
            if hit is None and not b.is_game_over() and b.turn == (sgn > 0):
                hit = _tactic(self.eng, before, setup, b)  # a tactic is for the mover
            if hit is None and _quiet(b, self.oracle):
                hit = _material(b, eval_w, base_mat, sgn, require_new)
            if hit is not None:
                duo, ev, is_mate, motifs = hit
                if abs(ev) < 0.1 and not is_mate:
                    hit = None          # below the smallest verdict band
            if hit is not None:
                duo, ev, is_mate, motifs = hit
                head = verdict_head(Translator(b.turn), ev, is_mate)
                sent_h = "".join(head.h) + "".join(duo.h) + "."
                # head/duo are anchored at the walk position's side to move;
                # bring them to the leaf anchor before assembling
                sent_m = Translator.reanchor(
                    "".join(head.m) + "".join(duo.m) + ".", b.turn, anchor)
                if sans:
                    sent_h = f"After {' '.join(sans)}, {sent_h}"
                    sent_m = f"After {''.join(walk.m)}, {sent_m}"
                return (sent_h, sent_m), motifs, list(sans)
            if b.is_game_over():
                break
        return None
