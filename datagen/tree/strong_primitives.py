"""Strong verdict primitives: one position -> classed prose or ``None``.

Classes, in response priority order:
  mate/tactic  1000-node screen -> clear-best at 10k (+1.5 mover POV, 5%
               expected-score gap) -> shortest representative line -> cook
               tags (deflection superseded by a raw fork) + our fork /
               discovered-attack detectors;
  positional   dominant activity feature at/above its frozen percentile
               threshold, material equal-or-behind for its side, 10k eval
               0.8-3.0 that side, 8-ply no-tactic rollout (rollout moves at
               100 nodes);
  material     nonzero material count, 1k screen >= +1.2 that side, 10k eval
               >= +1.5 that side;
  balance      dynamic (threshold-strong feature each side) / dry (none),
               10k eval within +-0.3.

Common classes are thinned by KEEP_P before any engine work, so a corpus scan
lands near CAP_PER_M rows per class per million positions scanned (the
per-class cap itself is enforced by the caller). The RNG driving KEEP_P is
seeded per position (crc32 of the fen), so acceptance is reproducible.
"""
import random
import zlib

import chess

from datagen.tree.primitives import activity
from datagen.tree.primitives.core.context import get_context
from datagen.sim import tactics
from datagen.sim.describe import (balance_sentence, feature_prose,
                                      fork_prose, material_prose, verdict_head)
from utils.translate_helpers import Translator
from datagen.sim.tactics import (COOK_TAGS, TAC_PRIORITY, Engines,
                                     clear_best, es, tactic_line)

CAP_PER_M = 20000
# natural occurrence per 1M generic positions (feature_yield on a 100k sample)
RATES = {"bad bishop": 162870, "passed pawns": 88510, "bad knight": 60350,
         "knight outpost": 37560, "space": 32420, "colour complexion": 23960,
         "active bishop pair": 21210, "pawn structure": 20430,
         "active rook": 15660, "active lone bishop": 9400, "bad rook": 5860,
         "material": 393600, "dry": 200000, "dynamic": 20000}
KEEP_P = {k: min(1.0, 3.0 * CAP_PER_M / r) for k, r in RATES.items()}

MAT = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
       chess.QUEEN: 9}


def material_w(board: chess.Board) -> int:
    m = 0
    for p in board.piece_map().values():
        if p.piece_type != chess.KING:
            m += MAT[p.piece_type] if p.color else -MAT[p.piece_type]
    return m


def primary_class(motifs: list) -> str:
    """The training-cap bucket of a motif list: 'tactic' for any tactical
    motif, else the motif itself (a feature name / material / dynamic / dry)."""
    key = motifs[0]
    return ("tactic" if key in TAC_PRIORITY or key.startswith("mateIn")
            or key == "backRankMate" else key)


def sf10k_w(eng: Engines, fen: str):
    """10k-node eval, White-POV pawns, or None on engine failure."""
    try:
        _, cp, _ = eng.judge.search(fen)
    except Exception:
        return None
    return (cp if fen.split()[1] == "w" else -cp) / 100.0


def no_tactic_rollout(eng: Engines, board: chess.Board, side: int) -> bool:
    """8 plies at 100 nodes; `side` must never hold material > 0 for 2 plies."""
    b = board.copy()
    streak = 0
    for _ in range(8):
        try:
            mv, _, _ = eng.rollout.search(b.fen())
        except Exception:
            return False
        if mv is None:
            break
        b.push(chess.Move.from_uci(mv))
        streak = streak + 1 if material_w(b) * side > 0 else 0
        if streak >= 2:
            return False
        if b.is_game_over():
            break
    return True


def classify(eng: Engines, before: chess.Board, setup: chess.Move,
             rng: random.Random = None):
    """The position after `setup` played on `before`, through the class
    acceptances. Returns {"human", "machine", "motifs", "eval_w"} (prose with
    the trailing period) or None. `before`/`setup` exist because the tactic
    tagger needs the opponent's actual setup move one ply earlier."""
    board = before.copy()
    if setup not in board.legal_moves:
        return None
    board.push(setup)
    if board.is_game_over():
        return None
    fen = board.fen()
    if rng is None:
        rng = random.Random(zlib.crc32(fen.encode()))
    thr = activity.position_thresholds()

    felt = None
    if not board.is_check():
        try:
            felt = activity.measure(board, get_context(board))
        except Exception:
            felt = None
    mat = material_w(board)

    # ---- candidates (cheap) ----
    dom = None
    if felt:
        norms = {n: abs(felt[n]) / thr[n] for n in thr}
        best = max(norms, key=norms.get)
        if norms[best] >= 1.0 and rng.random() < KEEP_P[best]:
            dom = best
        strong_w = [n for n in thr if felt[n] >= thr[n]]
        strong_b = [n for n in thr if -felt[n] >= thr[n]]
    else:
        strong_w = strong_b = []
    mat_cand = mat != 0 and rng.random() < KEEP_P["material"]
    dyn_cand = bool(strong_w) and bool(strong_b)
    dry_cand = felt is not None and not strong_w and not strong_b \
        and rng.random() < KEEP_P["dry"]

    # ---- engine screen (always) ----
    lines1 = eng.screen.search2(fen)
    if not lines1:
        return None
    cp1k = lines1[0][1]

    motifs, duo, eval_w, is_mate = [], None, None, False

    # ---- tactics ----
    tac_gate = cp1k >= tactics.EVAL_MIN - 30 and (
        len(lines1) == 1
        or es(cp1k) - es(lines1[1][1]) >= tactics.GAP_MIN - 0.02)
    if tac_gate:
        cb = clear_best(eng, fen)
        if cb is not None:
            mv1, cp10, gap = cb
            line = tactic_line(eng, board, mv1)
            m1 = chess.Move.from_uci(mv1)
            puzzle = tactics.build_puzzle(before, setup, line, cp10)
            tags = tactics.cook_tags(puzzle)
            raw_fork = tactics.fork_targets(board, m1, raw=True)
            tags = [t for t in tags if t in COOK_TAGS
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
            if our_fork:
                motifs.append("fork")
            if disc:
                motifs.append("discovered attack")
            motifs += tags
            is_mate = any(t.startswith("mateIn") for t in tags)
            eval_mover = cp10 / 100.0
            eval_w = eval_mover if board.turn else -eval_mover
            for t in TAC_PRIORITY:
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
                motifs, eval_w, is_mate = [], None, False

    # ---- positional ----
    if duo is None and dom and felt:
        side = 1 if felt[dom] > 0 else -1
        if mat * side <= 0:
            e = sf10k_w(eng, fen)
            if e is not None and 0.8 <= e * side <= 3.0 \
                    and no_tactic_rollout(eng, board, side):
                b2 = chess.Board(fen)
                d = feature_prose(dom, b2, get_context(b2), felt[dom])
                if d is not None:
                    duo, eval_w, motifs = d, e, [dom] + motifs

    # ---- material ----
    if duo is None and mat_cand:
        side = 1 if mat > 0 else -1
        mover = 1 if board.turn else -1
        if cp1k / 100.0 * side * mover >= 1.2:
            e = sf10k_w(eng, fen)
            if e is not None and e * side >= 1.5:
                duo = material_prose(Translator(board.turn), mat)
                eval_w, motifs = e, ["material"] + motifs

    # ---- balance ----
    if duo is None and (dyn_cand or dry_cand):
        e = sf10k_w(eng, fen)
        if e is not None and abs(e) <= 0.3:
            if dyn_cand:
                sent = balance_sentence("dynamic", board.turn, strong_w, strong_b)
                motifs = ["dynamic"] + motifs
            else:
                sent = balance_sentence("dry", board.turn)
                motifs = ["dry"] + motifs
            return {"human": sent.human + ".", "machine": sent.machine + ".",
                    "motifs": motifs, "eval_w": e}

    if duo is None:
        return None
    head = verdict_head(Translator(board.turn), eval_w, is_mate)
    return {"human": head.human + duo.human + ".",
            "machine": head.machine + duo.machine + ".",
            "motifs": motifs, "eval_w": eval_w}
