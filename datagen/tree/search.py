"""Build the search Tree for a position.

get_tree(fen) does two things, exactly as the lab/hce_eval pipeline did:

  1. A full-width alpha-beta search to `depth` (default 3) with quiescence at the
     horizon, using Stockfish's static NNUE as the value — the "Stockfish depth-D
     tree" (which moves it considered, and its verdict on each).
  2. Maia-1100's policy is then consulted at every node: any sibling the network
     would play at least as readily as the search-best move is surfaced (capped at
     `cap`=2), so the tree also contains the branches maia wanted that the best line
     skipped. Refuted deviations are shown as a single refutation line.

The result is assembled into a `Tree` (see tree.py) whose children are kept in both
the search order and our checks>captures>threats>other order, with refuted
candidates pointing at their refuter.

Engines (Stockfish, lc0-eigen maia-1100) are created lazily and reused across calls.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import chess

from datagen.tree.tree import Tree
from datagen.tree.moves import move_priority   # CCT ordering, local to datagen/tree

# Engine binaries / nets, relative to the repo root (all code is run from p-chess-lm/).
SF_BIN = Path("data/engines/stockfish_25080907_x64_avx2")
LC0_BIN = Path("data/lc0/lc0-src/lc0")
MAIA_NET = Path("data/engines/lc0_networks/maia-1100.pb.gz")
MATE = 300000
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


# ============================================================ engines

class SfEval:
    """Persistent Stockfish; eval(fen) -> static-NNUE cp from side-to-move POV.
    Non-check: `eval` (white-POV static NNUE, negated for black). In-check (static
    NNUE undefined): `go depth 1` (already mover-POV)."""

    def __init__(self, sf=SF_BIN):
        self.p = subprocess.Popen([str(sf)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.send("uci")
        for ln in self.p.stdout:
            if ln.startswith("uciok"):
                break
        self.cache = {}

    def send(self, x):
        self.p.stdin.write(x + "\n"); self.p.stdin.flush()

    def eval(self, fen):
        if fen in self.cache:
            return self.cache[fen]
        b = chess.Board(fen)
        wp = None
        if not b.is_check():
            self.send(f"position fen {fen}"); self.send("eval"); self.send("isready")
            for ln in self.p.stdout:
                m = re.search(r"Final evaluation\s+([-+]?\d+\.\d+)", ln)
                if m:
                    wp = float(m.group(1))
                if ln.startswith("readyok"):
                    break
        if wp is not None:
            v = int(round(wp * 100))
            v = v if b.turn == chess.WHITE else -v
        else:
            self.send(f"position fen {fen}"); self.send("go depth 1")
            cp = None
            for ln in self.p.stdout:
                m = re.search(r"score cp (-?\d+)", ln)
                if m:
                    cp = int(m.group(1))
                mm = re.search(r"score mate (-?\d+)", ln)
                if mm:
                    n = int(mm.group(1)); cp = (MATE - abs(n)) * (1 if n > 0 else -1)
                if ln.startswith("bestmove"):
                    break
            v = cp if cp is not None else 0
        self.cache[fen] = v
        return v

    def close(self):
        try:
            self.send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class MaiaPolicy:
    """Persistent lc0 (maia-1100, eigen); policy(fen) -> {uci: prob} from the policy head."""
    UCI = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

    def __init__(self, net=MAIA_NET, lc0=LC0_BIN):
        self.p = subprocess.Popen([str(lc0), f"--weights={net}", "--backend=eigen"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("uci")
        for ln in self.p.stdout:
            if ln.startswith("uciok"):
                break
        self.send("setoption name VerboseMoveStats value true")
        self.cache = {}

    def send(self, x):
        self.p.stdin.write(x + "\n"); self.p.stdin.flush()

    def policy(self, fen):
        if fen in self.cache:
            return self.cache[fen]
        self.send(f"position fen {fen}"); self.send("go nodes 1")
        pol, buf = {}, []
        for ln in self.p.stdout:
            buf.append(ln)
            if ln.startswith("bestmove"):
                break
        for ln in buf:
            if "info string" in ln and "(P:" in ln:
                toks = ln.split()
                mv = toks[toks.index("string") + 1]
                if self.UCI.match(mv):
                    pol[mv] = float(ln.split("(P:")[1].split("%")[0]) / 100.0
        self.cache[fen] = pol
        return pol

    def close(self):
        try:
            self.send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class SfJudge:
    """Stockfish at a fixed node budget — a real (few-node) player used to judge which
    positions are 'better' and whether a move is a blunder. Static NNUE eval is
    tactically blind (it misses e.g. a winning queen grab); a 100–1000 node search is
    not, and ~100 nodes already rates as a decent player on our ladder. search(fen) ->
    (bestmove_uci | None, cp) with cp from the side-to-move's POV."""

    def __init__(self, nodes=1000, sf=SF_BIN, multipv=2):
        self.nodes = int(nodes)
        self.multipv = int(multipv)
        if self.multipv not in (1, 2):
            raise ValueError("SfJudge multipv must be 1 or 2")
        self.p = subprocess.Popen([str(sf)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.send("uci")
        for ln in self.p.stdout:
            if ln.startswith("uciok"):
                break
        # MultiPV=2 provides the clear-best gap. Callers needing only the
        # primary evaluation (for example the plan blunder screen) can use 1.
        self.send(f"setoption name MultiPV value {self.multipv}")
        self.cache = {}

    def send(self, x):
        self.p.stdin.write(x + "\n"); self.p.stdin.flush()

    def search(self, fen):
        """-> (bestmove_uci | None, cp from side-to-move POV, pv as a list of uci moves)."""
        return self.search2(fen)[:3]

    def search2(self, fen):
        """search() plus the second line's cp (None when there is no second move)."""
        # A pooled engine may be reused at several node budgets.  Keep its
        # private cache exact; cross-budget subsumption belongs in sfpool's
        # capability-aware shared cache.
        key = (self.nodes, self.multipv, fen)
        if key in self.cache:
            return self.cache[key]
        b = chess.Board(fen)
        if b.is_checkmate():
            self.cache[key] = (None, -MATE, [], None); return self.cache[key]
        if b.is_stalemate() or b.is_insufficient_material() or b.can_claim_draw():
            self.cache[key] = (None, 0, [], None); return self.cache[key]
        self.send("ucinewgame")                       # clear the TT so evals don't depend on call history
        self.send(f"position fen {fen}"); self.send(f"go nodes {self.nodes}")
        cp, best, pv = {1: 0, 2: None}, None, {1: [], 2: []}
        for ln in self.p.stdout:
            if ln.startswith("info"):
                mp = re.search(r" multipv (\d+)", ln)
                idx = int(mp.group(1)) if mp else 1
                m = re.search(r"score cp (-?\d+)", ln)
                if m:
                    cp[idx] = int(m.group(1))
                mm = re.search(r"score mate (-?\d+)", ln)
                if mm:
                    n = int(mm.group(1)); cp[idx] = (MATE - abs(n)) * (1 if n > 0 else -1)
                pm = re.search(r" pv (.+)", ln)
                if pm:
                    pv[idx] = pm.group(1).split()
            elif ln.startswith("bestmove"):
                tok = ln.split()
                best = tok[1] if len(tok) > 1 and tok[1] != "(none)" else None
                break
        self.cache[key] = (best, cp[1], pv[1], cp[2])
        return self.cache[key]

    def eval(self, fen):
        return self.search(fen)[1]

    def close(self):
        try:
            self.send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# ============================================================ alpha-beta

def ordered(board):
    def key(m):
        if board.is_capture(m):
            vic = board.piece_at(m.to_square)
            return (0, -((VAL[vic.piece_type] if vic else 1) * 10 - VAL[board.piece_at(m.from_square).piece_type]))
        return (1, 0)
    return sorted(board.legal_moves, key=key)


QMAX, QCHECK = -8, -2   # quiescence: deepest qsearch ply; extend checks while above this


def q_moves(board, allow_checks):
    caps = [m for m in board.legal_moves if board.is_capture(m)]
    caps.sort(key=lambda m: -((VAL[board.piece_at(m.to_square).piece_type] if board.piece_at(m.to_square) else 1) * 10
                              - VAL[board.piece_at(m.from_square).piece_type]))
    if allow_checks:
        caps += [m for m in board.legal_moves if not board.is_capture(m) and board.gives_check(m)]
    return caps


def alpha_beta(fen, depth, budget, ev):
    """Negamax alpha-beta over the SF-NNUE static value with a capture+check quiescence
    at the horizon — builds the candidate tree structure (fast). All "better"/blunder
    judgements are made later by an SF-N search judge, NOT by these static values.
    Returns (nodes_dict, root_id)."""
    nodes, ctr, board = {}, [0], chess.Board(fen)

    def rec(depth_left, alpha, beta, parent, move_uci, ply):
        nid = ctr[0]; ctr[0] += 1
        n = {"id": nid, "parent": parent, "move": move_uci, "ply": ply,
             "stm": "w" if board.turn == chess.WHITE else "b", "rdepth": depth_left,
             "flag": "EXACT", "leaf": False, "value": 0, "seval": 0, "fen": board.fen()}
        nodes[nid] = n
        if board.is_checkmate():
            v = -(MATE - ply); n.update(leaf=True, value=v, seval=v); return v
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            n.update(leaf=True, value=0, seval=0); return 0
        in_qs = depth_left <= 0
        seval = ev.eval(board.fen()); n["seval"] = seval
        orig_alpha = alpha
        if not in_qs:
            moves, best = ordered(board), -MATE * 2
        elif board.is_check():
            moves, best = ordered(board), -MATE * 2
        else:
            best = seval
            if best >= beta or depth_left <= QMAX:
                n.update(leaf=True, value=best); return best
            if best > alpha:
                alpha = best
            moves = q_moves(board, depth_left > QCHECK)
            if not moves:
                n.update(leaf=True, value=best); return best
        if ctr[0] > budget:
            n.update(leaf=True, value=seval); return seval
        for m in moves:
            board.push(m)
            v = -rec(depth_left - 1, -beta, -alpha, nid, m.uci(), ply + 1)
            board.pop()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        n["value"] = best
        n["flag"] = "LOWER" if best >= beta else ("UPPER" if best <= orig_alpha else "EXACT")
        return best

    rec(depth, -MATE * 2, MATE * 2, -1, "", 0)
    return nodes, 0


def children_of(nodes):
    children = {nid: [] for nid in nodes}
    for nid, n in nodes.items():
        if n["parent"] >= 0:
            children[n["parent"]].append(nid)
    for k in children:
        children[k].sort()   # ascending id == search order
    return children


# ============================================================ policy pruning
# The "stubborn man" breadth rule (see the original policy_tree docstring): at every
# node follow the search-best child, and additionally surface each sibling whose maia
# policy >= policy(followed), capped at `cap`. Refuted deviations end on the refuter's
# move (parity), enforced by trim_parity.

def best_child(nodes, children, nid):
    ch = children.get(nid, [])
    return min(ch, key=lambda c: nodes[c]["value"]) if ch else None


def opp(stm):
    return "w" if stm == "b" else "b"


def trim_parity(nodes, children, tree, root, end_target):
    changed = True
    while changed:
        changed = False
        for L in list(tree):
            if L == root or any(c in tree for c in children.get(L, [])):
                continue
            n = nodes[L]
            if n["leaf"] and n["rdepth"] > 0:
                continue
            if n["stm"] != end_target.get(L, n["stm"]):
                tree.discard(L); changed = True


def policy_prune(nodes, children, root, pol_obj, cap=2, max_nodes=40,
                  collapse_refuted=True, breadth_max_ply=99, min_nodes=0, cover=0.0,
                  cover_deep=None):
    tree = {root}
    expanded, end_target = set(), {}
    root_ply = nodes[root]["ply"]

    def expand(nid, end_stm, deviated, single):
        if nid in expanded or len(tree) >= max_nodes:
            return
        expanded.add(nid); end_target[nid] = end_stm
        ch = children.get(nid, [])
        if not ch:
            return
        followed = best_child(nodes, children, nid)
        if single or nodes[nid]["ply"] - root_ply >= breadth_max_ply:
            extras = []
        else:
            pol = pol_obj.policy(nodes[nid]["fen"])
            fmove = nodes[followed]["move"]
            thr = pol.get(fmove, 0.0)
            ranked = sorted((c for c in ch if c != followed),
                            key=lambda c: -pol.get(nodes[c]["move"], 0.0))
            # (a) stubborn man: surface maia moves it would play at least as readily as the search-best
            extras = [c for c in ranked if pol.get(nodes[c]["move"], 0.0) >= thr][:cap]
            # (b) then top up with maia's favourites until the surfaced set covers `cv` of its mass
            # (cover at the root, cover_deep at every other node — taper breadth with depth).
            cv = cover if nid == root else (cover if cover_deep is None else cover_deep)
            if cv > 0.0:
                covered = pol.get(fmove, 0.0) + sum(pol.get(nodes[c]["move"], 0.0) for c in extras)
                for c in ranked:
                    if covered >= cv:
                        break
                    if c not in extras:
                        extras.append(c); covered += pol.get(nodes[c]["move"], 0.0)
        tree.add(followed)
        expand(followed, end_stm, deviated, single)
        for c in extras:
            tree.add(c)
            expand(c, end_stm if deviated else opp(nodes[c]["stm"]), True, collapse_refuted)

    expand(root, opp(nodes[root]["stm"]), False, False)
    trim_parity(nodes, children, tree, root, end_target)

    if len(tree) < min_nodes:
        pol = pol_obj.policy(nodes[root]["fen"])
        followed = best_child(nodes, children, root)
        cand = sorted((c for c in children.get(root, []) if c not in tree and c != followed),
                      key=lambda c: -pol.get(nodes[c]["move"], 0.0))
        for c in cand:
            if len(tree) >= min_nodes:
                break
            tree.add(c); expand(c, opp(nodes[c]["stm"]), True, collapse_refuted)
            trim_parity(nodes, children, tree, root, end_target)

    pv, nid = [], root
    while True:
        kids = [c for c in children.get(nid, []) if c in tree]
        if not kids:
            break
        nid = min(kids, key=lambda c: nodes[c]["value"]); pv.append(nid)
    return tree, pv


# ============================================================ assemble Tree

def win_rate(cp):
    """Lichess win% (0..100) for the side to move, from centipawns (mates clamp to 0/100)."""
    cp = max(-1500, min(1500, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def single_line(t):
    """True if t's subtree is a single unbranching line ending in a leaf — a forced sequence,
    so basically a leaf (a leaf, its lone forced recapture, that reply's lone reply, ...)."""
    while t.our_children:
        if len(t.our_children) != 1:
            return False
        t = t.our_children[0]
    return True


def min_leaf_depth(t):
    """Plies from t down to its shallowest leaf (0 if t is itself a leaf)."""
    if not t.our_children:
        return 0
    return 1 + min(min_leaf_depth(c) for c in t.our_children)


def classify(T, root, judge, mistake_wr_drop):
    """Judge with a few-node Stockfish search (NOT the tactically-blind static eval), then,
    bottom-up:
      * anti-blunder: at a node whose kept children ALL lose >= mistake_wr_drop win% vs what
        the judge says is achievable, splice in the judge's own best move as a leaf — the
        real, non-blundering continuation — so the tree never presents a forced blunder.
      * surviving_leaf(N) = the best-play leaf of N's subtree by judge eval (mover POV).
      * a losing leaf is beaten at W (shallowest ancestor where it stops surviving) by
        surviving_leaf(W); >= mistake_wr_drop win% loss there => `refuter`, else `superseded`.
    One leaf — surviving_leaf(root) — is unbeaten (the best line).
    """
    r = T[root]

    def lv(leaf, stm):                            # a leaf's JUDGE eval from a fixed side's POV
        return leaf.judge if leaf.stm == stm else -leaf.judge

    next_id = [max(T) + 1]                          # ids for synthesized nodes

    def noisy(fen, uci):                           # move is a capture or a check
        bb = chess.Board(fen); m = chess.Move.from_uci(uci)
        return bb.is_capture(m) or bb.gives_check(m)

    surv = {}

    def settle(leaf):
        """A variation must not end mid-tactic. While the judge's best move at the leaf is a
        capture/check, extend by up to 2 plies of the judge PV and re-check (2 preserves the
        move parity that ends a variation). Returns the settled leaf; sets surv along the chain."""
        chain = [leaf]; cur = leaf
        for _ in range(6):
            best, cp, pv = judge.search(cur.fen)
            if best is None or not noisy(cur.fen, best):
                break
            bd = chess.Board(cur.fen); prev = cur; added = []
            for mv in pv[:2]:
                m = chess.Move.from_uci(mv)
                if m not in bd.legal_moves:
                    break
                bd.push(m)
                node = Tree(id=next_id[0], parent=prev, move_uci=mv, fen=bd.fen(),
                            stm=("w" if bd.turn == chess.WHITE else "b"), value=0, seval=0,
                            flag="EXACT", search_leaf=False, rdepth=0, ply=prev.ply + 1)
                next_id[0] += 1; T[node.id] = node
                prev.our_children = [node]; prev.sf_children = [node]
                prev = node; added.append(node)
            if not added:
                break
            cur.search_leaf = False
            cur = added[-1]; chain.extend(added)
        cur.search_leaf = True
        cur.judge = judge.search(cur.fen)[1]
        for n in chain:
            surv[n.id] = cur
        return cur

    def surviving(t):
        s = surv.get(t.id)
        if s is not None:
            return s
        if not t.our_children:
            return settle(t)
        child_surv = [(c, surviving(c)) for c in t.our_children]
        best_move, best_cp, best_pv = judge.search(t.fen)   # judge's best move, eval, PV
        # A child is a blunder iff the MOVE itself loses >= mistake_wr_drop vs the judge's best
        # — judged apples-to-apples on the immediate position (NOT the deep, maybe-suboptimal
        # tree line, which would make almost every real move look like a blunder).
        def move_cp(c):
            return -judge.eval(c.fen)             # mover POV right after playing c
        if (best_move is not None
                and best_move not in {c.move_uci for c in t.our_children}
                and all(win_rate(best_cp) - win_rate(move_cp(c)) >= mistake_wr_drop for c in t.our_children)):
            # Splice in the judge's own PV, to the depth of the shallowest existing leaf (so the
            # added line reads like the other lines, not a bare stub).
            depth = min_leaf_depth(t)
            line = (best_pv[:depth] if best_pv else []) or [best_move]
            # A node whose ONLY continuation is a single forced line to a leaf — a leaf, or a node
            # "just above a leaf" (a lone forced recapture, e.g. Nxe4 Nxe4) — is basically a leaf;
            # replace that whole branch rather than dangling the blunder as a sibling of its fix.
            only = t.our_children[0] if len(t.our_children) == 1 else None
            if only is not None and single_line(only):
                for d in list(only.walk()):
                    T.pop(d.id, None); surv.pop(d.id, None)
                t.our_children = []; t.sf_children = []; child_surv = []
            bd = chess.Board(t.fen); prev = t; chain = []
            for mv in line:
                m = chess.Move.from_uci(mv)
                if m not in bd.legal_moves:
                    break
                bd.push(m)
                node = Tree(id=next_id[0], parent=prev, move_uci=mv, fen=bd.fen(),
                            stm=("w" if bd.turn == chess.WHITE else "b"), value=0, seval=0,
                            flag="EXACT", search_leaf=False, rdepth=0, ply=prev.ply + 1)
                next_id[0] += 1; T[node.id] = node
                if prev is not t:
                    prev.our_children = [node]; prev.sf_children = [node]
                chain.append(node); prev = node
            if chain:
                settled = settle(chain[-1])       # settle the spliced line too
                for cn in chain[:-1]:
                    surv[cn.id] = settled
                t.our_children.append(chain[0]); t.sf_children.append(chain[0])
                child_surv.append((chain[0], settled))
        surv[t.id] = max((sl for _, sl in child_surv), key=lambda sl: lv(sl, t.stm))
        return surv[t.id]

    surviving(r)
    pv_leaf = surv[r.id]                           # the single unbeaten leaf = the best line

    order = list(r.walk())[1:]                     # all nodes (spliced-in ones included), sans root
    for t in order:                               # leaves: find who beats them
        if t.our_children or t is pv_leaf:
            continue
        U = t                                     # highest ancestor where t is still the surviving leaf
        while U.parent is not None and surv[U.parent.id] is t:
            U = U.parent
        W = U.parent
        if W is None:
            continue
        beater = surv[W.id]
        drop = win_rate(lv(beater, W.stm)) - win_rate(lv(t, W.stm))
        if drop >= mistake_wr_drop:
            t.refuter = beater
        else:
            t.superseded = beater

    for t in order:                               # non-leaf, off-best-path nodes inherit their leaf's fate
        if not t.our_children or surv[t.id] is pv_leaf:
            continue
        sl = surv[t.id]
        t.refuter, t.superseded = sl.refuter, sl.superseded

    for nid, t in T.items():                      # invariant
        assert t.refuter is not t and t.superseded is not t, f"self-reference at {nid}"
        if surv[nid] is pv_leaf:                  # on the best (unbeaten) path -> unclassified
            assert t.refuter is None and t.superseded is None, f"best-path node {nid} classified"


TACTIC_WR_GAP = 5.0      # win% gap best-vs-second that makes a move "clearly best"
TACTIC_LINE_PLIES = 8    # judge-PV plies handed to the pattern matchers


def tag_tactics(root, judge):
    """At every tree node with a clearly-best move (>= TACTIC_WR_GAP win% over the
    judge's second line) whose move is a kept edge, run the tactical patterns on
    [setup, best, judge PV...]; a hit lands on that edge as `child.tactic` (clause
    specs, rendered at flatten time alongside the HCE prose)."""
    from datagen.tree.primitives import tactics

    def full_line(fen, best, pv):
        """Judge PVs at a small node budget are often 2-3 plies; extend by re-searching
        at the line's end (cached) so the deeper patterns see enough of the line."""
        bd, line = chess.Board(fen), []
        cur = pv if pv and pv[0] == best else [best]
        while cur and len(line) < TACTIC_LINE_PLIES:
            added = 0
            for u in cur:
                m = chess.Move.from_uci(u)
                if m not in bd.legal_moves or len(line) >= TACTIC_LINE_PLIES:
                    break
                bd.push(m); line.append(u); added += 1
            if not added or len(line) >= TACTIC_LINE_PLIES:
                break
            nb, _, npv, _ = judge.search2(bd.fen())
            if nb is None:
                break
            cur = npv if npv and npv[0] == nb else [nb]
        return line

    for t in root.walk():
        if not t.our_children:
            continue
        best, cp1, pv, cp2 = judge.search2(t.fen)
        if best is None or cp2 is None or win_rate(cp1) - win_rate(cp2) < TACTIC_WR_GAP:
            continue
        child = next((c for c in t.our_children if c.move_uci == best), None)
        if child is None:
            continue
        specs = tactics.detect(t.fen, t.move_uci, t.parent.fen if t.parent else None,
                               full_line(t.fen, best, pv))
        if specs:
            child.tactic = specs


def to_tree(nodes, children, keep, root, root_fen, pv_ids, judge, mistake_wr_drop=10.0):
    """Build the Tree graph over the kept node ids, then judge + classify."""
    T = {}
    for nid in keep:
        n = nodes[nid]
        T[nid] = Tree(id=nid, parent=None, move_uci=(n["move"] or None), fen=n["fen"],
                      stm=n["stm"], value=n["value"], seval=n["seval"], flag=n["flag"],
                      search_leaf=n["leaf"], rdepth=n["rdepth"], ply=n["ply"])
    for nid in keep:
        n = nodes[nid]
        if n["parent"] in T:
            T[nid].parent = T[n["parent"]]
    for nid in keep:
        kids = [c for c in children.get(nid, []) if c in keep]        # search (id) order
        T[nid].sf_children = [T[c] for c in kids]
        pb = chess.Board(nodes[nid]["fen"])
        T[nid].our_children = sorted(
            T[nid].sf_children,
            key=lambda t: move_priority(pb, chess.Move.from_uci(t.move_uci)))
    r = T[root]
    r.root_fen, r.root_stm, r.pv = root_fen, nodes[root]["stm"], [T[c] for c in pv_ids]
    classify(T, root, judge, mistake_wr_drop)
    tag_tactics(r, judge)
    return r


# ============================================================ persistent engines

EV = None
POL = None
JUDGE = None
JUDGE_NODES = 100


def engines(judge_nodes=1000):
    global EV, POL, JUDGE
    if EV is None:
        EV = SfEval()
    if POL is None:
        POL = MaiaPolicy()
    if JUDGE is None or JUDGE.nodes != judge_nodes:
        if JUDGE is not None:
            JUDGE.close()
        JUDGE = SfJudge(nodes=judge_nodes)
    return EV, POL, JUDGE


def get_tree(fen, depth=3, budget=60000, cap=2, mistake_wr_drop=10.0, judge_nodes=1000,
             max_nodes=100, cover=0.35, cover_deep=0.15):
    """Search `fen` and return the pruned, maia-anchored search Tree (root Tree).
    `judge_nodes`: the Stockfish node budget used to judge 'better'/blunder (1000 gives stable,
    order-independent leaf comparisons at negligible cost; 100 is too weak and TT-order-dependent).
    mistake_wr_drop: win% loss counting as a mistake -> refuted (vs merely off-best -> superseded).
    `cap`/`max_nodes`: breadth per node and total pruned-tree size target."""
    ev, pol, judge = engines(judge_nodes)
    ev.cache.clear(); pol.cache.clear()
    nodes, root = alpha_beta(fen, depth, budget, ev)
    children = children_of(nodes)
    keep, pv = policy_prune(nodes, children, root, pol, cap=cap, max_nodes=max_nodes,
                             collapse_refuted=True, min_nodes=0, cover=cover, cover_deep=cover_deep)
    return to_tree(nodes, children, keep, root, fen, pv, judge, mistake_wr_drop=mistake_wr_drop)


def close_engines():
    global EV, POL, JUDGE
    for e in (EV, POL, JUDGE):
        if e is not None:
            e.close()
    EV = POL = JUDGE = None
