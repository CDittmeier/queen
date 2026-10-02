"""The search Tree and its flattening to our narrative format.

`Tree` doubles as a search node: it holds the move that led to it, the search
verdict (value/seval/flag/leaf), and its children in TWO orderings —

  * sf_children  : the order the alpha-beta search considered them (search order)
  * our_children : our reading order, checks > captures > threats > others (CCT)

A pruned-tree leaf (no `our_children`) that ends a refuted line points at the move
that refuted it via `refuter`.

`Tree.string(mode)` flattens the whole tree into the search narrative (mode="human"
= English + SAN, mode="token" = chess-LM tokens). This is the DFS *scaffold* — go
down a line, back up (naming the line from the root), mark refutations, conclude.
The verbalization primitives it calls now live beside it:
  * primitives/<factor>.py — one file per HCE factor (material, king, threats, pawns,
    …); each Primitive exposes score() AND verbalize()/effects() for that factor.
  * primitives/common.py — the shared verbalize layer (naming, pick/join, dispatch).
  * moves.py — cross-factor ordering/combining (verbose_effects, move_priority).
  * glue.py  — the connectors (back up to a line, refute, PV/hedge, conclusion).
  * notation.py — the mode-aware renderer.
Everything is vendored under datagen/tree — nothing is imported from lab/.
"""
from __future__ import annotations

import hashlib
import math
import re

import chess

# --- verbalization primitives (all local to datagen/tree) ---
from datagen.tree import moves as moves                 # move-to-move change phrasers + CCT ordering
from datagen.tree import glue as glue                  # connectors (back-up / refute / pv / conclusion)
from datagen.tree.notation import Notation            # vendored renderer (datagen/tree/notation.py)
from datagen.tree.primitives import tactics as tactics  # tactical-motif clause rendering


class Tree:
    """A search node that is also the subtree rooted at it."""

    __slots__ = ("id", "parent", "move_uci", "fen", "stm", "value", "seval",
                 "flag", "search_leaf", "rdepth", "ply", "sf_children",
                 "our_children", "refuter", "superseded", "judge", "tactic",
                 "root_fen", "root_stm", "pv")

    def __init__(self, *, id, parent, move_uci, fen, stm, value, seval, flag,
                 search_leaf, rdepth, ply):
        self.id = id
        self.parent = parent          # Tree | None (None at root)
        self.move_uci = move_uci      # move that led here (None at root)
        self.fen = fen                # position AT this node
        self.stm = stm                # side to move here ("w"/"b")
        self.value = value            # negamax value at this node
        self.seval = seval            # static eval at this node
        self.flag = flag              # EXACT / LOWER (fail-high) / UPPER (fail-low)
        self.search_leaf = search_leaf  # terminal or horizon leaf in the SEARCH
        self.rdepth = rdepth
        self.ply = ply
        self.sf_children: list[Tree] = []   # search order
        self.our_children: list[Tree] = []  # CCT order (checks>captures>threats>other)
        # Two distinct notions for a non-PV node (see search.py::classify):
        #  refuter    : the move INTO this node is itself a mistake -> its best child
        #               punishes it (self-contained refutation).
        #  superseded : the move is fine but the branch is off-PV -> the alpha-beta
        #               cutoff node that pruned it (in the closest suboptimal ancestor).
        self.refuter: Tree | None = None
        self.superseded: Tree | None = None
        self.judge = None                   # SF-N eval (cp, this node's stm POV); set on leaves
        self.tactic = None                  # fired tactical-motif clause specs on the move INTO
                                            # this node (see search.py::tag_tactics), else None
        # root-only:
        self.root_fen = None
        self.root_stm = None
        self.pv: list[Tree] = []            # principal variation (list of Tree)

    # -- convenience --------------------------------------------------------
    @property
    def is_leaf(self) -> bool:              # leaf of the PRUNED tree (what we narrate)
        return not self.our_children

    @property
    def mover(self) -> str | None:          # side that played move_uci (parent's stm)
        return self.parent.stm if self.parent else None

    def walk(self):
        yield self
        for c in self.our_children:
            yield from c.walk()

    def line_san(self) -> str:
        """SAN move-line from the root to this node ('1. O-O c6 2. b4')."""
        chain, x = [], self
        while x.parent is not None:
            chain.append(x); x = x.parent
        chain.reverse()
        board = chess.Board(x.root_fen)
        return board.variation_san([chess.Move.from_uci(c.move_uci) for c in chain])

    def render(self) -> str:
        """Indented move view (search order), values in centipawns."""
        out = []

        def cp(v):
            return f"{v * 100 / 213:+.0f}"

        def go(t, depth):
            if t.parent is None:
                out.append(f"(root) {t.stm} to move  [{cp(t.value)}]")
            else:
                pb = chess.Board(t.parent.fen)
                san = pb.san(chess.Move.from_uci(t.move_uci))
                tag = "leaf" if t.is_leaf else t.flag.lower()
                mark = ("  [refuted]" if t.refuter else
                        "  [superseded]" if t.superseded else
                        "  <<best")                    # off-best-path nodes carry a pointer
                out.append("  " * depth + f"{san:<7} [{cp(t.value)}] ({tag}){mark}")
            for c in t.sf_children:
                go(c, depth + 1)

        go(self, 0)
        return "\n".join(out)

    def supersession_report(self) -> list[str]:
        """One line per non-PV node: its line and the LEAF that refuted / superseded it,
        with whether that leaf is earlier or later in serialization (which the coming
        verbalization uses). First line is the single unbeaten leaf (the best). For
        printing after the tree (NOT part of string())."""
        order = list(self.walk())                    # CCT-DFS preorder (root first)
        sidx = {t.id: i for i, t in enumerate(order)}
        rows = []
        best = next((t for t in order if t.is_leaf and t.refuter is None and t.superseded is None), None)
        if best is not None:
            rows.append(f"BEST (unbeaten):   {best.line_san()}")
        for t in order:
            if t.parent is None:
                continue
            ptr = t.refuter if t.refuter is not None else t.superseded
            if ptr is None:                          # on the best (unbeaten) path
                continue
            kind = "refuted" if t.refuter is not None else "superseded"
            rel = "earlier" if sidx.get(ptr.id, 1 << 30) < sidx.get(t.id, 0) else "later"
            rows.append(f"{t.line_san()}   —{kind} ({rel})→   {ptr.line_san()}")
        return rows

    # -- flatten to narrative ----------------------------------------------
    def string(self, mode: str = "human") -> str:
        """Flatten the whole tree into the search narrative. Call on the root."""
        return flatten(self, mode)


def _win_rate(cp):
    cp = max(-1500, min(1500, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def _child_toward(node, leaf):
    """The child of `node` on the path down to descendant `leaf`."""
    x = leaf
    while x.parent is not node:
        x = x.parent
    return x


def _supersession(root, mistake_wr_drop=10.0, info=None):
    """Supersession by pairwise proof over the *alive* leaves. A leaf A supersedes a leaf B (their
    LCA is L, chooser = L.stm) iff A is strictly better for the chooser AND every OPPONENT node on
    L->A is fully explored so far — then A is a guaranteed line proving B's move at L is not optimal
    — AND every CHOOSER node on L->B is fully explored so far: a later sibling under B's branch could
    still lift B, so a shallow comparison must not be spoken while the loser's branch is still
    growing. (Opponent nodes on L->B are left unguarded on purpose — they can only lower B further;
    chooser nodes on L->A needn't be complete — an unexplored alternative can only help A.) After each
    new leaf we apply all such supersessions, deepest-LCA first (so a line is Black's proven best
    defense before it supersedes anything shallower). A big enough margin is a refutation; the
    never-superseded leaf is the critical line. Pass `info` (a dict) to receive per-node diagnostics.
    Supersession events also record whether A had already been superseded when
    it made this lower-branch comparison. Returns events, labels, critical leaf.
    """
    preorder = list(root.walk())
    idx = {n.id: i for i, n in enumerate(preorder)}
    leaves = [n for n in preorder if n is not root and not n.our_children]

    def lv(leaf, stm):                                  # a leaf's judge eval from a fixed side's POV
        j = leaf.judge if leaf.judge is not None else 0
        return j if leaf.stm == stm else -j

    mmval, msurv = {}, {}                               # judge-minimax value + survivor (node POV)
    def mm(n):
        if not n.our_children:
            mmval[n.id], msurv[n.id] = lv(n, n.stm), n; return
        for c in n.our_children:
            mm(c)
        best = max(n.our_children, key=lambda c: (-mmval[c.id], -idx[msurv[c.id].id]))
        mmval[n.id], msurv[n.id] = -mmval[best.id], msurv[best.id]
    mm(root)

    last_leaf = {}                                     # reading-order index of a node's LAST leaf
    def last(n):
        last_leaf[n.id] = idx[n.id] if not n.our_children else max(last(c) for c in n.our_children)
        return last_leaf[n.id]
    last(root)

    anc = {lf.id: [] for lf in leaves}                 # each leaf's ancestor chain (leaf..root)
    for lf in leaves:
        x = lf
        while x is not None:
            anc[lf.id].append(x); x = x.parent

    def lca(A, B):
        aset = {n.id for n in anc[A.id]}
        return next(x for x in anc[B.id] if x.id in aset)

    def blunder_on(E):                                 # shallowest move on root->E that drops the
        for N in anc[E.id][-2::-1]:                    # value >= threshold for whoever played it
            if _win_rate(mmval[N.parent.id]) - _win_rate(-mmval[N.id]) >= mistake_wr_drop:
                return N                                # the move INTO N is a blunder by N.parent.stm
        return None

    if info is not None:                               # per-node diagnostics for tree dumps
        for n in preorder:
            d = None
            if n.parent is not None:
                d = _win_rate(mmval[n.parent.id]) - _win_rate(-mmval[n.id])
            info[n.id] = {"mmval_own_pov": mmval[n.id],
                          "value_to_mover": (None if n.parent is None
                                             else -mmval[n.id]),
                          "parent_best_to_mover": (None if n.parent is None
                                                   else mmval[n.parent.id]),
                          "wr_drop": (None if d is None else round(d, 2)),
                          "is_mistake": (None if d is None
                                         else d >= mistake_wr_drop),
                          "survivor_leaf_id": msurv[n.id].id}

    died_at = {}                                       # leaf.id -> LCA where it was itself superseded
    def can_supersede(A, B, e):
        L = lca(A, B)
        if A.id in died_at:                            # A is already superseded: its value is only locally
            D = died_at[A.id]                          # valid STRICTLY BELOW where it died (not at/above it)
            if not (idx[D.id] < idx[L.id] <= last_leaf[D.id]):
                return L, False
        va, vb = lv(A, L.stm), lv(B, L.stm)
        if not (va > vb or (va == vb and idx[A.id] < idx[B.id])):   # strictly better; ties -> earlier wins
            return L, False
        N = A.parent                                   # every OPPONENT node on L->A must be fully explored
        while N is not L:
            if N.stm != L.stm and last_leaf[N.id] > e:
                return L, False
            N = N.parent
        N = B.parent                                   # chooser nodes on L->B: complete?
        while N is not L:                              # (a later sibling could still lift B)
            if N.stm == L.stm and last_leaf[N.id] > e:
                return L, False
            N = N.parent
        return L, True

    alive, dead, refuted_nodes = [], [], set()         # dead = already-superseded leaves
    events = {l.id: [] for l in leaves}
    label_of, nextlabel = {}, [1]
    for E in leaves:
        bn = blunder_on(E)                             # a blunder is a bad outcome the moment it's reached
        if bn is not None:                             # -> refute immediately (E is a losing line: never
            if bn.id not in refuted_nodes:             # labelled, never a superseder). Don't `continue`: this
                events[E.id].append(("refuted", bn))   # refutation may have COMPLETED an opponent node, so still
                refuted_nodes.add(bn.id)               # run the fixpoint to catch a now-provable supersession
        else:
            label_of[E.id] = nextlabel[0]; nextlabel[0] += 1   # EVERY real line gets a number: even after being
            alive.append(E)                                    # superseded it may still supersede a later leaf
        e = idx[E.id]
        while True:                                    # finer 'superseded' comparisons, deepest LCA first
            best = None
            for B in alive:                            # the LOSER must still be alive...
                for A in alive + dead:                 # ...but the SUPERSEDER may itself be already superseded
                    if A is B:
                        continue
                    L, ok = can_supersede(A, B, e)
                    if ok and (best is None or idx[L.id] > idx[best[2].id]):
                        best = (A, B, L)
            if best is None:
                break
            A, B, L = best
            events[E.id].append(("superseded", A, B, L, A.id in died_at))
            alive.remove(B); dead.append(B); died_at[B.id] = L
    return events, label_of, (alive[0] if alive else leaves[-1])


def _unsuperseded_lines(events, label_of):
    """Numbered ``(leaf_id, label)`` pairs never killed by supersession."""
    rejected = {ev[2].id for evs in events.values() for ev in evs
                if ev[0] == "superseded"}
    return sorted(((leaf_id, label) for leaf_id, label in label_of.items()
                   if leaf_id not in rejected), key=lambda row: row[1])


def flatten(root: "Tree", mode: str) -> str:
    root_fen, root_stm = root.root_fen, root.root_stm
    moves.notation[0] = Notation(root_fen, mode)
    moves.PERSP[0] = chess.WHITE if root_stm == "w" else chess.BLACK
    NT = moves.notation[0]

    order: list[tuple[Tree, int]] = []
    def pre(t, depth):
        for c in t.our_children:
            order.append((c, depth))
            pre(c, depth + 1)
    pre(root, 0)

    def moves_to(t):
        chain, x = [], t
        while x is not root:
            chain.append(x); x = x.parent
        chain.reverse()
        return [chess.Move.from_uci(c.move_uci) for c in chain]

    def san_of(t):
        pb = chess.Board(t.parent.fen)
        return NT.move_prose(pb, chess.Move.from_uci(t.move_uci))

    pv = root.pv
    pv_leaf = pv[-1] if pv else None
    best_san = san_of(pv[0]) if pv else None

    # reading-order supersession verdicts (see _supersession); the narrative resolves lines
    # against each other at their branch points instead of hedging "not clear".
    events, label_of, critical_leaf = _supersession(root)
    def san_at(N, dest):                                # SAN of N's child toward `dest`, from N's board
        return NT.move_prose(chess.Board(N.fen), chess.Move.from_uci(_child_toward(N, dest).move_uci))
    def side_word(stm):
        return "White" if stm == "w" else "Black"
    def subject_of(rep, node):                          # how to refer to the outcome under discussion
        return "This" if rep is node else (f"Position [{label_of[rep.id]}]" if rep.id in label_of else "That line")

    # a refuter (first search-child of a LOWER alternative) answering >= 3 tries
    refuter_count: dict = {}
    for a in root.walk():
        if a.flag == "LOWER" and a.sf_children:
            k = (a.parent, san_of(a.sf_children[0]))
            refuter_count[k] = refuter_count.get(k, 0) + 1

    paras, cur = [], []
    prev, last_ret = root, None
    hedge_count, pv_found = 0, False
    seen_ref, first_eff = {}, {}
    for idx, (node, depth) in enumerate(order):
        parent = node.parent
        mover = parent.stm
        is_our = mover == root_stm
        san = san_of(node)
        mv = chess.Move.from_uci(node.move_uci)
        # recapture context: did the move INTO parent capture on some square (-> this move
        # taking there is a "recapture"); is this move's SOLE reply a recapture of it?
        prev_cap_sq = None
        if parent.parent is not None and parent.move_uci:
            pm = chess.Move.from_uci(parent.move_uci)
            if chess.Board(parent.parent.fen).is_capture(pm):
                prev_cap_sq = pm.to_square
        sole_recap = (len(node.our_children) == 1
                      and chess.Board(node.fen).is_capture(chess.Move.from_uci(node.our_children[0].move_uci))
                      and chess.Move.from_uci(node.our_children[0].move_uci).to_square == mv.to_square)
        eff = moves.elucidate(parent.fen, mv, prev_capture_sq=prev_cap_sq, sole_recapture=sole_recap)
        if node.tactic:                              # a clearly-best tactical shot: motif first
            eff = tactics.render(node.tactic) + (", and " + eff if eff else "")

        compressed, rkey, diff = False, None, ""
        if parent.flag == "LOWER":
            rkey = (parent.parent, san)
            if refuter_count.get(rkey, 0) >= 3:
                seen_ref[rkey] = seen_ref.get(rkey, 0) + 1
                emap = moves.effect_map(parent.fen, mv)
                if seen_ref[rkey] == 1:
                    first_eff[rkey] = emap
                else:
                    compressed = True
                    diff = moves.diff_clause(emap, first_eff.get(rkey, {}))

        jump = parent is not prev and idx > 0
        if jump:
            if cur:
                paras.append(" ".join(cur)); cur = []
            if parent is root:
                ret = glue.back_to_root(node.id)
            elif parent.stm != root_stm:
                where = glue.line_notation(root_fen, moves_to(parent))
                ret = glue.opp_another(node.id, where)
            elif parent is last_ret:
                ret = glue.another_try(node.id)
            else:
                where = glue.line_notation(root_fen, moves_to(parent))
                ret = glue.back_to(node.id, where)
            cur.append(ret)
            last_ret = parent

        if compressed:
            cur.append(glue.compressed_refute(node.id, san, diff))
            prev = node
            continue

        if jump or idx == 0:
            cur.append(glue.consider(node.id, san, eff))
        elif is_our:
            cur.append(glue.our_move(node.id, san, eff))
        else:
            cur.append(glue.opp_move(node.id, san, eff))

        if node.is_leaf:
            # supersession verdicts (some "open up" as a later line completes an opponent's options).
            for ev in events[node.id]:
                if ev[0] == "refuted":                   # the move into `bn` drops the eval — but into a bad
                    bn = ev[1]; P = bn.parent            # position, or merely a worse-than-best one?
                    ee = node.judge if node.judge is not None else 0
                    ee = ee if node.stm == P.stm else -ee   # end eval from the blunderer's (P.stm) POV
                    line = glue.line_notation(root_fen, moves_to(node))
                    blunder, prefix = san_at(P, node), glue.line_notation(root_fen, moves_to(P))
                    if ee < -100:                        # < -1 for the mover: genuinely bad -> a real refutation
                        advice = "should not" if P.stm == root_stm else "will not"
                        cur.append(glue.refutes(node.id, "This", side_word(P.stm), line, blunder, prefix, advice))
                    else:                                # still fine for the mover, just not best
                        cur.append(glue.suboptimal(node.id, "This", side_word(P.stm), line, blunder, prefix))
                    continue
                _, A, B, L, rejected = ev                # A supersedes B at branch L (chooser = L.stm)
                side = side_word(L.stm)
                L_line = glue.line_notation(root_fen, moves_to(L))
                if B is node:                            # this new line is the one that loses
                    cur.append(glue.worse_than(node.id, "This", label_of.get(A.id), side,
                                               san_at(L, A), san_at(L, B), L_line,
                                               better_rejected=rejected))
                else:                                    # A (this line, or an earlier one) proves superior to B
                    subj = "this position" if A is node else f"position [{label_of.get(A.id)}]"
                    cur.append(glue.supersedes(node.id, subj, label_of.get(B.id), side,
                                               san_at(L, A), san_at(L, B), L_line,
                                               rejected=rejected))
            if node.id in label_of:                      # a surviving position, worth referencing later
                cur.append(glue.name_position(node.id, label_of[node.id]))
        prev = node

    if cur:
        paras.append(" ".join(cur))
    survivors = _unsuperseded_lines(events, label_of)
    if len(survivors) == 1:
        paras.append(glue.unsuperseded(survivors[0][1]))
    elif survivors:
        paras.append(glue.multiple_unsuperseded([label for _, label in survivors]))
    crit_line = moves_to(critical_leaf)
    if crit_line:
        first_san = NT.move_prose(chess.Board(root_fen), crit_line[0])
        paras.append(glue.critical(glue.line_notation(root_fen, crit_line), first_san,
                                   label_of.get(critical_leaf.id)))

    header = "It is our move." if mode == "token" else (("White" if root_stm == "w" else "Black") + " to move.")
    body = "\n\n".join(paras)
    if mode == "token":
        body = NT.tok_free(body, default_color=(chess.WHITE if root_stm == "w" else chess.BLACK))
    body = re.sub(r"\s*→\s*", " to ", body)
    return f"{header}\n\n" + body
