"""Flatten a stage-5.sim tree into the training narrative.

HCE reading order + supersession events at the outer level; each node's plan
analysis (side to move only, live-plan repeats suppressed) indented. Two
modes:

  * mode="token"  the training format: moves as compact move tags, sides as
                  <PLAYER>/<OPPONENT> (root-relative), plan/verdict text as
                  the models' raw POV-token generations re-anchored to the
                  root mover, solid conclusions in their token rendering;
  * mode="human"  the same narrative decoded for reading (absolute colors,
                  SAN) — what the lab pipeline produced.

Two optional passes, independent of the search:
  move_desc       every candidate gets the HCE eval's own description of what
                  the move does ("Consider a4 (White), which ...").
  extend_refuted  a leaf that refutes a move, or whose verdict is decisive,
                  gets a board-derived solid conclusion where one can be
                  proved (datagen.sim.solid), REPLACING the model's verdict.
"""
import threading

import chess

from datagen.sim.plan_parse import (_stem_only, advance, is_expected,
                                    plan_entries)
from datagen.sim.search import MATE, verdict_value
from datagen.tree.plan_verbalize import render_plan_block
from datagen.tree.tree import _supersession, _unsuperseded_lines
from utils.translate_helpers import Translator


_RENDER_LOCK = threading.Lock()


def flatten_model_tree(root, builder, mode="token", move_desc=False,
                       extend_refuted=False):
    """Render one tree without racing the legacy tree phrasing globals.

    The move-description primitives retain their notation, root perspective,
    and a small amount of capture context in module-global one-item lists.
    Searches may run concurrently, but their final rendering must therefore
    be serialized or one position can leak its POV/mode into another.
    """
    with _RENDER_LOCK:
        return _flatten_model_tree(root, builder, mode=mode,
                                   move_desc=move_desc,
                                   extend_refuted=extend_refuted)


def _flatten_model_tree(root, builder, mode="token", move_desc=False,
                        extend_refuted=False):
    NTB = {}

    def board_at(t):
        if t.id not in NTB:
            NTB[t.id] = chess.Board(t.fen)
        return NTB[t.id]

    token = mode == "token"

    from datagen.tree.notation import Notation
    notation = Notation(root.root_fen, mode)

    def side(stm):
        """The side word for a node stm ('w'/'b') or chess color, per mode."""
        w = stm == "w" if isinstance(stm, str) else bool(stm)
        if token:
            return "<PLAYER>" if (w == (root.root_stm == "w")) else "<OPPONENT>"
        return "White" if w else "Black"

    def mv_of(t):
        return chess.Move.from_uci(t.move_uci)

    def render_move(board, mv):
        return notation.move_tag(board, mv) if token else board.san(mv)

    def san_of(t):
        return render_move(board_at(t.parent), mv_of(t))

    def line_to(t):
        chain, x = [], t
        while x.parent is not None:
            chain.append(x)
            x = x.parent
        return " ".join(san_of(c) for c in reversed(chain))

    def _child_toward_(node, leaf):
        x = leaf
        while x.parent is not node:
            x = x.parent
        return x

    def node_text(kind, t):
        """The model generation for node t, in the narrative's mode: raw
        token text re-anchored to the root mover, or the decoded rendering."""
        if not token:
            return (builder.plans if kind == "plan" else builder.verdicts)[t.id]
        raw = builder.M.raw(kind, t.fen, builder.history(t))
        return Translator.reanchor(raw, board_at(t).turn,
                                   root.root_stm == "w")

    def verdict_of(t):
        """Leaf verdict text in the narrative's mode ('checkmate.' / 'draw.'
        stay literal)."""
        v = builder.verdicts.get(t.id, "?")
        if not token or v in ("checkmate.", "draw.", "?"):
            return v
        return node_text("verdict", t)

    mover0 = "White" if root.root_stm == "w" else "Black"
    other0 = "Black" if root.root_stm == "w" else "White"

    if move_desc:
        # elucidate() phrases relative to the root mover ("our"/"the
        # opponent's"); the rest of this narrative is absolute (human) or
        # <PLAYER>-relative (token), so pin the perspective to the root and
        # translate the possessives per mode.
        from datagen.tree import moves as _moves
        _moves.notation[0] = notation
        _moves.PERSP[0] = chess.WHITE if root.root_stm == "w" else chess.BLACK

    describe_move = None
    if move_desc:
        # the same tactical-motif pass the production narrative runs: at any
        # node with a clearly-best move, the patterns fire on [setup, best,
        # judge PV...] and the motif lands on that edge, spoken before the HCE
        # prose ("which forks ..., and gains space")
        from datagen.tree.primitives import tactics as tactics_mod
        from datagen.tree.search import SfJudge, tag_tactics
        import os
        tag_tactics(root, SfJudge(
            nodes=int(os.environ.get("MT_TACTIC_NODES", 100_000))))
        from datagen.tree.moves import elucidate as describe_move

    def absolute(eff):
        """Root-relative possessives -> the narrative's side references."""
        a, b = (("<PLAYER>", "<OPPONENT>") if token else (mover0, other0))
        for pat, rep in (("the opponent's ", f"{b}'s "), ("our ", f"{a}'s "),
                         ("the opponent ", f"{b} "), ("we ", f"{a} ")):
            eff = eff.replace(pat, rep)
        return eff

    def effect(child):
        """', which ...' — the tactical motif, if one fired on this edge, then
        the HCE eval's description of what the move does. A move that ends the
        game gets nothing: the mate IS the point, and there is no position left
        for a positional remark to be about."""
        if describe_move is None or board_at(child).is_game_over():
            return ""
        eff = describe_move(child.parent.fen, child.move_uci)
        if child.tactic:                          # motif first, as upstream
            from datagen.tree.primitives import tactics as tactics_mod
            eff = tactics_mod.render(child.tactic) + (", and " + eff if eff else "")
        return f", which {absolute(eff)}" if eff else ""

    extender = None
    if extend_refuted:
        from datagen.sim.solid import SolidExtender
        extender = SolidExtender()

    def solid_text(pair, leaf):
        """Pick the mode's rendering of a solid conclusion; the machine side
        is anchored at the leaf's side to move — re-anchor to the root."""
        human, machine = pair
        if not token:
            return human
        return Translator.reanchor(machine, board_at(leaf).turn,
                                   root.root_stm == "w")

    if not root.sf_children:
        # Nothing to search: every plan the model offered either failed to
        # parse to a legal move or was screened out as a blunder. The root's
        # own evaluation is then the whole analysis.
        return (f"Root position ({side(root.stm)} to move): {verdict_of(root)}")

    events, label_of, critical = _supersession(root)
    refuted_ids = {ev[1].id for evs in events.values() for ev in evs
                   if ev[0] == "refuted"}
    out = []

    _mated = {}

    def is_mated(t):
        """Is the side to move at `t` mated by force? (oracle, cached)"""
        if t.id not in _mated:
            _, cp, _ = builder._judge().search(t.fen)
            _mated[t.id] = cp <= -(MATE - 1000)
        return _mated[t.id]

    def entry_lines(t, e):
        """An entry's line texts in the narrative's mode."""
        if not token:
            return e["lines"]
        if "machine_lines" in e:
            return [Translator.reanchor(ln, board_at(t).turn,
                                        root.root_stm == "w")
                    for ln in e["machine_lines"]]
        return e["lines"]                # raw misaligned: decoded fallback

    def plans_lines(t, live):
        """Render node t's plan block; returns the updated live list."""
        if t.id not in builder.plans or t.id in refuted_ids:
            return live          # a move that gets refuted goes straight to it
        board = board_at(t)
        mover = side(t.stm)
        entries = builder.entries.get(t.id) or plan_entries(
            t.fen, builder.plans[t.id],
            raw_text=builder.M.raw("plan", t.fen, builder.history(t)))
        if is_mated(t):
            # this side is getting mated: it never reaches any move the tree
            # does not actually play, so the plans shrink to the explored moves
            # and stop at the first of them
            kids = {c.move_uci for c in t.sf_children}
            entries = [{**e,
                        "lines": [_stem_only(e["lines"][0])],
                        **({"machine_lines": [_stem_only(e["machine_lines"][0])]}
                           if "machine_lines" in e else {}),
                        "seqs": [[st[:1], p] for st, p in e["seqs"]]}
                       for e in entries
                       if e["first"] is not None and e["first"].uci() in kids]
        held, fresh = [], []
        mover_word = "White" if t.stm == "w" else "Black"
        for e in entries:
            if e["first"] is not None and is_expected(live, board,
                                                      e["first"], mover_word):
                r = render_move(board, e["first"])
                if r not in held:          # two plans can open with one move
                    held.append(r)
            else:
                fresh.append(e)
        if not held and not fresh:
            return live
        # the header continues the line that introduced this node
        out[-1] += f" Plans for {mover}:"
        out.extend(render_plan_block(fresh, held, mover,
                                     mode=("token" if token else "human"),
                                     lines_of=lambda e: entry_lines(t, e)))
        return live + fresh

    out.append(f"Root position ({side(root.stm)} to move).")
    live0 = plans_lines(root, [])

    solids = {}              # leaf.id -> board-derived conclusion, when found
    refuted = set()          # ids of refuted nodes: their remaining subtrees
                             # are not verbalized once the refutation is shown

    def _under_refuted(c):
        x = c
        while x is not None:
            if x.id in refuted:
                return True
            x = x.parent
        return False

    def walk(t, live):
        for i, c in enumerate(t.our_children):
            if _under_refuted(c):
                continue
            san = san_of(c)
            mover = side(t.stm)
            eff = effect(c)
            if i == 0:                      # continuing the line we are in
                out.append(f"Consider {san} ({mover}){eff}.")
            else:                           # backtracking: the only blank line
                out.append("")
                if c.parent is root:
                    out.append(f"Back at the root, consider "
                               f"{san} ({mover}){eff}.")
                else:
                    out.append(f"Back after {line_to(c.parent)}, "
                               f"consider {san} ({mover}){eff}.")
            mover_word = "White" if t.stm == "w" else "Black"
            live_c = advance(live, board_at(t), mv_of(c), mover_word)
            if c.our_children:
                live_c = plans_lines(c, live_c)
                walk(c, live_c)
            else:
                lab = (f" This is line [{label_of[c.id]}]."
                       if c.id in label_of else "")
                verdict = builder.verdicts.get(c.id, "?")
                refs = [ev[1] for ev in events.get(c.id, ())
                        if ev[0] == "refuted"]
                # a refuted move needs to be told what it cost; a leaf the
                # model calls much better or above needs to be told WHY, and
                # that reason is usually a few moves away -- the queen falls,
                # or it is simply mate. Below that band the position is what it
                # looks like and the model's reason stands.
                solid = None
                decisive = abs(verdict_value(board_at(c), verdict)) >= 200
                if extender is not None and (refs or decisive):
                    if refs:
                        # the shallowest refuted move is the one being
                        # explained, so its starting position is the baseline
                        base, new = min(refs, key=lambda n: n.ply).parent.fen, True
                    else:
                        base, new = c.fen, False
                    got = extender.conclusion(c.fen, c.parent.fen, c.move_uci,
                                              base, new)
                    solid = solid_text(got[0], c) if got else None
                # a board-derived conclusion REPLACES the model's verdict
                if solid:
                    solids[c.id] = solid
                out.append(f"Here ({line_to(c)}): {solid or verdict_of(c)}{lab}")
                for ev in events.get(c.id, ()):
                    if ev[0] == "refuted":
                        bn = ev[1]
                        out.append(f"This refutes {side(bn.parent.stm)}'s "
                                   f"{san_of(bn)} (line {line_to(bn)}).")
                        refuted.add(bn.id)
                    else:
                        _, A, B, L, rejected = ev
                        la = (f"line [{label_of[A.id]}]"
                              if A.id in label_of else "that line")
                        lb = (f"line [{label_of[B.id]}]"
                              if B.id in label_of else "this line")
                        lead = (f"Even {la} (rejected)" if rejected
                                else la.capitalize())
                        out.append(f"{lead} supersedes {lb}: at "
                                   f"{line_to(L) or 'the root'} {side(L.stm)} "
                                   f"prefers {san_of(_child_toward_(L, A))}.")

    walk(root, live0)
    out.append("")
    survivors = _unsuperseded_lines(events, label_of)
    if len(survivors) == 1:
        out.append(f"Line [{survivors[0][1]}] is un-superseded, and hence "
                   "the best line.")
    elif survivors:
        labels = ", ".join(f"[{label}]" for _, label in survivors)
        out.append(f"Lines {labels} remain un-superseded, so no sole best "
                   "line is proven.")
    out.append(f"Critical line: {line_to(critical)} — "
               f"{solids.get(critical.id) or verdict_of(critical)}")
    return "\n".join(out)
