"""base.py — the Factor base for every HCE factor (material.py, king.py, ...).

Each factor subclass exposes exactly two things:
  * score(board, ctx=None) -> (total Score, {sub_factor: Score})
      the term's total plus its per-sub-factor breakdown (both sides netted White-POV). eval_full
      weighs these into cp; the verbalization ranks/signs sub-factors by their cp deltas.
  * describe(before, after, move, mover, before_eval, after_eval) -> {sub_factor: description-or-None}
      how the factor's contribution changed across a move.

KIND ("container" | "whole" | "material") tells moves.elucidate how to pick the term's clause;
run_phrasers()/whole_direction() are describe() helpers shared by all factors.
"""
from __future__ import annotations

import chess


class Factor:
    tag = "factor"
    KIND = "container"
    # position-activity features this factor owns (see primitives/activity.py)
    ACTIVITY: tuple = ()

    def is_active_position(self, board, ctx=None):
        """{feature: felt} for this factor's ACTIVITY features whose |felt|
        clears its frozen corpus percentile on this board (activity.py)."""
        from datagen.tree.primitives import activity
        act = activity.is_active_position(board, ctx)
        return {n: v for n, v in act.items() if n in self.ACTIVITY}

    def is_active_move(self, mover, before_eval, after_eval, pct=90, sub=None,
                       fallback=None):
        """Did the move shift this factor's term (or `sub`-factor) by at least
        its corpus `pct` percentile of per-ply |delta cp|? `before_eval` /
        `after_eval` are toplevel.eval_full snapshots; a channel absent from
        the table falls back to `fallback` cp (None = inactive)."""
        from datagen.tree.primitives.activity import move_bar
        sign = 1 if mover == "w" else -1
        if sub is None:
            channel, b, a = self.tag, before_eval[self.tag][0], after_eval[self.tag][0]
        else:
            channel = f"{self.tag}/{sub}"
            b = before_eval[self.tag][1].get(sub, 0.0)
            a = after_eval[self.tag][1].get(sub, 0.0)
        bar = move_bar(channel, pct, fallback)
        return bar is not None and abs(sign * (a - b)) >= bar

    def run_phrasers(self, phrasers, before, after, mover, before_eval, after_eval):
        """Apply a container factor's {sub_factor: (phraser | None, elucidator)} table, signing
        each sub-factor by the mover-POV direction of its cp change -> {sub_factor: desc-or-None}."""
        from .common import mech_fallback
        us = chess.WHITE if mover == "w" else chess.BLACK
        sign = 1 if mover == "w" else -1
        before_sub, after_sub = before_eval[self.tag][1], after_eval[self.tag][1]
        out = {}
        for sub, (phraser, elucidator) in phrasers.items():
            x = sign * (after_sub.get(sub, 0.0) - before_sub.get(sub, 0.0))
            sx = 1 if x >= 0 else -1
            c = phraser(before, after, us, sx) if phraser else mech_fallback(sub, elucidator, before, after, us, sx)
            out[sub] = c or None
        return out

    def whole_direction(self, mover, before_eval, after_eval):
        """+1/-1 — the mover-POV direction of this whole term's cp change."""
        sign = 1 if mover == "w" else -1
        return 1 if sign * (after_eval[self.tag][0] - before_eval[self.tag][0]) >= 0 else -1
