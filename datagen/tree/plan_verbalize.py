"""Plan-block verbalization for search narratives.

Renders one node's plan list as numbered points under a "Plans for {side}:"
header: plans whose next move a live plan already stated collapse to a
"0. ... as above." point; fresh plans are numbered, an "After <premise>, ..."
stem whose premise extends an earlier point's premise is rewritten as
"In point N, after <the new part>, ..."; each point may carry a
percentile-gated description of its opening move. Mode-agnostic: the caller
supplies the already-selected line texts (human or POV-token), the side word,
the move renderer, and the effect function.
"""
import re

# the "After <premise>, <mover> can <rest>" stem, per mode
_CHAIN_RE = {
    "human": re.compile(r"(\s*)- After (.+?), (White|Black) can (.*)$"),
    "token": re.compile(r"(\s*)- After (.+?), (<PLAYER>|<OPPONENT>) can (.*)$"),
}


def render_plan_block(entries, held_moves, mover_word, mode="human",
                      lines_of=None, effect_of=None):
    """The numbered plan block for one node.

    entries      the FRESH plan entries (not already-live ones), in order;
    held_moves   rendered moves of plans a live plan already stated;
    mover_word   the side word in the target mode ("White" / "<PLAYER>" ...);
    lines_of     entry -> its line texts in the target mode;
    effect_of    entry -> a description of its opening move ("" for none).

    Returns the block's lines (without the "Plans for ...:" header, which the
    caller appends to its own narrative line)."""
    lines_of = lines_of or (lambda e: e["lines"])
    effect_of = effect_of or (lambda e: "")
    out = []
    if held_moves:
        out.append(f"    0. {mover_word} can consider "
                   f"{', '.join(held_moves)} as above.")
    chain_re = _CHAIN_RE[mode]
    chains = []          # per printed point: its "After ..." move-clause list
    for k, e in enumerate(entries, 1):
        lines = lines_of(e)
        stem = lines[0]
        m = chain_re.match(stem)
        rewritten, chain = None, None
        if m:
            chain = m.group(2).split(" and ")
            best, bi = 0, None
            for pi, pc in enumerate(chains):
                if not pc:
                    continue
                c = 0
                while (c < len(pc) and c < len(chain) and pc[c] == chain[c]):
                    c += 1
                if c > best:
                    best, bi = c, pi
            if bi is not None and best >= 1:
                extra = ("" if best == len(chain)
                         else " and " + " and ".join(chain[best:]))
                rewritten = (f"{m.group(1)}{k}. In point {bi + 1}, after "
                             f"{chain[best - 1]}{extra}, "
                             f"{m.group(3)} can {m.group(4)}")
        if rewritten is None:
            rewritten = re.sub(r"^(\s*)- ", rf"\g<1>{k}. ", stem)
        eff = effect_of(e)
        if eff:
            rewritten = rewritten.rstrip(". ") + f" — {eff}."
        out.append("    " + rewritten)
        for ln in lines[1:]:
            out.append("    " + ln)
        chains.append(chain)
    return out
