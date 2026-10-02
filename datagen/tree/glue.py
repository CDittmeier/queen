"""datagen/tree/glue.py — connector verbalization primitives.

The connective tissue of the search narrative: how we move between lines (back up to the
start / to a prior position / to the opponent's other tries), how we frame a candidate,
how we mark a leaf (refuted / a promising PV line / mate / unclear), and how we conclude.
The move descriptions themselves come from moves.py; this module owns the *glue* phrasing
and the line-notation helper.

Deterministic template choice (pick) and the injected Notation renderer (notation) are shared
with moves.py. Every function keeps the exact template key + option list of the original
inline phrasing, so the narrative is byte-identical.
"""
from __future__ import annotations

from datagen.tree.primitives.common import pick, notation


def line_notation(root_fen, moves):
    """'1. Qh4+ Ke5 2. Qg5+' for a list of chess.Move from the root (move numbers from 1 in
    BOTH modes, for human/token consistency). Human = SAN, token = compact move tags."""
    if not moves:
        return ""
    return notation[0].line(root_fen, moves)


# -- backing up between lines ------------------------------------------------
def back_to_root(nid):
    return pick((nid, "r"), ["Let me go back to the starting position.",
                              "Returning to the initial position."])


def opp_another(nid, where):
    return pick((nid, "opp"), [f"Let me see if the opponent has another response after {where}.",
                                f"Does the opponent have another try after {where}?"])


def another_try(nid):
    return pick((nid, "a"), ["Another try from there:", "Also worth a look there:"])


def back_to(nid, where):
    return pick((nid, where), [f"Let me return to the position after {where}.",
                                f"Backing up to the position after {where}."])


# -- move clauses ------------------------------------------------------------
def consider(nid, san, eff):
    frame = pick((nid, "c"), ["Let's consider {m}.", "What about {m}?",
                               "Consider {m}.", "Let's look at {m}."]).format(m=san)
    return frame + (f" It {eff}." if eff else "")


def our_move(nid, san, eff):
    return pick((nid, "o"), ["Then {m}", "We continue with {m}", "Now {m}"]).format(m=san) \
        + (f", which {eff}." if eff else ".")


def opp_move(nid, san, eff):
    return pick((nid, "p"), ["The opponent replies {m}", "The opponent meets this with {m}",
                              "In reply, {m}"]).format(m=san) + (f", which {eff}." if eff else ".")


def compressed_refute(nid, san, diff):
    return pick((nid, "cr"), ["This is once again refuted by {m}", "Again, {m} refutes it",
                               "Once more, {m} is the reply", "{m} refutes this one too"]).format(m=san) \
        + diff + "."


# -- leaf verdicts -----------------------------------------------------------
def mate_pv(nid):
    return pick((nid, "pm"), ["This forces mate — winning.",
                               "That's mate — winning for us."])


def mate_against(nid):
    return pick((nid, "xm"), ["This walks right into mate.",
                               "And we mate — that fails for the opponent."])


def enough(nid):
    return pick((nid, "last"), ["I've seen enough lines now — let me settle on the best move.",
                                 "That's enough calculation; time to choose the best move.",
                                 "I've checked the candidates — let me pick the best."])


def pv_hedge(nid, opp_next):
    if opp_next:
        return pick((nid, "hpo"), ["This looks good for us.", "I'm happy with this for us."])
    return pick((nid, "h"), ["This looks good for us — let me check if there is anything better.",
                              "I like this for us; still, let me see whether something is stronger.",
                              "This is promising, but let me make sure there is nothing better."])


def refuted(nid, refuted_san):
    return pick((nid, "rf"), [f"So {refuted_san} doesn't work.", f"That refutes {refuted_san}.",
                               f"So {refuted_san} falls short."])


def refuted_hedge(nid):
    return pick((nid, "rh"), ["Let me see if there is something better.",
                               "Let me look for something stronger.",
                               "I'll keep looking for something better."])


def unclear(nid):
    return pick((nid, "q"), ["This is not clear — let me look for something better.",
                              "I'm not sure about this; let me keep looking."])


def candidate(nid, best_san):
    return pick((nid, "cf"), [f"So far {best_san} is the most convincing — I'll keep it as my main candidate.",
                               f"{best_san} still looks like my best option so far.",
                               f"Of these, {best_san} is the one I trust most so far."])


def no_candidate(nid):
    return pick((nid, "cu"), ["I haven't found anything fully convincing yet — let me keep looking.",
                               "Nothing has stood out decisively so far; let me search further."])


def conclusion(best_san):
    return f"So **{best_san}** is best."


# -- supersession verdicts (reading-order line-vs-line resolution) ------------
def name_position(nid, label):
    return pick((nid, "nm"), [f"Let me call this position [{label}] for future reference.",
                              f"I'll note this position as [{label}] for later."])


def better_than(nid, subject, label, side, my_move, old_move, branch):
    where = f"on the branch {branch}, " if branch else ""
    return (f"{subject} is a better outcome for {side} than position [{label}]. This implies that "
            f"{where}{my_move} is superior to {old_move}.")


def worse_than(nid, subject, label, side, better_move, worse_move, branch,
               better_rejected=False):
    at = f" after {branch}" if branch else ""
    status = "rejected " if better_rejected else ""
    return (f"{subject} is a worse outcome for {side} than {status}position [{label}], as {better_move} "
            f"is superior to {worse_move}{at}.")


def refutes(nid, subject, side, line, blunder, prefix, advice):
    where = f" after {line}" if line else ""
    at = f" at {prefix}" if prefix else ""
    return (f"{subject} appears to be a bad outcome for {side}{where}. Therefore, {side} {advice} play "
            f"{blunder}{at}.")


def suboptimal(nid, subject, side, line, blunder, prefix):
    """A move that drops the eval but not into a bad position: worse than best, still fine."""
    where = f" after {line}" if line else ""
    at = f" at {prefix}" if prefix else ""
    return (f"{subject} appears to be a sub-optimal outcome for {side}{where}; {side} can do better "
            f"than {blunder}{at}.")


def supersedes(nid, subject, label_b, side, a_move, b_move, branch,
               rejected=False):
    at = f" after {branch}" if branch else ""
    if rejected:
        subject = f"even {subject} (rejected)"
    cap = subject[:1].upper() + subject[1:]
    return (f"I have considered all reasonable responses to {a_move}, and it appears that under best "
            f"defense, we reach {subject}. {cap} is a better outcome for {side} than position "
            f"[{label_b}], hence {a_move} is superior to {b_move}{at}.")


def unsuperseded(label):
    return f"Position [{label}] is un-superseded, and hence the best line."


def multiple_unsuperseded(labels):
    joined = ", ".join(f"[{label}]" for label in labels)
    return f"Positions {joined} remain un-superseded, so no sole best line is proven."


def critical(line, first_move, label):
    dest = f" position [{label}]" if label else " the following"
    return (f"Let me recall the best variation I have found, which leads to{dest}: it runs {line}. "
            f"Therefore, I should play **{first_move}**.")
