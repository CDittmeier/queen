"""Plan-model output -> validated plan entries (the search's candidate source).

A decoded plan generation is parsed into the side-to-move's bullets and each
bullet is validated before the search may use it:

  * blunder screen — every clause line is replayed (null moves fake the turn
    when the same side moves twice; a fake is refused out of check); an own
    recommendation that is a detectable blunder (low-node Stockfish: drops
    >= BLUNDER_CP AND lands badly) kills the line, as does a line that keeps
    going after the game is over;
  * sub-recommendation arms are screened individually, textual duplicates
    dropped, survivors re-lettered; all arms dead -> the "From there:" stub is
    repaired; a single survivor is inlined into the stem;
  * legality truncation — own moves are replayed in order (with a loose retry
    ignoring the from-square: the corpus names a piece's ORIGINAL square);
    a bullet whose first own move does not replay is dropped, one that stops
    replaying partway is cut back to its stem;
  * a bullet whose flat move sequence is a strict sublist of another's is
    dropped.

The bullet's first move is the mover's first OWN move across the whole stem —
premise included: a stem may open "after {our preparing move} ..., {mover} can
...", and the plan's real first move is the preparation (this matches the
corpus's uci field and the bullet's own live-plan seqs; the old lab code read
the consequent instead).

Live-plan tracking (`advance` / `is_expected`) lets the narrative advance a
side's plans past played moves and suppress re-statements.
"""
import re
import random
import zlib

import chess

BLUNDER_CP = 250
BLUNDER_NODES = 1000

PIECES = {"pawn", "knight", "bishop", "rook", "queen", "king"}
VERBS = {"move": "move", "moves": "move", "push": "push", "pushes": "push",
         "take": "take", "takes": "take", "continue": "continue",
         "continues": "continue", "castle": "castle", "castles": "castle"}
_SQRE = re.compile(r"^[a-h][1-8]$")


# ------------------------------------------------- prose-level parsing helpers
# (shared with the flatten's plan-block rendering and the force injector)

def compress_prose(text, default_actor):
    """Human-display compression of a decoded plan line: 'White's white X' ->
    'white X'; a piece colour matching the clause's current actor -> 'the'."""
    text = text.replace("White's white ", "white ").replace(
        "Black's black ", "black ")
    toks = re.split(r"(\W)", text)      # keep separators
    actor = default_actor
    for i, t in enumerate(toks):
        if t in ("White", "Black"):
            actor = t
        elif t in ("white", "black") and t.capitalize() == actor:
            j = i + 1                    # colour adjective of the actor's piece?
            while j < len(toks) and not toks[j].strip():
                j += 1
            if j < len(toks) and toks[j] in PIECES:
                toks[i] = "the"
    return "".join(toks)


def move_seq(text, default_actor):
    """Normalized (actor, verb, piece, squares...) tuples, in order."""
    toks = [t for t in re.split(r"[^\w']+", text) if t]
    actor, out, i = default_actor, [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("White", "Black"):
            actor = t
        elif t in VERBS:
            piece, sqs = None, []
            j = i + 1
            while j < len(toks) and toks[j] not in VERBS \
                    and toks[j] not in ("if", "after", "and", "White", "Black"):
                if toks[j] in PIECES and piece is None:
                    piece = toks[j]
                if _SQRE.match(toks[j]):
                    sqs.append(toks[j])
                j += 1
            out.append((actor, VERBS[t], piece, tuple(sqs)))
            i = j
            continue
        i += 1
    return out


def _sublist(small, big):
    n, m = len(small), len(big)
    return n < m and any(big[k:k + n] == small for k in range(m - n + 1))


def _own_tuples(text, mover):
    """(verb, piece, from?, to) own-move tuples of `mover` in one plan line."""
    out = []
    for actor, verb, piece, sqs in move_seq(text, mover):
        if actor != mover:
            continue
        if verb == "castle":
            out.append(("castle", "king", None, None))
            continue
        if verb == "take":
            to = sqs[0] if sqs else None
            frm = sqs[1] if len(sqs) > 1 else None
        else:
            to = sqs[-1] if sqs else None
            frm = sqs[0] if len(sqs) > 1 else None
        if to:
            out.append((verb, piece, frm, to))
    return out


def _current_move(board, tup):
    """Resolve one prose tuple directly on the current board.

    This deliberately does not use the normal loose from-square retry: the
    ladder fallback asks whether the move as written is legal *now*, rather
    than replaying a manoeuvre whose piece may have moved earlier in a plan.
    Ambiguous promotions prefer a queen; other ambiguity is UCI-stable.
    """
    cands = [move for move in board.legal_moves
             if _tuple_matches(board, move, tup)]
    if not cands:
        return None
    return min(cands, key=lambda move: (
        move.promotion is not None and move.promotion != chess.QUEEN,
        move.uci(),
    ))


def _root_plan_blocks(decoded, mover):
    """The side-to-move's top-level plan bullets, in generation order."""
    plans, current, in_own = [], None, False
    for line in decoded.splitlines():
        stripped = line.strip()
        if stripped.endswith("'s plans:"):
            in_own = stripped.startswith(f"{mover}'s plans:")
            current = None
            continue
        if not in_own:
            continue
        if line.lstrip().startswith("- **"):
            current = [line]
            plans.append(current)
        elif current is not None and stripped:
            current.append(line)
    return [" ".join(lines) for lines in plans]


def fallback_move(raw_text, board, rng=None):
    """Legal move salvage for a root plan generation with no search PV.

    Tiers, in order:
      1. first own move of the earliest root plan for which that move is legal;
      2. any own move in a root plan, ordered by its within-plan own-move
         ordinal and then by plan order;
      3. any mentioned move in the entire generation that is legal now;
      4. a random legal move.

    Returns ``(move, tier, details)``.  The final random choice is stable per
    FEN unless a caller supplies its own RNG.
    """
    from utils.translate_helpers import Translator

    decoded = Translator(board.turn).decode_absolute(raw_text)
    mover = "White" if board.turn else "Black"
    plans = _root_plan_blocks(decoded, mover)
    own_steps = [_own_tuples(plan, mover) for plan in plans]

    for plan_idx, steps in enumerate(own_steps):
        if not steps:
            continue
        move = _current_move(board, steps[0])
        if move is not None:
            return move, "plan_first", {
                "plan": plan_idx + 1, "move_ordinal": 1,
            }

    max_steps = max((len(steps) for steps in own_steps), default=0)
    for ordinal in range(1, max_steps + 1):
        for plan_idx, steps in enumerate(own_steps):
            if ordinal > len(steps):
                continue
            move = _current_move(board, steps[ordinal - 1])
            if move is not None:
                return move, "plan_later", {
                    "plan": plan_idx + 1, "move_ordinal": ordinal,
                }

    for mention_idx, (actor, verb, piece, sqs) in enumerate(
            move_seq(decoded, mover), 1):
        if verb == "castle":
            tup = ("castle", "king", None, None)
        elif verb == "take":
            tup = (verb, piece, sqs[1] if len(sqs) > 1 else None,
                   sqs[0] if sqs else None)
        else:
            tup = (verb, piece, sqs[0] if len(sqs) > 1 else None,
                   sqs[-1] if sqs else None)
        if tup[3] is None and verb != "castle":
            continue
        move = _current_move(board, tup)
        if move is not None:
            return move, "generation_legal", {"mention": mention_idx}

    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError("fallback requested in a position with no legal moves")
    chooser = rng or random.Random(zlib.crc32(board.fen().encode()))
    return chooser.choice(legal), "random_legal", {}


def _tuple_matches(board, mv, tup):
    """Does legal move `mv` on `board` realize the plan-step tuple?"""
    verb, piece, frm, to = tup
    if verb == "castle":
        return board.is_castling(mv)
    if to != chess.square_name(mv.to_square):
        return False
    if frm is not None and frm != chess.square_name(mv.from_square):
        return False
    if piece is not None:
        if verb == "take":
            vic = board.piece_at(mv.to_square)
            if vic is None or chess.piece_name(vic.piece_type) != piece:
                return False
        else:
            pc = board.piece_at(mv.from_square)
            if pc is None or chess.piece_name(pc.piece_type) != piece:
                return False
    return True


_MOVE_RE = re.compile(
    r"(?:move|push) (?:White's|Black's) (?:white|black) "
    r"(pawn|knight|bishop|rook|queen|king)(?: ([a-h][1-8]))? to ([a-h][1-8])")
_TAKE_RE = re.compile(
    r"take (?:the )?(?:(?:White's|Black's) )?(?:white|black) \w+ ?([a-h][1-8]) "
    r"with (?:White's|Black's) "
    r"(?:white|black) (pawn|knight|bishop|rook|queen|king)(?: ([a-h][1-8]))?")
_CASTLE_RE = re.compile(r"castles? (kingside|queenside|short|long)")
_PTYPE_RE = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
             "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}


def _legacy_first(board, stem, mover):
    """The pre-fix first move: parsed from the consequent after '{mover} can'
    (reads past an 'After ...' premise; castle mentions short-circuit).
    Kept only so a port validation can reproduce the lab trees byte-for-byte."""
    body = stem.split(":", 1)[1] if ":" in stem else stem
    can = re.search(rf"{mover} can (.*)", body)
    clause = can.group(1) if can else body
    m = _CASTLE_RE.search(clause)
    if m:
        king = board.king(board.turn)
        to = ((chess.G1 if board.turn else chess.G8)
              if m.group(1) in ("kingside", "short")
              else (chess.C1 if board.turn else chess.C8))
        mv = chess.Move(king, to)
        return mv if board.is_legal(mv) else None
    hits = []
    for m in _MOVE_RE.finditer(clause):
        hits.append((m.start(), _PTYPE_RE[m.group(1)], m.group(2), m.group(3)))
    for m in _TAKE_RE.finditer(clause):
        hits.append((m.start(), _PTYPE_RE[m.group(2)], m.group(3), m.group(1)))
    if not hits:
        return None
    _, ptype, frm, to = min(hits)
    cands = [mv for mv in board.legal_moves
             if mv.to_square == chess.parse_square(to)
             and board.piece_at(mv.from_square).piece_type == ptype
             and (frm is None or mv.from_square == chess.parse_square(frm))]
    return cands[0] if cands else None


def _first_own_move(board, stem_tuples):
    """The mover's first own move of a bullet, resolved on `board`: strict
    (piece, from, to) match, then a loose retry ignoring the from-square (the
    corpus names a piece's original square). None when nothing matches."""
    if not stem_tuples:
        return None
    tup = stem_tuples[0]
    cands = [m for m in board.legal_moves if _tuple_matches(board, m, tup)]
    if not cands and tup[0] != "castle":
        loose = (tup[0], tup[1], None, tup[3])
        cands = [m for m in board.legal_moves if _tuple_matches(board, m, loose)]
    return cands[0] if cands else None


def _clean(line, mover, human=True):
    """Drop the bold per-piece title; in human mode also compress the decode
    artifacts (token-space lines carry neither)."""
    line = re.sub(r"- \*\*.*?\*\*: ", "- ", line)
    return compress_prose(line, mover) if human else line


def _stem_only(line):
    """A bullet cut back to its first own move."""
    for sep in (". From there", "; if", ", then", ". "):
        i = line.find(sep)
        if i > 0:
            return line[:i].rstrip(",;. ") + "."
    return line


# ------------------------------------------------------------ blunder screen

_BLUNDER_JUDGE = None


def _get_blunder_judge():
    """The shared SfJudge facade used for primary-eval-only plan checks."""
    global _BLUNDER_JUDGE
    if _BLUNDER_JUDGE is None:
        from datagen.tree.search import SfJudge
        _BLUNDER_JUDGE = SfJudge(nodes=BLUNDER_NODES, multipv=1)
    return _BLUNDER_JUDGE


def _is_blunder(board, mv, faked=False):
    """A blunder both drops the eval AND leaves the mover badly off — merely
    missing a faster win doesn't count. In faked-turn (happens when
    same side moves twice in a row in a plan) context the
    numbers are hypothetical, so only a real hang counts (after < -150); on
    real turns the floor is 'not better' (after < 100)."""
    sf = _get_blunder_judge()
    before = sf.eval(board.fen())
    b2 = board.copy(stack=False)
    b2.push(mv)
    after = -sf.eval(b2.fen())
    return before - after >= BLUNDER_CP and after < (-150 if faked else 100)


def _line_ok(board0, line_text, mover):
    """Replay the line's clauses; False if an own recommendation is a
    detectable blunder, or if the line goes on after the game is already over
    (a plan that continues past the mate it delivers is describing moves that
    do not exist). Replay stops (keeping the line) where the clause order
    diverges from the position (elided opponent moves etc.).

    Note: True means 'no blunder was proven'. When we cannot prove the legality of a move, we return True.
    """
    b = board0.copy(stack=False)
    faked = False
    seq = move_seq(line_text, mover)
    for i, (actor, verb, piece, sqs) in enumerate(seq):
        if verb == "castle":
            tup = ("castle", "king", None, None)
        elif verb == "take":
            tup = (verb, piece, sqs[1] if len(sqs) > 1 else None,
                   sqs[0] if sqs else None)
        else:
            tup = (verb, piece, sqs[0] if len(sqs) > 1 else None,
                   sqs[-1] if sqs else None)
        if tup[3] is None and verb != "castle":
            return True
        if actor != ("White" if b.turn else "Black"):
            if b.is_check():             # can't fake the turn out of check - can't verify legality
                return True
            b.push(chess.Move.null())    # same side moves twice: fake the turn; can't verify legality
            if not b.is_valid():
                b.pop()
                return True
            faked = True                 # numbers are hypothetical from here on - changes threshold of blunder
        cands = [m for m in b.legal_moves if _tuple_matches(b, m, tup)]
        if len(cands) != 1:
            return True
        if actor == mover and _is_blunder(b, cands[0], faked=faked):
            return False
        b.push(cands[0])
        if b.is_game_over() and i + 1 < len(seq):
            return False
    return True


def _legal_own_moves(board0, line_text, mover):
    """How many of `mover`'s own moves in the line actually replay.

    Same convention the plan corpus was built under: when the same side moves
    twice in a row the turn is faked, so a plan may describe a manoeuvre without
    spelling out the opponent's replies. A clause that names a from-square which
    does not match any legal move is retried on (piece, destination) alone --
    the corpus writes a piece's ORIGINAL square, so a piece that has already
    moved earlier in the same plan is still named by where it started.

    ``complete`` is False when replay stops at any unresolved clause, including
    an opponent clause.  Blunder screening remains permissive in that case,
    but the narrative must not print the unverified tail (for example a model
    hallucinating that a pawn captures the king).

    -> (n own moves that replayed, total own moves named, replay completed)"""
    b = board0.copy(stack=False)
    seq = move_seq(line_text, mover)
    ok = 0
    total = sum(actor == mover for actor, _, _, _ in seq)
    complete = True
    for actor, verb, piece, sqs in seq:
        if verb == "castle":
            tup = ("castle", "king", None, None)
        elif verb == "take":
            tup = (verb, piece, sqs[1] if len(sqs) > 1 else None,
                   sqs[0] if sqs else None)
        else:
            tup = (verb, piece, sqs[0] if len(sqs) > 1 else None,
                   sqs[-1] if sqs else None)
        if tup[3] is None and verb != "castle":
            complete = False
            break
        if actor != ("White" if b.turn else "Black"):
            if b.is_check():
                complete = False
                break
            b.push(chess.Move.null())
            if not b.is_valid():
                b.pop()
                complete = False
                break
        cands = [m for m in b.legal_moves if _tuple_matches(b, m, tup)]
        if not cands:                       # retry ignoring the from-square
            loose = (tup[0], tup[1], None, tup[3])
            cands = [m for m in b.legal_moves if _tuple_matches(b, m, loose)]
        if len(cands) != 1:
            complete = False
            break
        if actor == mover:
            ok += 1
        b.push(cands[0])
    return ok, total, complete


# ------------------------------------------------------------- plan entries

def plan_entries(fen, plans_text, human=True, premise_first=True, raw_text=None):
    """The side-to-move's plan bullets, parsed: cleaned lines, resolved first
    move, own-move arm sequences. Strict-subset bullets and lines whose own
    recommendations contain a low-node-SF blunder are dropped. `plans_text` is
    the DECODED (human) rendering; `human` controls only how the kept lines
    are cleaned for the narrative. `premise_first=False` restores the pre-fix
    consequent-derived first move (validation against the lab trees only).

    Pass the token-space generation as `raw_text` to additionally get
    `machine_lines` per entry: the raw lines are structurally identical to the
    decoded ones (decode is line-preserving and all glue — bullets, arm
    markers, "From there:" — is plain English in both), so every filter,
    re-lettering, inlining and truncation is applied to both in lockstep."""
    board = chess.Board(fen)
    mover = "White" if board.turn else "Black"
    raw_lines = raw_text.split("\n") if raw_text is not None else None
    dec_lines = plans_text.split("\n")
    if raw_lines is not None and len(raw_lines) != len(dec_lines):
        raw_lines = None                 # cannot align: skip machine twins
    own, in_own = [], False              # [(decoded, raw-or-None)]
    for i, ln in enumerate(dec_lines):
        if ln.endswith("plans:"):
            in_own = ln.startswith(f"{mover}'s")
            continue
        if in_own and ln.strip():
            own.append((ln, raw_lines[i] if raw_lines is not None else None))
    bullets = []
    for pair in own:
        if pair[0].lstrip().startswith("- **"):
            bullets.append([pair])
        elif bullets:
            bullets[-1].append(pair)
    entries = []
    for b in bullets:
        stem_d, stem_r = b[0]
        if not _line_ok(board, stem_d, mover):   # own recommendation blunders
            continue
        arm_pairs = [p for p in b[1:] if re.match(r"\s*- [A-H]\. ", p[0])]
        other_pairs = [p for p in b[1:] if p not in arm_pairs]
        arm_pairs = [p for p in arm_pairs
                     if _line_ok(board, stem_d + " " + p[0], mover)]
        seen_arm, uniq = set(), []       # drop textually identical sub-recs
        for p in arm_pairs:
            key = re.sub(r"^\s*- [A-H]\. ", "", p[0]).strip()
            if key not in seen_arm:
                seen_arm.add(key)
                uniq.append(p)

        def _reletter(ln, i):
            return re.sub(r"(?<=- )[A-H](?=\. )", chr(65 + i), ln, count=1)
        arm_pairs = [(_reletter(d, i), _reletter(r, i) if r is not None else None)
                     for i, (d, r) in enumerate(uniq)]
        had_arms = any(re.match(r"\s*- [A-H]\. ", p[0]) for p in b[1:])
        if had_arms and not arm_pairs:
            stem_d = re.sub(r"\.? From there:$", ".", stem_d)
            if stem_r is not None:
                stem_r = re.sub(r"\.? From there:$", ".", stem_r)
        elif had_arms and len(arm_pairs) == 1:   # single survivor: inline it
            d1, r1 = arm_pairs[0]
            content = re.sub(r"^\s*- A\. ", "", d1).strip()
            stem_d = (re.sub(r" From there:$", "", stem_d)
                      + " From there, " + content)
            if stem_r is not None and r1 is not None:
                stem_r = (re.sub(r" From there:$", "", stem_r) + " From there, "
                          + re.sub(r"^\s*- A\. ", "", r1).strip())
            arm_pairs = []
        pairs = [(stem_d, stem_r)] + other_pairs + arm_pairs
        stem = _own_tuples(stem_d, mover)
        first = (_first_own_move(board, stem) if premise_first
                 else _legacy_first(board, stem_d, mover))
        seqs = ([stem + _own_tuples(d, mover) for d, _ in arm_pairs]
                if arm_pairs else [stem])
        ok, total, complete = _legal_own_moves(
            board, " ".join(d for d, _ in pairs), mover
        )
        if ok == 0:
            continue          # not even the opening move replays
        if not complete or ok < total:
            # The line stops replaying partway: retain only its verified stem
            # in the narrative.  This is independent of _line_ok's permissive
            # "no blunder proven" result for an unresolved clause.
            pairs = [(_stem_only(_clean(stem_d, mover, human)),
                      _stem_only(_clean(stem_r, mover, human=False))
                      if stem_r is not None else None)]   # part that replays
            seqs = [stem[:ok]]
            lines = [pairs[0][0]]
        else:
            lines = [_clean(d, mover, human) for d, _ in pairs]
        if first is None:
            continue          # the bullet's own move does not exist on this
                              # board (a knight that is not on the named square,
                              # a blocked path): it can never be played, so it
                              # is not a plan
        e = {"side": mover, "first": first, "lines": lines,
             "seqs": [[list(s), 0] for s in map(list, seqs)],
             "flat": sum((move_seq(d, mover) for d, _ in pairs), [])}
        if raw_lines is not None and all(r is not None for _, r in pairs):
            e["machine_lines"] = ([pairs[0][1]] if ok < total else
                                  [_clean(r, mover, human=False)
                                   for _, r in pairs])
        entries.append(e)
    keep = [e for i, e in enumerate(entries)
            if not any(_sublist(e["flat"], o["flat"])
                       for j, o in enumerate(entries) if j != i)]
    return keep


# --------------------------------------------------------------- live plans

def advance(live, board, mv, mover):
    """New live list with `mover`'s plans advanced past a played move."""
    out = []
    for pl in live:
        if pl["side"] != mover:
            out.append(pl)
            continue
        seqs = []
        for steps, ptr in pl["seqs"]:
            if ptr < len(steps) and _tuple_matches(board, mv, tuple(steps[ptr])):
                seqs.append([steps, ptr + 1])
            else:
                seqs.append([steps, ptr])
        out.append({**pl, "seqs": seqs})
    return out


def is_expected(live, board, mv, mover):
    """Is `mv` the next own move of some live plan of `mover`?"""
    for pl in live:
        if pl["side"] != mover:
            continue
        for steps, ptr in pl["seqs"]:
            if ptr < len(steps) and _tuple_matches(board, mv, tuple(steps[ptr])):
                return True
    return False
