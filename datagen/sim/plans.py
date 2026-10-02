"""Games -> plans: which moves / maneuvers are over-represented in the games
the side to move wins, verbalized as per-piece plan tries.

Feature extraction (first PLY_CAP plies of each game): single moves and
maneuvers (2..MAX_WIN own moves of one piece; square-revisiting windows
dropped), where every capture splices in the victim's own history and every
quiet move onto a root-occupied square splices in the occupant's departure —
recursively, in causal order — so each feature is a self-contained causal
chain. Pieces are named from the root position and tracked.

Target outcome: pov wins; if under WIN_FRAC of the games it widens to
wins+draws. Features need support in >= MIN_GAMES games, rank by the smoothed
ratio P(feature | target) / P(feature | other), keep TOP under the validity
rule (an item's first square must be the piece's root square or a square an
already-listed item of that piece reached; prefix maneuvers omitted).

Plan building: an item is admitted only if its chain starts with the owner's
own move; a piece's items sharing a first step merge into one trie; up to 4
plans per side, priority maneuver1 > maneuver2 > move1 > maneuver3 > move2...
Every mentioned move is replayed on a simulation board from the root
(_sim_apply): illegal moves, capture mismatches, landings a cheaper piece
attacks, and redundant extensions truncate the plan there; prerequisites that
fail the replay are silently not cited. Rendering emits paired human / machine
(stage-1-4 POV token) texts.

Entry point: plans_from_games(root_fen, games) with games =
[{"white", "black", "result", "moves": [uci...]}, ...] — the shape the
simulate sampler stores as position metadata.
"""
from collections import defaultdict

import chess

from utils.utils import POV_SQUARE_TOKENS, _PIECE_TO_POV_TOKEN

MIN_GAMES = 5
TOP = 10
MAX_WIN = 4     # maneuver window length (own moves)
PLY_CAP = 10    # only the first N plies of each game count
WIN_FRAC = 0.15  # widen the target to wins+draws below this win rate


def _labels(board):
    """Readable piece names from the root position."""
    counts = defaultdict(int)
    for p in board.piece_map().values():
        counts[(p.color, p.piece_type)] += 1
    out = {}
    for sq, p in board.piece_map().items():
        side = "White's" if p.color else "Black's"
        name = chess.piece_name(p.piece_type)
        if counts[(p.color, p.piece_type)] == 1:
            out[sq] = f"{side} {name}"
        else:
            out[sq] = f"{side} {chess.square_name(sq)}-{name}"
    return out


def _expand(label, steps, _seen=None):
    """Causal order: each capture is preceded by the victim's full story; each
    quiet move onto a root-occupied square by the occupant's departure. A step
    already cited earlier in the chain is not repeated."""
    seen = _seen if _seen is not None else set()
    out = []
    for (f, t, cap, vlab, vstory, vac) in steps:
        if vac:
            out += _expand(vac[0], vac[1], seen)
        if vstory:
            out += _expand(vlab, vstory, seen)
        step = (label, f, t, cap)
        if step not in seen:
            seen.add(step)
            out.append(step)
    return out


def _render_exp(exp) -> str:
    segs = []                        # [piece label, text, last-to-square]
    for lab, f, t, cap in exp:
        arrow = chess.square_name(t) + (f"×{cap}" if cap else "")
        if segs and segs[-1][0] == lab and segs[-1][2] == f:
            segs[-1][1] += "→" + arrow
            segs[-1][2] = t
        else:
            segs.append([lab, f"{chess.square_name(f)}→{arrow}", t])
    return " | ".join(f"{lab}: {txt}" for lab, txt, _ in segs)


def _add(feats_side, kind, label, steps, meta, chains):
    exp = _expand(label, steps)
    s = f"{kind}|" + _render_exp(exp)
    feats_side.add(s)
    if s not in meta:
        own = [chess.square_name(steps[0][0])] + [chess.square_name(st[1])
                                                  for st in steps]
        meta[s] = (label, tuple(own))
        chains[s] = tuple(exp)


def game_features(root_board, moves, result, meta, chains):
    """-> (result, {white feature set}, {black feature set},
           [ordered (label, from, to, cap) steps of the first PLY_CAP plies])"""
    board = root_board.copy(stack=False)
    ids = dict(_labels(board))
    root_occ = dict(ids)             # root square -> its original occupant
    root_of = {lab: sq for sq, lab in ids.items()}
    vacate = {}                      # root square -> (label, steps) that freed it
    feats = {chess.WHITE: set(), chess.BLACK: set()}
    stories = defaultdict(list)      # label -> [(from, to, cap, vlab, vstory, vac)]
    owner = {}                       # label -> side
    seq = []

    for ply, uci in enumerate(moves):
        if ply >= PLY_CAP:
            break
        mv = chess.Move.from_uci(uci)
        us = board.turn
        lab = ids.get(mv.from_square)
        if lab is None:
            board.push(mv)
            continue
        owner[lab] = us
        cap_name = vlab = vstory = None
        if board.is_capture(mv):
            vsq = mv.to_square
            if board.is_en_passant(mv):
                vsq = mv.to_square + (-8 if us else 8)
            cap_name = chess.piece_name(board.piece_at(vsq).piece_type)
            vlab = ids.get(vsq)
            if vlab:
                vstory = tuple(stories.get(vlab, ()))
                feats[us].add(f"cap|{lab} takes {vlab} on "
                              f"{chess.square_name(vsq)}")
                ids.pop(vsq, None)
        vac = None
        if cap_name is None:         # captures: the victim's chain explains the square
            occ = root_occ.get(mv.to_square)
            if occ and occ != lab:
                vac = vacate.get(mv.to_square)
        step = (mv.from_square, mv.to_square, cap_name, vlab, vstory, vac)
        if not stories[lab]:         # first move: this is what frees its root square
            vacate.setdefault(root_of[lab], (lab, (step,)))
        if vlab and vstory == ():    # victim captured in place, never having moved
            vacate.setdefault(vsq, (lab, (step,)))
        stories[lab].append(step)
        seq.append((lab, mv.from_square, mv.to_square, cap_name))
        _add(feats[us], "move", lab, [step], meta, chains)
        ids.pop(mv.from_square, None)
        ids[mv.to_square] = lab
        if board.is_castling(mv):
            rf, rt = ((chess.H1, chess.F1) if mv.to_square == chess.G1 else
                      (chess.A1, chess.D1) if mv.to_square == chess.C1 else
                      (chess.H8, chess.F8) if mv.to_square == chess.G8 else
                      (chess.A8, chess.D8))
            if rf in ids:
                rlab = ids.pop(rf)
                ids[rt] = rlab
                rstep = (rf, rt, None, None, None, None)
                if not stories[rlab]:
                    vacate.setdefault(rf, (rlab, (rstep,)))
                stories[rlab].append(rstep)
                seq.append((rlab, rf, rt, None))
                owner[rlab] = us
        board.push(mv)

    # maneuvers: windows of 2..MAX_WIN of a piece's own steps (always linked);
    # windows revisiting a square are dropped
    for lab, steps in stories.items():
        us = owner[lab]
        for i in range(len(steps) - 1):
            for j in range(i + 2, min(i + MAX_WIN + 1, len(steps) + 1)):
                win = steps[i:j]
                sqs = [win[0][0]] + [st[1] for st in win]
                if len(set(sqs)) < len(sqs):
                    continue
                _add(feats[us], "man", lab, win, meta, chains)
    return result, feats[chess.WHITE], feats[chess.BLACK], seq


def assoc(games):
    """games: [(in_target, featset)] -> {feat: (n_t, n_nt, ratio)}"""
    T = sum(1 for t, _ in games if t)
    NT = len(games) - T
    cnt_t, cnt_nt = defaultdict(int), defaultdict(int)
    for t, fs in games:
        for f in fs:
            (cnt_t if t else cnt_nt)[f] += 1
    out = {}
    for f in set(cnt_t) | set(cnt_nt):
        nt_, nnt = cnt_t.get(f, 0), cnt_nt.get(f, 0)
        if nt_ + nnt < MIN_GAMES:
            continue
        ratio = ((nt_ + 0.5) / (T + 1)) / ((nnt + 0.5) / (NT + 1))
        out[f] = (nt_, nnt, ratio)
    return out


def drop_subchains(rows, meta):
    """Drop an item whose own path is a contiguous subpath of another item of
    the same piece with identical counts (adds nothing)."""
    kept = []
    for f, (a, b, r) in rows:
        piece, own = meta[f]
        s = "→".join(own)
        dominated = any(f2 != f and meta[f2][0] == piece
                        and s in "→".join(meta[f2][1]) and (a2, b2) == (a, b)
                        for f2, (a2, b2, _) in rows)
        if not dominated:
            kept.append((f, (a, b, r)))
    return kept


def select_valid(data, top, root_board, meta):
    """Validity-constrained top lists over moves + maneuvers, in ratio order."""
    root = {lab: chess.square_name(sq)
            for sq, lab in _labels(root_board).items()}
    cands = sorted(((f, v) for f, v in data.items() if not f.startswith("cap|")),
                   key=lambda kv: (-kv[1][2], -(kv[1][0] + kv[1][1])))
    cands = drop_subchains(cands, meta)
    valid = defaultdict(set)
    moves, mans = [], []
    for f, v in cands:
        piece, own = meta[f]
        if not valid[piece]:
            valid[piece].add(root[piece])
        if own[0] not in valid[piece]:
            continue
        valid[piece].update(own[1:])
        (moves if f.startswith("move|") else mans).append((f, v))
        if len(moves) >= top and len(mans) >= top:
            break
    # omit maneuvers that are prefixes of other selected maneuvers
    mans = [m for m in mans if not any(
        o != m and meta[o[0]][0] == meta[m[0]][0]
        and meta[m[0]][1] == meta[o[0]][1][:len(meta[m[0]][1])]
        for o in mans)]
    return moves[:top], mans[:top]


MARKS = (("A", "B", "C", "D", "E", "F", "G", "H"),
         ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii"),
         ("1", "2", "3", "4", "5", "6", "7", "8"),
         ("a", "b", "c", "d", "e", "f", "g", "h"))


def _tside(lab):
    return lab.startswith("White's")


def _pname(lab):
    return lab.split(" ", 1)[1]


def _trie(chains_list):
    root = {"kids": {}, "end": False}
    for ch in chains_list:
        n = root
        for st in ch:
            n = n["kids"].setdefault(st, {"kids": {}, "end": False})
        n["end"] = True
    return root


def _short(nm):
    return nm.split("-", 1)[1] if "-" in nm else nm


_PTYPE = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
          "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}


def _sqpair(sq, pov_white):
    """(human square name, POV square token), stage-1-4 convention."""
    return (chess.square_name(sq),
            POV_SQUARE_TOKENS[sq if pov_white else sq ^ 56])


def _piecepair(lab, named, pov_white):
    """(human name, machine tokens) for a piece reference: root-square-
    qualified at first mention (piece token + square token), bare type
    afterwards -- unless another mentioned piece of the same side and type
    makes the bare name ambiguous."""
    nm = _pname(lab)
    mine = _tside(lab) == pov_white
    tok = _PIECE_TO_POV_TOKEN[(mine, _PTYPE[_short(nm)])]
    if lab in named and not any(
            l != lab and _tside(l) == _tside(lab)
            and _short(_pname(l)) == _short(nm) for l in named):
        return _short(nm), tok
    if "-" in nm:
        sq = chess.parse_square(nm.split("-", 1)[0])
        return nm, tok + _sqpair(sq, pov_white)[1]
    return nm, tok


def _victpair(cap, mine, t, pov_white):
    """Victim reference of a capture by a `mine` piece onto t."""
    th, tm = _sqpair(t, pov_white)
    vtok = _PIECE_TO_POV_TOKEN[(not mine, _PTYPE[cap])]
    if mine:
        return f"the {cap} on {th}", f"{vtok}{tm}"
    return f"our {cap} on {th}", f"<PLAYER>'s {vtok}{tm}"


def _vp(st, prev_lab, pov_white, named=frozenset()):
    """(human, machine) verb phrase for a step by the plan owner."""
    lab, _f, t, cap = st
    mine = _tside(lab) == pov_white
    ph, pm = _piecepair(lab, named, pov_white)
    poss_h, poss_m = (("our", "<PLAYER>'s") if mine
                      else ("their", "<OPPONENT>'s"))
    th, tm = _sqpair(t, pov_white)
    if cap:
        vh, vm = _victpair(cap, mine, t, pov_white)
        return (f"take {vh} with {poss_h} {ph}",
                f"take {vm} with {poss_m} {pm}")
    if lab == prev_lab:
        return (f"continue with {poss_h} {ph} to {th}",
                f"continue with {poss_m} {pm} to {tm}")
    verb = "push" if _short(_pname(lab)) == "pawn" else "move"
    return (f"{verb} {poss_h} {ph} to {th}", f"{verb} {poss_m} {pm} to {tm}")


def _cond(st, pov_white, named=frozenset()):
    """(human, machine) full clause for a step cited as condition/prereq."""
    lab, _f, t, cap = st
    mine = _tside(lab) == pov_white
    ph, pm = _piecepair(lab, named, pov_white)
    poss_h, poss_m = (("our", "<PLAYER>'s") if mine
                      else ("their", "<OPPONENT>'s"))
    subj_h = "we" if mine else "the opponent"
    subj_m = "<PLAYER>" if mine else "<OPPONENT>"
    th, tm = _sqpair(t, pov_white)
    if cap:
        vh, vm = _victpair(cap, mine, t, pov_white)
        return (f"{subj_h} take{'' if mine else 's'} {vh} "
                f"with {poss_h} {ph}",
                f"{subj_m} takes {vm} with {poss_m} {pm}")
    verb = "push" if _short(_pname(lab)) == "pawn" else "move"
    verb_h = verb if mine else verb + ("es" if verb == "push" else "s")
    verb_m = verb + ("es" if verb == "push" else "s")
    return (f"{subj_h} {verb_h} {poss_h} {ph} to {th}",
            f"{subj_m} {verb_m} {poss_m} {pm} to {tm}")


def _prereq_fn(game_steps):
    """pre(b, a) -> steps always played between b and a (and always in the
    same order) across every game where a follows b; () otherwise."""
    cache = {}

    def pre(b, a):
        key = (b, a)
        if key in cache:
            return cache[key]
        between = []
        for gs in game_steps:
            try:
                i = gs.index(b) if b is not None else -1
                j = gs.index(a, i + 1)
            except ValueError:
                continue
            between.append(gs[i + 1:j])
        common = set(between[0]) if between else set()
        for s in between[1:]:
            common &= set(s)
        common.discard(a)
        out = ()
        if common:
            P = [{s: sq.index(s) for s in common} for sq in between]
            ordered = sorted(common, key=lambda s: P[0][s])
            bad = set()
            for ii in range(len(ordered)):
                for jj in range(ii + 1, len(ordered)):
                    x, y = ordered[ii], ordered[jj]
                    if any(p[x] > p[y] for p in P):
                        bad.add(x)
                        bad.add(y)
            out = tuple(s for s in ordered if s not in bad)
        cache[key] = out
        return out
    return pre


_CK = {(chess.E1, chess.G1): ((chess.H1, chess.F1), "kingside"),
       (chess.E1, chess.C1): ((chess.A1, chess.D1), "queenside"),
       (chess.E8, chess.G8): ((chess.H8, chess.F8), "kingside"),
       (chess.E8, chess.C8): ((chess.A8, chess.D8), "queenside")}
_CR = {v[0]: (k, v[1]) for k, v in _CK.items()}


def _kc(st):
    """King half of a castling move (a king never moves two files otherwise)."""
    lab, f, t, cap = st
    if not cap and _pname(lab).endswith("king"):
        return _CK.get((f, t))
    return None


def _rc(st):
    lab, f, t, cap = st
    if not cap and _pname(lab).endswith("rook"):
        return _CR.get((f, t))
    return None


_VAL = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9,
        "king": 1000}


def _sim_new(root_board):
    return {"board": root_board.copy(stack=False), "traj": {}, "ncaps": 0,
            "log": []}


def _sim_copy(sim):
    # "log" is shared (same list) across copies: the first owner-side entry is
    # the plan's first verbalized move, which lands before any branch forks
    return {"board": sim["board"].copy(stack=False),
            "traj": dict(sim["traj"]), "ncaps": sim["ncaps"],
            "log": sim["log"]}


def _fix_turn(board, side):
    if board.turn != side:
        board.push(chess.Move.null())


def _mk_move(board, f, t, ptype):
    promo = (chess.QUEEN if ptype == "pawn"
             and chess.square_rank(t) in (0, 7) else None)
    return chess.Move(f, t, promotion=promo)


def _sim_apply(sim, st, castle_king=None, quality=False):
    """Replay a mentioned move on the running simulation (mutates sim; null
    moves fix turn alternation). Returns None if acceptable, else why not:
    the move must be legal on the simulated board and consistent with its
    stated capture; with `quality` (the plan owner's own moves) the piece
    must not land where something cheaper attacks it (cheaper than piece
    value minus victim value on captures), and a move reachable directly
    from an earlier mentioned square, with no captures mentioned in between,
    is a redundant extension."""
    lab, f, t, cap = st
    side = _tside(lab)
    board = sim["board"]
    if castle_king is not None:
        _fix_turn(board, side)
        m = chess.Move(*castle_king)
        if m not in board.legal_moves:
            return "illegal castle"
        board.push(m)
        sim["log"].append((side, m.uci()))
        return None
    ptype = _short(_pname(lab))
    _fix_turn(board, side)
    m = _mk_move(board, f, t, ptype)
    if m not in board.legal_moves:
        return "illegal"
    vsq = t + (-8 if side else 8) if board.is_en_passant(m) else t
    victim = board.piece_at(vsq)
    if bool(cap) != (victim is not None) or \
            (cap and victim and chess.piece_name(victim.piece_type) != cap):
        return "capture mismatch"
    if quality:
        for (c_sq, snap, nc) in sim["traj"].get(lab, ()):
            if nc != sim["ncaps"] or c_sq == f:
                continue
            b2 = snap.copy(stack=False)
            _fix_turn(b2, side)
            if _mk_move(b2, c_sq, t, ptype) in b2.legal_moves:
                return "redundant extension"
    sim["traj"][lab] = sim["traj"].get(lab, ()) + (
        (f, board.copy(stack=False), sim["ncaps"]),)
    board.push(m)
    if cap:
        sim["ncaps"] += 1
    if quality:
        thr = _VAL["queen" if m.promotion else ptype] - \
              (_VAL.get(cap, 0) if cap else 0)
        for sq in board.attackers(not side, t):
            p = board.piece_at(sq)
            if _VAL.get(chess.piece_name(p.piece_type), 1000) < thr:
                return "capturable by cheaper"
    sim["log"].append((side, m.uci()))
    return None


def _say(node, prev, owner_white, pov_white, depth, text, state, pre,
         seen=frozenset(), named=frozenset(), implied=frozenset(),
         roots=None, sim=None):
    """Render a plan trie linearly from `node`; branches become marker bullets
    (A/B, i/ii, 1/2, a/b by nesting depth). `prev` is the previously mentioned
    step; pre(prev, step) gives moves always played between them, verbalized
    as "after ..." prerequisites. `seen` steps and `implied` (side, from, to)
    castle halves are never re-mentioned; `named` tracks pieces mentioned on
    this path. `sim` replays every mentioned move from the root position:
    steps that fail it (see _sim_apply) truncate the plan there; prerequisites
    that fail it are silently not cited. `state`: start | chain | cond.
    Returns [] when nothing could be rendered. All texts are (human, machine)
    pairs; the machine side uses the stage-1-4 POV vocabulary."""
    owner_pov = owner_white == pov_white
    owner_pron = ("we", "<PLAYER>") if owner_pov else ("they", "<OPPONENT>")

    def T(t, *xs):
        h, m = t
        for x in xs:
            if isinstance(x, str):
                h, m = h + x, m + x
            else:
                h, m = h + x[0], m + x[1]
        return h, m

    def R(t):
        return t[0].rstrip(",; "), t[1].rstrip(",; ")

    def castle_cond(side, name):
        if side == pov_white:
            return "we castle " + name, "<PLAYER> castles " + name
        return "the opponent castles " + name, "<OPPONENT> castles " + name

    def prep(st, prev_st, seen, named, implied, sim):
        """Prerequisites + castling fusion + replay for st. Returns (ok,
        aft_pair, vp_pair, cond_pair, seen, named, implied, sim)."""
        side = _tside(st[0])
        sim = _sim_copy(sim) if sim is not None else None
        cs = [c for c in pre(prev_st, st) if c not in seen
              and (_tside(c[0]), c[1], c[2]) not in implied]
        castle = castle_king = None
        kc = _kc(st)
        if kc:
            castle, castle_king = kc[1], (st[1], st[2])
            implied = implied | {(side,) + kc[0]}
        else:
            rc = _rc(st)
            if rc:
                king = next((c for c in cs if _tside(c[0]) == side
                             and (c[1], c[2]) == rc[0] and _kc(c)), None)
                if king is not None:
                    cs.remove(king)
                    castle, castle_king = rc[1], rc[0]
                    implied = implied | {(side, king[1], king[2])}
        parts, skip, cited = [], set(), set()
        for c in cs:
            if c in skip:
                continue
            # citable only if traceable: the piece moves off its root square
            # or off a square whose arrival was already mentioned
            if not (roots is None or c[1] == roots.get(c[0])
                    or any(s[0] == c[0] and s[2] == c[1]
                           for s in seen | cited)):
                continue
            ckc = _kc(c)
            if sim is not None and _sim_apply(
                    sim, c, castle_king=(c[1], c[2]) if ckc else None):
                continue             # not legal in the replay: don't cite
            if ckc:
                pair = next((d for d in cs if _tside(d[0]) == _tside(c[0])
                             and (d[1], d[2]) == ckc[0]), None)
                if pair is not None:
                    skip.add(pair)
                    cited.add(pair)
                parts.append(castle_cond(_tside(c[0]), ckc[1]))
                implied = implied | {(_tside(c[0]),) + ckc[0]}
            else:
                parts.append(_cond(c, pov_white, named))
            cited.add(c)
            named = named | {c[0]}
        if parts:
            aft = ("after " + " and ".join(p[0] for p in parts) + ", ",
                   "after " + " and ".join(p[1] for p in parts) + ", ")
        else:
            aft = ("", "")
        seen = seen | cited
        ok = True
        if sim is not None:
            ok = _sim_apply(sim, st, castle_king=castle_king,
                            quality=side == owner_white) is None
        if castle:
            vpc = ("castle " + castle, "castle " + castle)
            condc = castle_cond(side, castle)
        else:
            plab = prev_st[0] if prev_st else None
            vpc = _vp(st, plab, pov_white, named)
            condc = _cond(st, pov_white, named)
        return ok, aft, vpc, condc, seen, named, implied, sim

    cur = node
    stable = text if state == "chain" else ("", "")
    while True:
        kids = cur["kids"]
        if not kids:
            return [T(R(text), ".")] if text[0] else []
        if len(kids) > 1:
            arms = []
            marks = MARKS[depth % len(MARKS)]
            for st, kid in kids.items():
                ok, aft, vpc, condc, bseen, bnamed, bimpl, bsim = prep(
                    st, prev, seen, named, implied, sim)
                if not ok:
                    continue
                if _tside(st[0]) == owner_white:
                    blead = T(aft, owner_pron, " can ", vpc)
                    bstate = "chain"
                else:
                    blead, bstate = T(aft, "if ", condc), "cond"
                sub = _say(kid, st, owner_white, pov_white, depth + 1,
                           blead, bstate, pre, bseen | {st},
                           bnamed | {st[0]}, bimpl, roots, bsim)
                if sub:
                    arms.append(sub)
            if not arms:
                return [T(R(text), ".")] if text[0] else []
            if len(arms) == 1:       # single surviving arm: keep it inline
                return ([T(R(text), ". From there, ", arms[0][0])]
                        + arms[0][1:])
            lines = [T(R(text), ". From there:")]
            for k, sub in enumerate(arms):
                mk = marks[min(k, len(marks) - 1)]
                pfx = f"{'  ' * (depth + 1)}- {mk}. "
                lines.append((pfx + sub[0][0], pfx + sub[0][1]))
                lines += sub[1:]
            return lines
        (st, kid), = kids.items()
        lab = st[0]
        if st in seen or (_tside(lab), st[1], st[2]) in implied:
            cur = kid            # already stated (prerequisite/castle half)
            continue
        ok, aft, vpc, condc, nseen, nnamed, nimplied, nsim = prep(
            st, prev, seen, named, implied, sim)
        if not ok:
            return [T(R(stable), ".")] if stable[0] else []
        seen, named, implied, sim = nseen, nnamed, nimplied, nsim
        if cur["end"]:
            text = T(R(text), ". From there, ", aft)
            if _tside(lab) == owner_white:
                text = T(text, owner_pron, " can also ", vpc)
                state = "chain"
            else:
                text = T(text, "if ", condc)
                state = "cond"
        elif _tside(lab) == owner_white:
            if state == "start":
                intro = (("we can ", "<PLAYER> can ") if owner_pov
                         else ("the opponent can ", "<OPPONENT> can "))
                lh, lm = T(aft, intro, vpc)
                text = T(text, (lh[0].upper() + lh[1:],
                                lm[0].upper() + lm[1:]))
            elif aft[0]:
                text = T(text, "; ", aft, owner_pron, " can ", vpc)
            elif state == "cond":
                text = T(text, ", ", owner_pron, " can ", vpc)
            else:
                text = T(text, ", then ", vpc)
            state = "chain"
        else:
            text = T(text, "; ", aft, "if ", condc)
            state = "cond"
        if state == "chain":
            stable = text
        named = named | {lab}
        seen = seen | {st}
        prev = st
        cur = kid


def build_plans(mv_rows, man_rows, owner_white, pov_white, meta, chains,
                game_steps=(), root_board=None):
    """Merged, verbalized plans for one side. Conventions: an item is admitted
    only if its causal chain starts with the owner's own move; a piece's items
    sharing the same first step merge into one trie (one merged plan per
    piece, the group of its highest-priority item, except that an item whose
    chain is a strict superstring of the winner's supersedes it); up to 4
    plans total, in priority order maneuver 1 > maneuver 2 > move 1 >
    maneuver 3 > move 2 ..."""
    def ok(f):
        return chains[f] and _tside(chains[f][0][0]) == owner_white
    movs = [f for f, _ in mv_rows if ok(f)]
    mans = [f for f, _ in man_rows if ok(f)]
    prio = list(mans[:2])
    i, j = 0, 2
    while i < len(movs) or j < len(mans):
        if i < len(movs):
            prio.append(movs[i])
            i += 1
        if j < len(mans):
            prio.append(mans[j])
            j += 1
    groups = defaultdict(list)
    by_piece = defaultdict(list)
    for f in prio:
        groups[(meta[f][0], chains[f][0])].append(f)
        by_piece[meta[f][0]].append(f)

    def chain_in(small, big):
        n, m = len(small), len(big)
        return n < m and any(big[k:k + n] == small for k in range(m - n + 1))

    pre = (_prereq_fn(list(game_steps)) if game_steps
           else (lambda b, a: ()))
    roots = {}
    for gs in game_steps:
        for lab, fr, _t, _c in gs:
            roots.setdefault(lab, fr)   # a piece's first move leaves its root
    plans, seen = [], set()
    for f in prio:
        piece = meta[f][0]
        if piece in seen:
            continue
        seen.add(piece)
        while True:                  # strict superstrings supersede the winner
            sup = next((g for g in by_piece[piece]
                        if chain_in(chains[f], chains[g])), None)
            if sup is None:
                break
            f = sup
        chain_group = [chains[g] for g in groups[(piece, chains[f][0])]]
        owner_pov = owner_white == pov_white
        sim = _sim_new(root_board) if root_board is not None else None
        lines = _say(_trie(chain_group), None, owner_white, pov_white, 0,
                     ("", ""), "start", pre, roots=roots, sim=sim)
        if not lines:
            continue                 # first step fails the replay: drop plan
        own_from = roots.get(piece)
        if (sim is not None and own_from is not None
                and not any(s == owner_white
                            and u[:2] == chess.square_name(own_from)
                            for s, u in sim["log"])):
            seen.discard(piece)      # rendering trimmed the piece's own move:
            continue                 # not this piece's plan; let its next item try
        title_h = ("Our " if owner_pov else "The opponent's ") + _pname(piece)
        title_m = (("<PLAYER>'s " if owner_pov else "<OPPONENT>'s ")
                   + _piecepair(piece, frozenset(), pov_white)[1])
        plans.append({
            "piece": piece,
            "uci": (next((u for s, u in sim["log"] if s == owner_white), None)
                    if sim is not None else None),
            "text": "\n".join([f"- **{title_h}**: {lines[0][0]}"]
                              + [ln[0] for ln in lines[1:]]),
            "machine": "\n".join([f"- **{title_m}**: {lines[0][1]}"]
                                 + [ln[1] for ln in lines[1:]])})
        if len(plans) == 4:
            break
    return plans


def plans_from_games(root_fen: str, games: list) -> dict:
    """Both sides' plans from a position's simulated games.

    games: [{"white", "black", "result", "moves": [uci...]}, ...]. Returns the
    sets-json shape: {fen, n_games, pov, results, target, sets, plans}."""
    root_board = chess.Board(root_fen)
    meta, chains = {}, {}
    per_game = [game_features(root_board, g["moves"], g["result"], meta, chains)
                for g in games]

    pov = root_board.turn
    pov_win = "1-0" if pov == chess.WHITE else "0-1"
    results = [p[0] for p in per_game]
    n_win = results.count(pov_win)
    widened = n_win < WIN_FRAC * len(per_game)

    def is_target(r):
        return r == pov_win or (widened and r == "1/2-1/2")

    out = {"fen": root_fen, "n_games": len(per_game),
           "pov": "white" if pov else "black",
           "results": {"pov_wins": n_win,
                       "draws": results.count("1/2-1/2"),
                       "pov_losses": len(results) - n_win
                       - results.count("1/2-1/2")},
           "target": "win+draw" if widened else "win", "sets": {},
           "plans": {}}
    for who in ("pov", "opp"):
        use_white = (pov == chess.WHITE) == (who == "pov")
        gsets = [(is_target(r), w if use_white else b)
                 for r, w, b, _ in per_game]
        data = assoc(gsets)
        mv_rows, man_rows = select_valid(data, TOP, root_board, meta)
        for kind, rows in (("moves", mv_rows), ("maneuvers", man_rows)):
            out["sets"][f"{who}_{kind}"] = [
                {"feature": f.split("|", 1)[1], "n_target": a, "n_other": b_,
                 "ratio": round(r, 3)} for f, (a, b_, r) in rows]
        out["plans"][who] = build_plans(mv_rows, man_rows, use_white,
                                        pov == chess.WHITE, meta, chains,
                                        [p[3] for p in per_game],
                                        root_board=root_board)
    return out
