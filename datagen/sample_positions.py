#!/usr/bin/env python3
"""Sample positions (with move history) into JSONL.

The default source is Lichess PGN/PGN.zst.  Two self-distillation sources are
also supported: ``--puzzles`` samples rating-stratified Lichess puzzle JSONLs,
and ``--play-games`` samples model-vs-Stockfish games.  Puzzle sources can also
use ``motifs: true`` to run the same stage-5 motif detector at each post-setup
puzzle root. Gameplay is delegated to ``self_distill.get_play_positions``;
``--score-play PATH`` delegates CPU scoring and filtering to
``self_distill.score_play_positions``.

Output lines are [start_fen, move_list, end_fen] triples:
  - start_fen: FEN at most HISTORY_PLIES (7) moves before end_fen
  - move_list: the <=7 UCI moves from start_fen to end_fen
  - end_fen:   FEN of the sampled position
LC0 encodes 8 history boards (start_fen + up to 7 moves), so more history is wasted.

`--lookahead_ply k` (k>0, stage-3/4 forward data) gates the SAME destination
position P as usual, but presents the position Q = P - k instead: the triple is
Q's (its own <=7-ply history), with two extra fields appended —
    [start_fen, move_list, end_fen(=Q), lookahead_moves(Q->P, k UCI moves), final_fen(=P)]
so the model sees Q + history, plays the k saved moves, and answers on P (whose
FEN is saved to avoid recomputing it). Requires P - k >= START_PLY so Q stays in
the configured move window. k=0 is exactly the 3-element triple above.

The lookahead spec may be a fixed int `k` or a `[lo, hi]` range (config key
`lookahead_ply: [1, 8]`), in which case each position draws k ~ randint(lo,
min(hi, P - START_PLY)) — uniform in [lo, hi] except near the opening, where it
is capped by the room available before START_PLY.

One sampler, gated by features. The --config YAML lists zero or more feature gates
plus how many positions to take per game:

    features: []                 # random sample (no gate)
    per_game: 5

    features: [in_check]         # only positions where the side to move is in check
    per_game: 4

    features: [is_checkmate]     # checkmate positions
    per_game: 1

    features: [in_check, near_mate]   # a check <= near_plies before a game-ending mate
    per_game: 1
    near_plies: 6

Verdict mode (stage 5.a): `verdict: true` replaces the feature gates with the
percentile-threshold class acceptances of datagen.tree.strong_primitives (mate/tactic,
positional at the frozen activity cutoffs, material, dynamic/dry balance). Per
game, `per_game` (default 1) random plies are drawn and each surviving position
is written as the usual triple; class balance follows the seeded keep
probabilities plus a running per-class cap of `cap_per_million` (default 20000)
rows per million candidates scanned. Needs Stockfish (datagen.tree.search).

    verdict: true
    per_game: 1

Simulate mode (stage 5.b): `simulate: true` samples positions generically
(per_game random plies >= START_PLY, game-over positions skipped) and plays
`n_games` engine-pool games from each across `workers` processes
(datagen.sim.games), storing them IN the output record —
    [start_fen, move_list, end_fen, {"games": [...]}]
— so the dataset builder derives plans from the stored games without ever
re-running them. `max_positions` caps the number of examples (the process is
slow: ~n_games engine games per position); output is appended and flushed per
position and an interrupted run resumes past the positions already written.

    simulate: true
    per_game: 1
    n_games: 100
    workers: 16
    max_positions: 30000

Motif mode: `motifs: true` scans every eligible game ply and writes positions
where at least one threshold-strong positional, exact-material, or filtered
tactical motif is present.  Unlike the other PGN modes, its JSONL rows are
named objects containing a nonempty motif list and up to 15 preceding plies:

    {
      "schema_version": 1,
      "start_fen": "...",
      "moves": ["..."],
      "fen": "...",
      "source": {"game_id": "...", "ply": 37},
      "motifs": [{"motif": "knight_outpost", "metadata": {...}}]
    }

The 15 plies support a later split into at most 7 LC0-history plies followed
by a prompt sequence of at most 8 plies. Material-only positions are rejected;
all tactical positions are retained; half the positions with a sub-10%-yield
positional motif are retained independently; and `per_game` random positions
are selected from the remaining non-material common pool.

    motifs: true
    per_game: 10
    rare_keep_probability: 0.5

For each game the sampler enumerates every position from START_PLY to the end,
applies each feature gate (position predicates from datagen.position_features, plus
the game-level `near_mate` gate) to whittle the set down, then writes
min(available, per_game) random survivors. Feature names are the keys of
position_features.FEATURES; `near_mate` is the one game-level gate.

Run:
    python -m datagen.sample_positions <pgn-glob> --config cfg.yaml --output out.jsonl [--seed N]

The PGN argument is a glob, so one config is reused across shards by varying
--output (and train vs val-test simply point at different source PGNs).

Interactive sanity check (no config):
    python -m datagen.sample_positions <pgn> --interactive [--seed N]
"""
import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from collections import Counter
from glob import glob
from pathlib import Path

import chess
import chess.pgn
import yaml

from datagen import position_features as pf

# LC0 encodes up to 8 history boards: storing more than 7 moves adds no signal.
HISTORY_PLIES = 7
# Motif records retain 7 LC0-history plies plus up to 8 prompt plies.
MOTIF_CONTEXT_PLIES = 15
# Positional motifs below 10% yield in the calibration pilot. A position with
# any of these bypasses common-pool sampling with probability 0.5.
RARE_POSITIONAL_MOTIFS = frozenset({
    "active_bishop",
    "active_bishop_pair",
    "bad_rook",
    "knight_outpost",
    "pawn_structure",
    "space_advantage",
})
RARE_POSITIONAL_KEEP_PROBABILITY = 0.5
# Skip openings — both players must have made at least 5 moves.
START_PLY = 10
# near_mate gate: how far before a game-ending mate to look back.
NEAR_PLIES = 6


def normalized_fen(fen: str) -> str:
    """Position identity without move counters."""
    return " ".join(fen.split()[:4])


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            digest.update(line)
    temporary.replace(path)
    return digest.hexdigest()


def _jsonl_sources(patterns: list[str]) -> list[Path]:
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        if not matches:
            raise FileNotFoundError(f"no JSONLs matched: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _open_jsonl(path: Path):
    if path.name.endswith(".zst"):
        import zstandard

        raw = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def open_pgn(path: str):
    if path.endswith(".zst"):
        import zstandard
        f = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(f)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_games(path: str, max_games: int = None):
    """Yield valid games (non-empty, no parse errors) from a single PGN file."""
    with open_pgn(path) as handle:
        count = 0
        while True:
            if max_games is not None and count >= max_games:
                break
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if game.errors or not list(game.mainline_moves()):
                continue
            yield game
            count += 1


def iter_games_glob(pgn_glob: str, max_games: int = None):
    """Chain iter_games across every PGN matching the glob, sharing one cap."""
    paths = sorted(glob(pgn_glob))
    if not paths:
        sys.exit(f"no PGNs matched: {pgn_glob}")
    n = 0
    for path in paths:
        for game in iter_games(path):
            if max_games is not None and n >= max_games:
                return
            n += 1
            yield game


def _scan_game(game: chess.pgn.Game, history_plies: int = HISTORY_PLIES):
    """Replay a game; return (moves_all, boards, L) or None if shorter than
    START_PLY. `boards[ply]` is a board snapshot for every ply in
    [START_PLY - history_plies, L]: plies >= START_PLY are gate candidates;
    the lower plies only backstop window start_fens. Snapshots use board.copy()
    (cheap), so no FEN is rendered until a position is actually chosen."""
    moves_all = [m.uci() for m in game.mainline_moves()]
    L = len(moves_all)
    if L < START_PLY:
        return None
    keep_from = max(0, START_PLY - history_plies)
    board = chess.Board(game.headers.get("FEN", chess.STARTING_FEN))
    boards = {0: board.copy(stack=False)} if keep_from == 0 else {}
    for i, uci in enumerate(moves_all):
        board.push_uci(uci)
        ply = i + 1
        if ply >= keep_from:
            boards[ply] = board.copy(stack=False)
    return moves_all, boards, L


def _window(moves_all, boards, ply, lookahead=0):
    """Triple [start_fen, moves(<=HISTORY_PLIES), end_fen] for the presented
    position Q = `ply` - `lookahead`. With lookahead>0 the gated destination is
    `ply` (= P) and two fields are appended: the k UCI moves Q->P and P's FEN.
    lookahead=0 returns the plain 3-element triple ending at `ply`."""
    q = ply - lookahead
    start = max(0, q - HISTORY_PLIES)
    window = [boards[start].fen(), moves_all[start:q], boards[q].fen()]
    if lookahead:
        window += [moves_all[q:ply], boards[ply].fen()]
    return window


def _motif_record(moves_all: list[str], boards: dict[int, chess.Board],
                  ply: int, game_id: str, motifs: list[dict]) -> dict:
    """Build and replay-validate one structured motif-mining record."""
    start = max(0, ply - MOTIF_CONTEXT_PLIES)
    moves = moves_all[start:ply]
    start_fen = boards[start].fen()
    target_fen = boards[ply].fen()
    replay = chess.Board(start_fen)
    for move in moves:
        replay.push_uci(move)
    if replay.fen() != target_fen:
        raise ValueError(f"motif context does not reproduce ply {ply}")
    if not motifs:
        raise ValueError("motif records require at least one motif")
    return {
        "schema_version": 1,
        "start_fen": start_fen,
        "moves": moves,
        "fen": target_fen,
        "source": {"game_id": game_id, "ply": ply},
        "motifs": motifs,
    }


def motif_prompt_window(record: dict, prompt_plies: int) -> list:
    """Convert a mined record to the established lookahead five-field window.

    The returned fields are `[history_start_fen, history_moves, prompt_fen,
    prompt_moves, motif_fen]`.  The history is the at-most-seven plies directly
    before the prompt root; the prompt moves are the requested suffix ending at
    the motif position.
    """
    moves = list(record["moves"])
    if prompt_plies < 0 or prompt_plies > 8:
        raise ValueError("prompt_plies must be in [0, 8]")
    if prompt_plies > len(moves):
        raise ValueError("motif record has too little context for the prompt")
    first = max(0, len(moves) - prompt_plies - HISTORY_PLIES)
    split = len(moves) - prompt_plies
    board = chess.Board(record["start_fen"])
    for move in moves[:first]:
        board.push_uci(move)
    history_start_fen = board.fen()
    history_moves = moves[first:split]
    for move in history_moves:
        board.push_uci(move)
    prompt_fen = board.fen()
    prompt_moves = moves[split:]
    for move in prompt_moves:
        board.push_uci(move)
    if board.fen() != record["fen"]:
        raise ValueError("motif prompt window does not reproduce its target")
    return [history_start_fen, history_moves, prompt_fen,
            prompt_moves, record["fen"]]


def _select_motif_positions(candidates: list[tuple[int, list[dict]]],
                            rng: random.Random, common_per_game: int,
                            rare_keep_probability: float
                            = RARE_POSITIONAL_KEEP_PROBABILITY):
    """Apply the motif-stage per-game selection policy.

    Returns `(selected, counts)`, where selected entries retain their original
    `(ply, motifs)` shape. Selection classes are mutually exclusive: tactical
    first, then the rare-position coin flip, then the capped common pool.
    """
    forced = []
    common = []
    counts = Counter()
    for candidate in candidates:
        _ply, motifs = candidate
        useful = [motif for motif in motifs
                  if motif["motif"] != "material_imbalance"]
        if not useful:
            counts["material_only_rejected"] += 1
            continue
        if any(motif["metadata"]["source"] == "tactical"
               for motif in useful):
            forced.append(candidate)
            counts["tactical_forced"] += 1
        elif (any(motif["motif"] in RARE_POSITIONAL_MOTIFS
                  for motif in useful)
              and rng.random() < rare_keep_probability):
            forced.append(candidate)
            counts["rare_positional_forced"] += 1
        else:
            common.append(candidate)
    sampled = rng.sample(common, min(common_per_game, len(common)))
    counts["common_sampled"] = len(sampled)
    return sorted(forced + sampled, key=lambda item: item[0]), counts


# ---------------------------------------------------------------------------
# Feature gates: each whittles a list of candidate plies down to those that pass.
# ---------------------------------------------------------------------------

def _position_gate(pred):
    """Keep plies whose board satisfies a position predicate."""
    return lambda plies, boards, L: [p for p in plies if pred(boards[p])]


def _near_mate_gate(near_plies):
    """Game-level: keep plies in [L - near_plies, L) when the game ends in mate
    (excludes the mate itself, which is its own category)."""
    def gate(plies, boards, L):
        if not boards[L].is_checkmate():
            return []
        lo = max(START_PLY, L - near_plies)
        return [p for p in plies if lo <= p < L]
    return gate


def _make_gate(name: str, near_plies: int):
    if name == "near_mate":
        return _near_mate_gate(near_plies)
    if name not in pf.FEATURES:
        sys.exit(f"unknown feature {name!r}; known: {sorted(pf.FEATURES)} + 'near_mate'")
    return _position_gate(pf.FEATURES[name])


def _lookahead_bounds(lookahead) -> tuple[int, int]:
    """Normalize the lookahead spec to (lo, hi). Falsy/0 -> (0, 0) (off); int k
    -> fixed (k, k); [lo, hi] -> random per position. Validated here."""
    if not lookahead:
        return 0, 0
    if isinstance(lookahead, (list, tuple)):
        lo, hi = int(lookahead[0]), int(lookahead[1])
    else:
        lo = hi = int(lookahead)
    if lo < 1 or hi < lo:
        sys.exit(f"invalid lookahead_ply {lookahead!r}; want k>=1 or [lo,hi], 1<=lo<=hi")
    return lo, hi


def run_verdict(pgn: str, output: str, cfg: dict, rng: random.Random) -> None:
    """Verdict-mode sampling: per game, draw `per_game` random plies (>= START_PLY,
    with the previous move as the tactic setup) and keep those the stage-5.a class
    acceptances admit (datagen.tree.strong_primitives, seeded per position so the
    stage-5.a builder re-derives the same class). A running per-class cap of
    `cap_per_million` rows per million candidates keeps the common classes from
    swamping the mix. Output rows are the standard triples."""
    import math

    from datagen.sim.tactics import Engines
    from datagen.tree.strong_primitives import CAP_PER_M, classify, primary_class

    eng = Engines()
    per_game = cfg.get("per_game", 1)
    cap_per_m = cfg.get("cap_per_million", CAP_PER_M)
    games = scanned = written = 0
    counts: dict = {}
    with open(output, "w") as out:
        for game in iter_games_glob(pgn, cfg.get("max_games")):
            scanned_game = _scan_game(game)
            if scanned_game is None:
                continue
            games += 1
            moves_all, boards, L = scanned_game
            plies = rng.sample(range(START_PLY, L + 1),
                               min(per_game, L + 1 - START_PLY))
            for ply in plies:
                scanned += 1
                before = boards[ply - 1]
                setup = chess.Move.from_uci(moves_all[ply - 1])
                res = classify(eng, before, setup)
                if res is None:
                    continue
                cls = primary_class(res["motifs"])
                cap = max(1, math.ceil(cap_per_m * scanned / 1e6))
                if counts.get(cls, 0) >= cap:
                    continue
                counts[cls] = counts.get(cls, 0) + 1
                out.write(json.dumps(_window(moves_all, boards, ply)) + "\n")
                written += 1
            if games % 2000 == 0:
                print(f"  {games:,} games | {scanned:,} candidates | "
                      f"{written:,} positions | {counts}", file=sys.stderr)
    print(f"Done. {games:,} games, {scanned:,} candidates. Wrote {written:,} "
          f"positions to {output}; classes: {counts}", file=sys.stderr)


def run_motifs(pgn: str, output: str, cfg: dict, rng: random.Random) -> None:
    """Scan every eligible ply, then apply the motif-stage selection policy.

    Positional/material detection is threshold-based and tactical detection
    retains the verdict pipeline's 1k screen plus 10k clear-best filter.  The
    engine bundle is process-local and lazy, so one sampler reuses its pools.
    """
    from datagen.sim.tactics import Engines
    from datagen.tree.motifs import motifs_for_game_ply

    engines = Engines()
    common_per_game = int(cfg["per_game"])
    rare_keep_probability = float(cfg.get(
        "rare_keep_probability", RARE_POSITIONAL_KEEP_PROBABILITY
    ))
    max_positions = int(cfg.get("max_positions", 0))
    if common_per_game < 1:
        raise ValueError("motif mode requires per_game >= 1")
    if not 0.0 <= rare_keep_probability <= 1.0:
        raise ValueError("rare_keep_probability must be in [0, 1]")
    if max_positions < 0:
        raise ValueError("max_positions must be nonnegative")

    games = scanned = matched = written = 0
    counts = Counter()
    selection_counts = Counter()
    with open(output, "w") as out:
        for game in iter_games_glob(pgn, cfg.get("max_games")):
            if max_positions and written >= max_positions:
                break
            scanned_game = _scan_game(game, history_plies=MOTIF_CONTEXT_PLIES)
            if scanned_game is None:
                continue
            games += 1
            moves_all, boards, length = scanned_game
            digest_input = boards[0].fen() + "\0" + " ".join(moves_all)
            game_id = hashlib.sha256(digest_input.encode()).hexdigest()[:20]
            candidates = []
            for ply in range(START_PLY, length + 1):
                scanned += 1
                before = boards[ply - 1]
                setup = chess.Move.from_uci(moves_all[ply - 1])
                motifs = motifs_for_game_ply(engines, before, setup)
                if motifs:
                    matched += 1
                    candidates.append((ply, motifs))

            selected, game_selection = _select_motif_positions(
                candidates,
                rng,
                common_per_game,
                rare_keep_probability,
            )
            selection_counts.update(game_selection)
            for ply, motifs in selected:
                record = _motif_record(moves_all, boards, ply, game_id, motifs)
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
                counts.update(motif["motif"] for motif in motifs)
            if games % 100 == 0:
                print(f"  {games:,} games | {scanned:,} plies | "
                      f"{matched:,} matched | {written:,} written | "
                      f"selection={dict(selection_counts)} | motifs={dict(counts)}",
                      file=sys.stderr)
    print(f"Done. {games:,} games, {scanned:,} plies, {matched:,} matches. "
          f"Wrote {written:,} positions to {output}; "
          f"selection: {dict(selection_counts)}; motifs: {dict(counts)}",
          file=sys.stderr)


def run_simulate(pgn: str, output: str, cfg: dict, rng: random.Random) -> None:
    """Simulate-mode sampling: per source game, draw `per_game` random plies
    (>= START_PLY, not game-over), play `n_games` engine-pool games from each
    position across a shared `workers`-process pool, and write
    [start_fen, moves, end_fen, {"games": [...]}] — appended and flushed per
    position, resuming past end-fens already present in the output."""
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from datagen.sim import games as G

    per_game = cfg.get("per_game", 1)
    n_games = cfg.get("n_games", G.N_GAMES)
    workers = cfg.get("workers", 16)
    max_positions = cfg.get("max_positions", 0)
    sim_seed = cfg.get("sim_seed", 0)   # pairing schedule (shared by all positions)

    done = set()
    if Path(output).exists():
        with open(output) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)[2])
                except Exception:
                    pass                # a line torn by a kill mid-write
    if done:
        print(f"  resuming: {len(done)} positions already done", file=sys.stderr)

    games_scanned = written = len(done)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex, open(output, "a") as out:
        for game in iter_games_glob(pgn, cfg.get("max_games")):
            if max_positions and written >= max_positions:
                break
            scanned = _scan_game(game)
            if scanned is None:
                continue
            games_scanned += 1
            moves_all, boards, L = scanned
            plies = [p for p in range(START_PLY, L + 1)
                     if not boards[p].is_game_over(claim_draw=True)]
            for ply in rng.sample(plies, min(len(plies), per_game)):
                window = _window(moves_all, boards, ply)
                if window[2] in done:
                    continue
                if max_positions and written >= max_positions:
                    break
                gs = G.play_games(window[2], n_games, seed=sim_seed, executor=ex)
                out.write(json.dumps(window + [{"games": gs}]) + "\n")
                out.flush()
                written += 1
                el = time.time() - t0
                print(f"  {written:,} positions | {games_scanned:,} games scanned"
                      f" | {3600 * (written - len(done)) / max(el, 1e-9):.1f} pos/hour",
                      file=sys.stderr)
    print(f"Done. Wrote {written:,} positions (with {n_games} games each) to "
          f"{output}", file=sys.stderr)


def run(pgn: str, output: str, cfg: dict, rng: random.Random, lookahead=0) -> None:
    """Enumerate each game's positions, apply the configured feature gates, and
    write min(available, per_game) random survivors. `lookahead` (int or [lo,hi])
    gates the destination ply P as usual but emits the position Q = P - k plus the
    saved P-bound sequence, with k fixed or drawn per position; the only gating
    change is requiring P - lo >= START_PLY (so Q stays in the move window)."""
    lo, hi = _lookahead_bounds(lookahead)
    near_plies = cfg.get("near_plies", NEAR_PLIES)
    gates = [_make_gate(f, near_plies) for f in (cfg.get("features") or [])]
    per_game = cfg["per_game"]
    games = written = 0
    with open(output, "w") as out:
        for game in iter_games_glob(pgn, cfg.get("max_games")):
            scanned = _scan_game(game)
            if scanned is None:
                continue
            games += 1
            moves_all, boards, L = scanned
            plies = list(range(START_PLY + lo, L + 1))
            for gate in gates:
                plies = gate(plies, boards, L)
                if not plies:
                    break
            for ply in rng.sample(plies, min(len(plies), per_game)):
                k = rng.randint(lo, min(hi, ply - START_PLY)) if lo else 0
                out.write(json.dumps(_window(moves_all, boards, ply, k)) + "\n")
                written += 1
            if games % 20000 == 0:
                print(f"  {games:,} games | {written:,} positions", file=sys.stderr)
    print(f"Done. {games:,} games scanned. Wrote {written:,} positions to {output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Self-distillation seed sources.
# ---------------------------------------------------------------------------

def _puzzle_buckets(cfg: dict) -> list[dict]:
    specs = cfg.get("rating_buckets")
    if not specs:
        raise ValueError("puzzle config requires a non-empty rating_buckets list")
    names = set()
    result = []
    for source in specs:
        spec = dict(source)
        name = str(spec.get("name", "")).strip()
        count = int(spec.get("count", 0))
        lower = spec.get("min_exclusive")
        upper = spec.get("max_inclusive")
        if not name or name in names or count <= 0:
            raise ValueError(f"invalid puzzle rating bucket: {source}")
        if lower is None and upper is None:
            raise ValueError(f"rating bucket {name!r} has no bounds")
        lower = -math.inf if lower is None else int(lower)
        upper = math.inf if upper is None else int(upper)
        if lower >= upper:
            raise ValueError(f"invalid bounds for rating bucket {name!r}")
        names.add(name)
        result.append({"name": name, "count": count, "lower": lower, "upper": upper})
    ordered = sorted(result, key=lambda row: (row["lower"], row["upper"]))
    for left, right in zip(ordered, ordered[1:]):
        if left["upper"] > right["lower"]:
            raise ValueError(
                f"overlapping puzzle rating buckets {left['name']!r} and {right['name']!r}"
            )
    return result


def _puzzle_bucket(rating: int, buckets: list[dict]) -> dict | None:
    for bucket in buckets:
        if bucket["lower"] < rating <= bucket["upper"]:
            return bucket
    return None


def _puzzle_history(source: dict) -> list[str]:
    """Convert a raw puzzle's setup plies to prior-FEN encoder history."""
    meta = source["meta"]
    board = chess.Board(meta["original_fen"])
    history = []
    setup = list(source.get("history") or [])
    for uci in setup:
        history.append(board.fen())
        board.push_uci(uci)
    if normalized_fen(board.fen()) != normalized_fen(source["fen"]):
        raise ValueError("puzzle setup history does not reach its root FEN")
    return history[-HISTORY_PLIES:]


def _puzzle_motif_record(source: dict, motifs: list[dict], source_path: Path,
                         source_line: int) -> dict:
    """Build a motif record rooted after a Lichess puzzle's setup move."""
    meta = source["meta"]
    puzzle = meta["puzzle"]
    moves = list(source.get("history") or [])
    if not moves:
        raise ValueError("puzzle motif record has no setup move")
    board = chess.Board(meta["original_fen"])
    for move in moves:
        board.push_uci(move)
    if normalized_fen(board.fen()) != normalized_fen(source["fen"]):
        raise ValueError("puzzle setup history does not reach its root FEN")
    if not motifs:
        raise ValueError("puzzle motif records require at least one motif")
    return {
        "schema_version": 1,
        "start_fen": meta["original_fen"],
        "moves": moves,
        "fen": source["fen"],
        "source": {
            "kind": "lichess_puzzle",
            "puzzle_id": puzzle["id"],
            "rating": int(puzzle["rating"]),
            "themes": list(puzzle.get("themes") or []),
            "solution_uci": list(puzzle.get("moves") or [])[1:],
            "source_path": str(source_path),
            "source_line": source_line,
            "source_idx": source.get("idx"),
        },
        "motifs": motifs,
    }


def run_puzzle_motifs(patterns: list[str], output: str, cfg: dict,
                      max_puzzles: int | None = None,
                      skip_puzzles: int | None = None) -> None:
    """Mine stage-5 motifs at post-setup Lichess puzzle roots.

    This intentionally calls ``motifs_for_game_ply`` rather than trusting the
    Lichess theme labels: positional thresholds and the 1k/10k tactical filters
    therefore remain identical to PGN motif mining. Material-only roots are
    rejected under the same policy as ordinary game positions.
    """
    from datagen.sim.tactics import Engines
    from datagen.tree.motifs import motifs_for_game_ply

    paths = _jsonl_sources(patterns)
    limit = int(cfg.get("max_puzzles", 0) if max_puzzles is None else max_puzzles)
    skip = int(cfg.get("skip_puzzles", 0) if skip_puzzles is None else skip_puzzles)
    if limit < 0:
        raise ValueError("max_puzzles must be nonnegative")
    if skip < 0:
        raise ValueError("skip_puzzles must be nonnegative")
    engines = Engines()
    skipped = scanned = written = tactical_rows = 0
    counts = Counter()
    rejected = Counter()
    started = time.monotonic()
    with open(output, "w") as out:
        for path in paths:
            with _open_jsonl(path) as handle:
                for source_line, line in enumerate(handle, 1):
                    if skipped < skip:
                        skipped += 1
                        continue
                    if limit and scanned >= limit:
                        break
                    source = json.loads(line)
                    scanned += 1
                    setup_moves = list(source.get("history") or [])
                    if not setup_moves:
                        rejected["missing_setup"] += 1
                        continue
                    before = chess.Board(source["meta"]["original_fen"])
                    try:
                        for uci in setup_moves[:-1]:
                            before.push_uci(uci)
                        setup = chess.Move.from_uci(setup_moves[-1])
                    except (ValueError, chess.IllegalMoveError):
                        rejected["invalid_setup"] += 1
                        continue
                    motifs = motifs_for_game_ply(engines, before, setup)
                    useful = [motif for motif in motifs
                              if motif["motif"] != "material_imbalance"]
                    if not useful:
                        rejected["empty_or_material_only"] += 1
                        continue
                    record = _puzzle_motif_record(
                        source, motifs, path, source_line
                    )
                    out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    written += 1
                    names = [motif["motif"] for motif in motifs]
                    counts.update(names)
                    if any(motif["metadata"]["source"] == "tactical"
                           for motif in motifs):
                        tactical_rows += 1
                    if scanned % 1000 == 0:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        print(
                            f"  {scanned:,} puzzles | {written:,} written | "
                            f"{tactical_rows:,} tactical | {scanned / elapsed:.1f} pos/s",
                            file=sys.stderr,
                        )
            if limit and scanned >= limit:
                break
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"Done. Scanned {scanned:,} puzzles in {elapsed / 3600:.2f}h "
        f"({scanned / elapsed:.1f} pos/s). Wrote {written:,} positions, "
        f"including {tactical_rows:,} with tactical motifs, to {output}; "
        f"rejected: {dict(rejected)}; motifs: {dict(counts)}",
        file=sys.stderr,
    )


def run_puzzles(
    patterns: list[str],
    output: str,
    cfg: dict,
    rng: random.Random,
    seed: int,
) -> None:
    """Stream and reservoir-sample rating-stratified puzzle JSONLs."""
    # puzzle setup: resolve inputs and create one fixed-size reservoir per rating bin.
    paths = _jsonl_sources(patterns)
    buckets = _puzzle_buckets(cfg)
    reservoirs = {bucket["name"]: [] for bucket in buckets}
    available = Counter()
    counts = Counter()
    seen = set()

    # puzzle scan: validate each position and reservoir-sample it into its rating bin.
    for path in paths:
        with _open_jsonl(path) as handle:
            for line_number, line in enumerate(handle, 1):
                counts["rows_scanned"] += 1
                source = json.loads(line)
                board = chess.Board(source["fen"])
                meta = source["meta"]
                puzzle = meta["puzzle"]
                rating = int(puzzle["rating"])
                bucket = _puzzle_bucket(rating, buckets)
                moves = list(puzzle["moves"])
                if bucket is None:
                    counts["outside_rating_buckets"] += 1
                    continue
                if len(moves) < 2:
                    raise ValueError("puzzle has no solution move")
                if board.is_game_over(claim_draw=False):
                    counts["terminal"] += 1
                    continue
                history = _puzzle_history(source)
                key = normalized_fen(source["fen"])
                if key in seen:
                    counts["duplicate_position"] += 1
                    continue
                seen.add(key)
                name = bucket["name"]
                available[name] += 1
                record = {
                    "record_id": f"puzzle_{puzzle['id']}",
                    "fen": source["fen"],
                    "history": history,
                    "extra": {
                        "source_kind": "puzzle",
                        "source_path": str(path),
                        "source_line": line_number,
                        "source_idx": source.get("idx"),
                        "normalized_fen": key,
                        "puzzle_id": puzzle["id"],
                        "rating": rating,
                        "rating_bin": name,
                        "themes": list(puzzle.get("themes") or []),
                        "setup_move_uci": moves[0],
                        "correct_move_uci": moves[1],
                        "solution_uci": moves[1:],
                        "puzzle": puzzle,
                    },
                }
                target = bucket["count"]
                reservoir = reservoirs[name]
                if len(reservoir) < target:
                    reservoir.append(record)
                else:
                    replacement = rng.randrange(available[name])
                    if replacement < target:
                        reservoir[replacement] = record

    # puzzle validation: require every requested rating bin to contain enough positions.
    short = {
        bucket["name"]: {
            "available": available[bucket["name"]],
            "requested": bucket["count"],
        }
        for bucket in buckets
        if available[bucket["name"]] < bucket["count"]
    }
    if short:
        raise RuntimeError(f"puzzle rating buckets are short: {short}")
    # puzzle output: combine the bins, shuffle them, and write data plus provenance.
    rows = [
        record
        for bucket in buckets
        for record in reservoirs[bucket["name"]]
    ]
    rng.shuffle(rows)
    output_path = Path(output).resolve()
    digest = _write_jsonl(output_path, rows)
    manifest = {
        "mode": "puzzles",
        "seed": seed,
        "sources": [str(path) for path in paths],
        "rating_buckets": [
            {
                "name": bucket["name"],
                "min_exclusive": None if math.isinf(-bucket["lower"]) else bucket["lower"],
                "max_inclusive": None if math.isinf(bucket["upper"]) else bucket["upper"],
                "requested": bucket["count"],
                "available": available[bucket["name"]],
            }
            for bucket in buckets
        ],
        "counts": dict(counts),
        "unique_eligible_positions": len(seen),
        "output": {"path": str(output_path), "rows": len(rows), "sha256": digest},
    }
    _atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), file=sys.stderr)


def run_play(
    games: int,
    output: Path,
    job: str,
    cfg: dict,
    seed: int,
    model: Path,
    encoder: Path,
    stockfish: Path,
) -> Path:
    # play setup: translate the sampler configuration into the game-loop interface.
    from datagen.self_distill.get_play_positions import (
        PlayConfig,
        get_play_positions,
        cleanup_play_job,
    )
    play = PlayConfig(
        model=model.resolve(),
        encoder=encoder.resolve(),
        stockfish=stockfish.resolve(),
        games=games,
        seed=seed,
        job=job,
        scratch_dir=Path(cfg.get("scratch_dir", "data/scratch/self_distill_play")),
        game_start=int(cfg.get("game_start", 0)),
        concurrent_games=int(cfg.get("concurrent_games", min(128, games))),
        opponent_nodes_min=int(cfg.get("opponent_nodes_min", 100)),
        opponent_nodes_max=int(cfg.get("opponent_nodes_max", 100_000)),
        stockfish_workers=int(cfg.get("stockfish_workers", 6)),
        max_plies=int(cfg.get("max_plies", 200)),
        max_output_tokens=int(cfg.get("max_output_tokens", 4096)),
        temperature=float(cfg.get("temperature", 0.6)),
        top_k=int(cfg.get("top_k", 20)),
        top_p=float(cfg.get("top_p", 0.95)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.78)),
        max_num_seqs=int(cfg.get("max_num_seqs", 128)),
        use_v1_vllm=bool(cfg.get("use_v1_vllm", False)),
        start_fen=str(cfg.get("start_fen", chess.STARTING_FEN)),
    )
    result = get_play_positions(play, output)
    cleanup_play_job(play, result)
    return result


def run_interactive(pgn: str, max_games: int, seed: int) -> None:
    rng = random.Random(seed)
    chosen = None
    for k, game in enumerate(iter_games_glob(pgn, max_games=max_games)):
        if rng.random() < 1.0 / (k + 1):
            chosen = game
    if chosen is None:
        sys.exit("No games found.")
    scanned = _scan_game(chosen)
    if scanned is None:
        sys.exit("Sampled game is shorter than START_PLY; try again.")
    moves_all, boards, L = scanned
    ply = rng.randint(START_PLY, L)
    board = boards[ply]
    h = chosen.headers
    print(f"Game   : {h.get('White', '?')} vs {h.get('Black', '?')}")
    print(f"Result : {h.get('Result', '?')}    Event: {h.get('Event', '?')}")
    print(f"Plies  : {L}   sampled ply: {ply}")
    print(f"Position FEN: {board.fen()}\n")
    print(board)
    print(f"\nSide to move: {'White' if board.turn == chess.WHITE else 'Black'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Feature-gated position sampler (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pgn", nargs="?", help="PGN path or glob (.pgn / .pgn.zst)")
    sources = p.add_mutually_exclusive_group()
    sources.add_argument(
        "--puzzles",
        nargs="+",
        metavar="JSONL",
        help="Puzzle JSONL path(s) or glob(s), sampled by rating bucket",
    )
    sources.add_argument(
        "--play-games",
        type=int,
        metavar="N",
        help="Play N model-vs-Stockfish games and publish a .games.jsonl shard",
    )
    sources.add_argument("--score-play", type=Path, metavar="PATH", help="Process a .games.jsonl shard into .scored.jsonl")
    p.add_argument("--config", help="Sampler config YAML (features + per_game)")
    p.add_argument("--output", default=None, help="Output JSONL path; .games.jsonl for gameplay")
    p.add_argument("--job", help="Caller-assigned gameplay scratch subdirectory (or play.job)")
    p.add_argument("--seed", type=int, default=None, help="Overrides config 'seed' (default 42)")
    p.add_argument(
        "--max-puzzles",
        type=int,
        default=None,
        help="Maximum puzzle roots to scan in puzzle motif mode",
    )
    p.add_argument(
        "--skip-puzzles",
        type=int,
        default=None,
        help="Puzzle roots to skip before scanning in puzzle motif mode",
    )
    p.add_argument("--model", type=Path, help="Merged Flamingo checkpoint for --play-games")
    p.add_argument("--use-v1-vllm", action=argparse.BooleanOptionalAction, default=None,
                   help="Use the legacy hybrid runner for gameplay (default: V2)")
    p.add_argument("--encoder", type=Path, help="LC0 encoder for --play-games")
    p.add_argument("--stockfish", type=Path, help="Stockfish binary for gameplay or scoring")
    p.add_argument("--lookahead_ply", type=int, default=None,
                   help="Forward (stage-3/4) data: present the position k ply before the "
                        "gated one and save the k-ply sequence + final FEN. Overrides config "
                        "'lookahead_ply' (default 0 = plain triples).")
    p.add_argument("--interactive", action="store_true",
                   help="Inspect one random position and exit (ignores --config).")
    args = p.parse_args()

    if args.use_v1_vllm is not None and args.play_games is None:
        p.error("--use-v1-vllm applies only to --play-games")

    selected_sources = sum((args.pgn is not None, bool(args.puzzles), args.play_games is not None, args.score_play is not None))
    if selected_sources != 1:
        p.error("choose exactly one source: positional pgn, --puzzles, --play-games, or --score-play")

    if args.interactive:
        if args.pgn is None:
            p.error("--interactive requires a positional PGN source")
        run_interactive(args.pgn, max_games=1000, seed=args.seed)
        return

    if not args.config:
        p.error("--config is required (or use --interactive)")
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    if args.play_games is not None and not (args.output or cfg.get("output")):
        p.error("--play-games requires --output (or config output) ending in .games.jsonl")
    if args.score_play is not None and args.output is not None:
        p.error("--score-play derives its .scored.jsonl output from the input filename")
    args.output = args.output or cfg.get("output", "positions.jsonl")
    if args.score_play is not None:
        from datagen.self_distill.score_play_positions import run_score
        score_cfg = cfg.get("score", {})
        stockfish = args.stockfish or score_cfg.get("stockfish")
        if stockfish is None:
            p.error("--stockfish (or score.stockfish) is required")
        run_score(args.score_play, score_cfg, Path(stockfish))
        return
    if args.puzzles:
        puzzle_cfg = cfg.get("puzzles", cfg)
        if puzzle_cfg.get("motifs"):
            run_puzzle_motifs(
                args.puzzles,
                args.output,
                puzzle_cfg,
                args.max_puzzles,
                args.skip_puzzles,
            )
        else:
            if args.max_puzzles is not None or args.skip_puzzles is not None:
                p.error("--max-puzzles/--skip-puzzles require puzzles.motifs=true")
            run_puzzles(
                args.puzzles,
                args.output,
                puzzle_cfg,
                random.Random(seed),
                seed,
            )
        return
    if args.play_games is not None:
        if args.play_games <= 0:
            p.error("--play-games must be positive")
        play_cfg = cfg.get("play", cfg)
        if args.use_v1_vllm is not None:
            play_cfg = {**play_cfg, "use_v1_vllm": args.use_v1_vllm}

        def required_path(name: str) -> Path:
            value = getattr(args, name) or play_cfg.get(name)
            if value is None:
                p.error(f"--{name} (or play.{name} in the config) is required")
            return Path(value)

        job = args.job or play_cfg.get("job")
        if not job:
            p.error("--job (or play.job) is required for gameplay")
        output = run_play(
            args.play_games,
            Path(args.output),
            job,
            play_cfg,
            seed,
            required_path("model"),
            required_path("encoder"),
            required_path("stockfish"),
        )
        print(output)
        return

    if "per_game" not in cfg:
        p.error("config missing required field: 'per_game'")
    # CLI (fixed int) overrides config, which may be an int or a [lo, hi] range.
    lookahead = args.lookahead_ply if args.lookahead_ply is not None else cfg.get("lookahead_ply", 0)
    special_modes = [name for name in ("verdict", "simulate", "motifs")
                     if cfg.get(name)]
    if len(special_modes) > 1:
        p.error(f"sampling modes are mutually exclusive: {', '.join(special_modes)}")
    if special_modes:
        if lookahead:
            p.error(f"{special_modes[0]} mode is incompatible with lookahead_ply")
        mode = {
            "verdict": run_verdict,
            "simulate": run_simulate,
            "motifs": run_motifs,
        }[special_modes[0]]
        mode(args.pgn, args.output, cfg, random.Random(seed))
    else:
        run(args.pgn, args.output, cfg, random.Random(seed), lookahead)


if __name__ == "__main__":
    main()
