"""Batched, resumable model-vs-Stockfish gameplay."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import json
import random
import signal
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess

from datagen.self_distill.analysis import (
    ANALYSIS_PROMPT, MOVE, field_map, legal_structured_best, pov_move,
)
from datagen.self_distill.play_engines import StockfishPool as _StockfishPool
from datagen.self_distill.play_storage import (
    PlayStore, fingerprint, publish_jsonl, scored_path, shard_lock,
)


_HISTORY_PLIES = 7


@dataclass(frozen=True, slots=True)
class PlayConfig:
    """Configuration for one independent gameplay process."""

    model: Path
    encoder: Path
    stockfish: Path
    games: int
    seed: int
    job: str
    scratch_dir: Path = Path("data/scratch/self_distill_play")
    game_start: int = 0
    concurrent_games: int = 128
    opponent_nodes_min: int = 100
    opponent_nodes_max: int = 100_000
    stockfish_workers: int = 6
    max_plies: int = 200
    max_output_tokens: int = 4096
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.78
    max_num_seqs: int = 128
    use_v1_vllm: bool = False
    start_fen: str = chess.STARTING_FEN

    def validate(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.game_start < 0:
            raise ValueError("game_start must be nonnegative")
        if not 0 < self.concurrent_games <= self.games:
            raise ValueError("concurrent_games must be in [1, games]")
        if self.opponent_nodes_min <= 0 or self.opponent_nodes_max < self.opponent_nodes_min:
            raise ValueError("invalid Stockfish opponent node range")
        if self.stockfish_workers <= 0:
            raise ValueError("stockfish_workers must be positive")
        if self.max_plies <= 0 or self.max_output_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("generation limits must be positive")
        if not 0.0 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in [0, 1]")
        chess.Board(self.start_fen)
        for path in (self.model, self.encoder, self.stockfish):
            if not path.exists():
                raise FileNotFoundError(path)


@dataclass(slots=True)
class _Game:
    index: int
    model_white: bool
    opponent_nodes: int
    board: chess.Board
    moves: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Played:
    fen: str
    history: tuple[str, ...]
    move: str
    mover: str
    game: int
    ply: int
    model_white: bool
    opponent_nodes: int
    move_source: str
    cached_root: dict | None


def _stable_seed(seed: int, game: int, ply: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{game}|{ply}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _opponent_nodes(config: PlayConfig, game: int) -> int:
    rng = random.Random(_stable_seed(config.seed, game, -1))
    return round(math.exp(rng.uniform(
        math.log(config.opponent_nodes_min),
        math.log(config.opponent_nodes_max),
    )))


def _history_fens(board: chess.Board) -> tuple[str, ...]:
    replay = board.root()
    positions = [replay.fen()]
    for move in board.move_stack:
        replay.push(move)
        positions.append(replay.fen())
    return tuple(positions[max(0, len(positions) - _HISTORY_PLIES - 1):-1])


def _choose_model_move(
    board: chess.Board,
    generation: str,
    seed: int,
) -> tuple[chess.Move, str]:
    """Best move, first critical move, promising moves, analysis, then random."""
    best = legal_structured_best(board.fen(), generation)
    if best is not None:
        return best, "best_move"
    fields = field_map(generation)
    # A variation is a sequence: never salvage a later move as a root choice.
    first = MOVE.search(fields.get("CRITICAL_LINE", ""))
    if first is not None:
        move, _ = pov_move(first, board, board.turn)
        if move is not None:
            return move, "critical_line"
    # These sections list independent root candidates, not a sequential PV.
    for name in ("PROMISING_MOVES", "ANALYSIS"):
        for atom in MOVE.finditer(fields.get(name, "")):
            move, _ = pov_move(atom, board, board.turn)
            if move is not None:
                return move, name.lower()
    return (
        random.Random(seed).choice(list(board.legal_moves)),
        "random_legal_fallback",
    )


def _finish_game(game: _Game, max_plies: int) -> dict | None:
    capped = game.board.ply() >= max_plies
    outcome = game.board.outcome(claim_draw=True)
    if outcome is None and not capped:
        return None
    if capped or outcome is None or outcome.winner is None:
        score = "1/2-1/2"
    else:
        score = "1-0" if outcome.winner else "0-1"
    termination = "ply_cap" if capped else outcome.termination.name.lower()
    return {
        "game": game.index,
        "model_white": game.model_white,
        "stockfish_nodes": game.opponent_nodes,
        "moves": list(game.moves),
        "result": score,
        "termination": termination,
    }


def gameplay_specification(config: PlayConfig) -> dict:
    """Identify gameplay independently of scratch paths and downstream scoring."""
    settings = asdict(config)
    if config.use_v1_vllm:
        settings.pop("use_v1_vllm")  # Legacy V1 gameplay receipts omitted this field.
    for key in ("scratch_dir", "job", "model", "encoder", "stockfish", "stockfish_workers"):
        settings.pop(key)
    versions = {}
    for package in ("torch", "transformers", "vllm", "chess"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "gameplay_policy_version": 2,
        "settings": settings,
        "prompt": ANALYSIS_PROMPT,
        "artifacts": {name: fingerprint(getattr(config, name))
                      for name in ("model", "encoder", "stockfish")},
        "packages": versions,
    }


def _load_generator(config: PlayConfig):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from models.vllm.flamingo_generate import ChessFlamingoGenerator

    return ChessFlamingoGenerator(
        config.model, config.encoder,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_output_tokens + 256,
        max_num_seqs=config.max_num_seqs, seed=config.seed, enforce_eager=True,
        use_v1_vllm=config.use_v1_vllm,
    )


@contextmanager
def _stop_after_cycle():
    """SIGTERM/SIGINT finish the current bounded cycle before returning control."""
    requested = [False]

    def stop(_signum, _frame):
        requested[0] = True

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _restore_game(row: dict, start_fen: str) -> _Game:
    board = chess.Board(start_fen)
    for move in row["moves"]:
        board.push_uci(move)
    return _Game(row["index"], row["model_white"], row["opponent_nodes"],
                 board, list(row["moves"]))


def get_play_positions(config: PlayConfig, output: Path) -> Path:
    """Resume caller-named gameplay and publish a self-contained .games.jsonl."""
    config.validate()
    output = output.resolve()
    specification = gameplay_specification(config)
    with shard_lock(output):
        if scored_path(output).exists():
            raise FileExistsError(f"shard already scored: {scored_path(output)}")
        if output.exists() and not (config.scratch_dir / config.job).exists():
            # The previous invocation may have exited after successful caller cleanup.
            with output.open() as handle:
                metadata = json.loads(next(handle))["metadata"]
                if metadata["specification"] != specification:
                    raise FileExistsError(f"raw shard belongs to a different gameplay run: {output}")
                counts = {"game": 0, "position": 0}
                for line in handle:
                    row = json.loads(line)
                    if len(row) != 1 or next(iter(row)) not in counts:
                        raise RuntimeError("invalid raw gameplay record")
                    counts[next(iter(row))] += 1
                if (counts["game"] != metadata["statistics"]["games"]
                        or counts["position"] != metadata["statistics"]["positions"]):
                    raise RuntimeError("raw shard is incomplete")
            return output
        with PlayStore(config.scratch_dir, config.job, specification) as store:
            if output.is_relative_to(store.path):
                raise ValueError("durable shard must be outside the job scratch directory")
            if output.exists():
                # Publication succeeded before a caller interruption. Verify it before cleanup.
                if not store.complete:
                    raise FileExistsError(f"output exists for incomplete job: {output}")
                _verify_games(output, store)
                return output
            if not store.complete:
                with _stop_after_cycle() as stop:
                    _play_games(config, store, stop)
                    if stop[0]:
                        raise InterruptedError("gameplay checkpointed; resume to publish")
            publish_jsonl(output, _game_records(store))
    return output


def _game_records(store):
    # A provenance header followed by game summaries and streamed position records.
    yield {"metadata": {"specification": store.specification, "statistics": store.state["statistics"]}}
    for game in store.records("games"):
        yield {"game": game}
    for position in store.records("positions"):
        yield {"position": position}


def _verify_games(path, store):
    from itertools import zip_longest
    with path.open() as handle:
        for actual, expected in zip_longest((json.loads(line) for line in handle), _game_records(store)):
            if actual != expected:
                raise RuntimeError("existing games shard differs from completed gameplay")


def cleanup_play_job(config: PlayConfig, output: Path) -> None:
    """Caller cleanup: verify the durable handoff before deleting exactly this job."""
    with shard_lock(output):
        if not (config.scratch_dir / config.job).exists():
            return
        with PlayStore(config.scratch_dir, config.job) as store:
            _verify_games(output, store)
            store.delete()


def _play_games(config: PlayConfig, store: PlayStore, stop: list[bool]) -> None:
    # recovery: reconstruct boards with their full move stacks for draw detection.
    state = store.state or {
        "next_game": config.game_start, "finished_games": 0, "active": [],
        "statistics": dict.fromkeys((
            "games", "positions", "model_positions", "stockfish_positions",
            "generated_tokens", "truncated_generations", "random_legal_fallbacks",
            "cycles", "model_load_seconds", "model_generation_seconds",
            "opponent_seconds", "wall_seconds",
        ), 0),
    }
    active = [_restore_game(row, config.start_fen) for row in state["active"]]
    next_game = state["next_game"]
    finished = state["finished_games"]
    stats = dict(state["statistics"])
    prior_seconds = stats["wall_seconds"]
    started = time.perf_counter()

    def checkpoint():
        stats["games"] = finished
        stats["wall_seconds"] = prior_seconds + time.perf_counter() - started
        store.checkpoint({
            "next_game": next_game, "finished_games": finished,
            "complete": finished == config.games,
            "active": [{"index": game.index, "model_white": game.model_white,
                        "opponent_nodes": game.opponent_nodes, "moves": game.moves}
                       for game in active],
            "statistics": stats,
        })

    def finish_terminals():
        nonlocal finished
        survivors = []
        for game in active:
            result = _finish_game(game, config.max_plies)
            if result is None:
                survivors.append(game)
            else:
                store.append("games", result)
                finished += 1
        active[:] = survivors

    def record_move(game, move, mover, source, cached_root=None):
        store.append("positions", asdict(_Played(
            fen=game.board.fen(), history=_history_fens(game.board),
            move=move.uci(), mover=mover, game=game.index, ply=game.board.ply(),
            model_white=game.model_white, opponent_nodes=game.opponent_nodes,
            move_source=source, cached_root=cached_root,
        )))
        game.board.push(move)
        game.moves.append(move.uci())
        stats["positions"] += 1
        stats[f"{mover}_positions"] += 1

    # Initialize durable empty outputs before any inference or appended records.
    checkpoint()
    pool = _StockfishPool(config.stockfish, config.stockfish_workers)
    generator = None
    try:
        while finished < config.games:
            if stop[0]:
                raise InterruptedError(f"gameplay checkpoint saved; resume artifact {store.artifact_id}")
            while len(active) < config.concurrent_games and next_game < config.game_start + config.games:
                active.append(_Game(next_game, next_game % 2 == 0,
                                    _opponent_nodes(config, next_game), chess.Board(config.start_fen)))
                next_game += 1

            # Opponent turns retain the same ordering and node budgets as before.
            finish_terminals()
            opponents = [game for game in active if game.board.turn != game.model_white]
            if opponents:
                phase_started = time.perf_counter()
                moves = pool.play([(game.board.fen(), game.opponent_nodes) for game in opponents])
                stats["opponent_seconds"] += time.perf_counter() - phase_started
                for game, uci in zip(opponents, moves, strict=True):
                    record_move(game, chess.Move.from_uci(uci), "stockfish", "stockfish")

            # Model turns: bounded batches, per-game/ply seeds, unchanged move selection.
            finish_terminals()
            if active and generator is None:
                phase_started = time.perf_counter()
                generator = _load_generator(config)
                stats["model_load_seconds"] += time.perf_counter() - phase_started
            for offset in range(0, len(active), config.max_num_seqs):
                chunk = active[offset:offset + config.max_num_seqs]
                phase_started = time.perf_counter()
                outputs = generator.generate(
                    [game.board.fen() for game in chunk], [ANALYSIS_PROMPT] * len(chunk),
                    [list(_history_fens(game.board)) for game in chunk],
                    temperature=config.temperature, top_k=config.top_k, top_p=config.top_p,
                    max_tokens=config.max_output_tokens,
                    seeds=[_stable_seed(config.seed, game.index, game.board.ply()) for game in chunk],
                )
                stats["model_generation_seconds"] += time.perf_counter() - phase_started
                for game, output in zip(chunk, outputs, strict=True):
                    move, source = _choose_model_move(
                        game.board, output["text"],
                        _stable_seed(config.seed, game.index, game.board.ply()),
                    )
                    record_move(game, move, "model", source, {
                        "generation": output["text"], "token_count": len(output["token_ids"]),
                        "finish_reason": output["finish_reason"],
                    })
                    stats["generated_tokens"] += len(output["token_ids"])
                    stats["truncated_generations"] += output["finish_reason"] == "length"
                    stats["random_legal_fallbacks"] += source == "random_legal_fallback"

            finish_terminals()
            stats["cycles"] += 1
            checkpoint()
            print(f"[play {stats['cycles']}] job={config.job} "
                  f"finished={finished}/{config.games} positions={stats['positions']:,}", flush=True)
    finally:
        pool.close()
