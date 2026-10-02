"""Players for the benchmark: one uniform interface over Stockfish (by node
budget) and lc0 networks (by weights file), plus a random-legal baseline that
anchors the bottom of the ladder.

``Player`` is the single extension point. Stockfish and the lc0 binary are both
UCI engines that differ only in argv/options, so they share ``UciPlayer``. A
future LLM move-picker becomes just another ``Player`` subclass — the match and
rating code never needs to know which kind it is.
"""
from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chess
import chess.engine

REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo-local symlinks (data/engines/... -> the shared engines tree) keep eval/
# self-contained; override with $STOCKFISH_BIN / $LC0_BIN.
DEFAULT_STOCKFISH_BIN = Path(
    os.environ.get("STOCKFISH_BIN", REPO_ROOT / "data/engines/stockfish_25080907_x64_avx2")
)
DEFAULT_LC0_BIN = Path(
    os.environ.get("LC0_BIN", REPO_ROOT / "data/engines/lc0/build/release/lc0")
)


class Player(ABC):
    """A named move-chooser. Subclass to add a new engine kind."""

    name: str
    kind: str = "player"

    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move: ...

    def new_game(self) -> None:
        """Reset per-game search state (e.g. transposition table)."""

    def close(self) -> None:
        """Release any subprocess / resources."""

    def __enter__(self) -> "Player":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class RandomPlayer(Player):
    """Uniform random legal move — the ladder floor (anchored to 0 Elo)."""

    kind = "random"

    def __init__(self, seed: int = 12345, name: str = "random_legal") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(list(board.legal_moves))


class UciPlayer(Player):
    """A UCI engine played to a fixed node budget. Covers Stockfish and lc0."""

    def __init__(
        self,
        name: str,
        kind: str,
        command: list[str],
        *,
        nodes: int,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.nodes = int(nodes)
        self._engine = chess.engine.SimpleEngine.popen_uci(command)
        if options:
            self._engine.configure(options)
        self._token = 0  # bumped per game so the engine treats each game as fresh

    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(
            board, chess.engine.Limit(nodes=self.nodes), game=self._token
        )
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move")
        return result.move

    def new_game(self) -> None:
        self._token += 1

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass


def stockfish(
    nodes: int,
    *,
    hash_mb: int = 64,
    threads: int = 1,
    binary: Path = DEFAULT_STOCKFISH_BIN,
    name: str | None = None,
) -> UciPlayer:
    """Stockfish at a fixed node count (deterministic, single-threaded)."""
    return UciPlayer(
        name or f"stockfish_n{nodes}",
        "stockfish",
        [str(binary)],
        nodes=nodes,
        options={"Threads": threads, "Hash": hash_mb},
    )


def lc0(
    weights: str | Path,
    *,
    nodes: int = 1,
    binary: Path = DEFAULT_LC0_BIN,
    name: str | None = None,
    extra_args: list[str] | None = None,
) -> UciPlayer:
    """An lc0 network from a weights file. nodes=1 is policy/value with no search."""
    weights = Path(weights)
    command = [str(binary), f"--weights={weights}", *(extra_args or [])]
    return UciPlayer(name or f"lc0_{weights.stem}", "lc0", command, nodes=nodes)


def build_player(spec: dict[str, Any]) -> Player:
    """Construct a Player from a config dict keyed by ``kind``."""
    kind = spec["kind"]
    if kind == "random":
        return RandomPlayer(seed=spec.get("seed", 12345), name=spec.get("name", "random_legal"))
    if kind == "stockfish":
        return stockfish(
            int(spec["nodes"]),
            hash_mb=spec.get("hash_mb", 64),
            threads=spec.get("threads", 1),
            name=spec.get("name"),
        )
    if kind == "lc0":
        return lc0(
            spec["weights"],
            nodes=int(spec.get("nodes", 1)),
            name=spec.get("name"),
            extra_args=spec.get("extra_args"),
        )
    raise ValueError(f"unknown engine kind: {kind!r}")
