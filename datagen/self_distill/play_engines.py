"""Persistent one-thread engines shared by gameplay and CPU move auditing."""

import math
import queue
import threading
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

_MATE_SCORE = 100_000


def close_engines(engines) -> None:
    """Terminate every transport, including dead or partially started engines."""
    for engine in engines:
        with suppress(Exception):
            engine.close()


def start_engines(binary: Path, workers: int):
    if workers <= 0:
        raise ValueError("Stockfish workers must be positive")
    engines = []
    try:
        for _ in range(workers):
            engine = chess.engine.SimpleEngine.popen_uci(str(binary))
            engines.append(engine)
            engine.configure({"Threads": 1})
    except BaseException:
        close_engines(engines)
        raise
    return engines


def _expected_winrate(cp: int) -> float:
    exponent = max(-60.0, min(60.0, -0.00368208 * cp))
    return 1.0 / (1.0 + math.exp(exponent))


class StockfishPool:
    """A bounded set of persistent, one-thread Stockfish processes."""

    def __init__(self, binary: Path, workers: int):
        self.engines = start_engines(binary, workers)
        self.failed = threading.Event()
        self.available: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for engine in self.engines:
            self.available.put(engine)
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _with_engine(self, function, *args):
        if self.failed.is_set():
            raise RuntimeError("Stockfish pool failed; restart the task")
        engine = self.available.get()
        try:
            if self.failed.is_set():
                raise RuntimeError("Stockfish pool failed; restart the task")
            engine.configure({"Clear Hash": None})
            return function(engine, *args)
        except BaseException:
            self.failed.set()
            close_engines(self.engines)
            raise
        finally:
            self.available.put(engine)

    @staticmethod
    def _play(engine, fen: str, nodes: int) -> str:
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(nodes=nodes))
        if result.move is None:
            raise RuntimeError(f"Stockfish returned no move at {fen}")
        return result.move.uci()

    @staticmethod
    def _score(info: dict, turn: chess.Color) -> int:
        value = info["score"].pov(turn).score(mate_score=_MATE_SCORE)
        if value is None:
            raise RuntimeError("Stockfish returned a score without cp or mate")
        return value

    @classmethod
    def _audit(cls, engine, fen: str, played_uci: str, nodes: int) -> tuple[str, int, int, float]:
        board = chess.Board(fen)
        played = chess.Move.from_uci(played_uci)
        if played not in board.legal_moves:
            raise ValueError(f"played move {played_uci} is illegal at {fen}")
        limit = chess.engine.Limit(nodes=nodes)
        best_info = engine.analyse(board, limit)
        pv = best_info.get("pv") or []
        if not pv:
            raise RuntimeError(f"Stockfish returned no principal variation at {fen}")
        best = pv[0]
        best_cp = cls._score(best_info, board.turn)
        if played == best:
            played_cp = best_cp
        else:
            # Compare both moves in one restricted root search.  The total node
            # budget scales with MultiPV, matching the canonical oracle.
            engine.configure({"Clear Hash": None})
            compared = engine.analyse(
                board,
                chess.engine.Limit(nodes=nodes * 2),
                multipv=2,
                root_moves=[best, played],
            )
            by_move = {
                info["pv"][0]: info
                for info in compared
                if info.get("pv")
            }
            missing = [move for move in (best, played) if move not in by_move]
            for move in missing:
                engine.configure({"Clear Hash": None})
                by_move[move] = engine.analyse(
                    board,
                    limit,
                    root_moves=[move],
                )
            best_cp = cls._score(by_move[best], board.turn)
            played_cp = cls._score(by_move[played], board.turn)
        drop = max(0.0, _expected_winrate(best_cp) - _expected_winrate(played_cp))
        return best.uci(), best_cp, played_cp, drop

    def play(self, requests: list[tuple[str, int]]) -> list[str]:
        futures = [
            self.executor.submit(self._with_engine, self._play, fen, nodes)
            for fen, nodes in requests
        ]
        return [future.result() for future in futures]

    def audit(
        self,
        requests: list[tuple[str, str]],
        nodes: int,
    ) -> list[tuple[str, int, int, float]]:
        futures = [
            self.executor.submit(
                self._with_engine, self._audit, fen, played, nodes
            )
            for fen, played in requests
        ]
        return [future.result() for future in futures]

    def close(self) -> None:
        self.failed.set()
        close_engines(self.engines)
        self.executor.shutdown(wait=True, cancel_futures=True)


