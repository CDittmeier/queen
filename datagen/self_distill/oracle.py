"""Parallel, cached Stockfish searches for self-distillation."""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import chess
import chess.engine

from datagen.self_distill.analysis import expected_winrate
from datagen.self_distill.play_engines import start_engines, close_engines


MATE_SCORE = 100_000
CACHE_SIZE = 16_384  # Bound retained searches/PVs across long-running mining tasks.


class ParallelOracle:
    """Run independent one-thread Stockfish requests through a fixed pool."""

    def __init__(self, binary: Path, workers: int, nodes_per_move: int):
        self.workers = workers
        self.nodes_per_move = nodes_per_move
        self.engines = start_engines(binary, workers)
        self.failed = threading.Event()
        self.available: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()
        for engine in self.engines:
            self.available.put(engine)
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self.lock = threading.Lock()
        self.query_count = 0
        self.node_budget = 0
        self.wall_seconds = 0.0

    @staticmethod
    def _cp(info: dict, turn: chess.Color) -> int:
        score = info["score"].pov(turn).score(mate_score=MATE_SCORE)
        if score is None:
            raise RuntimeError("Stockfish returned neither a centipawn nor mate score")
        return score

    def _search(
        self,
        engine: chess.engine.SimpleEngine,
        fen: str,
        multipv: int,
        moves: tuple[str, ...],
    ) -> list[dict]:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            cp = -MATE_SCORE if board.is_checkmate() else 0
            return [{"move": None, "cp": cp, "pv": []}]

        # search: compare all requested root moves in one shared-node MultiPV call.
        root_moves = [chess.Move.from_uci(move) for move in moves]
        engine.configure({"Clear Hash": None})
        infos = engine.analyse(
            board,
            chess.engine.Limit(nodes=self.nodes_per_move * max(1, multipv)),
            multipv=multipv,
            root_moves=root_moves or None,
        )
        rows = [
            {
                "move": info["pv"][0].uci() if info.get("pv") else None,
                "cp": self._cp(info, board.turn),
                "pv": [move.uci() for move in info.get("pv", [])],
            }
            for info in infos
        ]
        if not moves:
            return rows

        # recovery: force-search an occasionally omitted restricted root move.
        by_move = {row["move"]: row for row in rows}
        for move in moves:
            if move in by_move:
                continue
            engine.configure({"Clear Hash": None})
            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=self.nodes_per_move),
                root_moves=[chess.Move.from_uci(move)],
            )
            by_move[move] = {
                "move": move,
                "cp": self._cp(info, board.turn),
                "pv": [pv_move.uci() for pv_move in info.get("pv", [])],
            }
        return [{**by_move[move], "move": move} for move in moves]

    def _one(
        self,
        key: tuple,
        fen: str,
        multipv: int,
        moves: tuple[str, ...],
    ) -> tuple[tuple, list[dict]]:
        if self.failed.is_set():
            raise RuntimeError("Stockfish pool failed; restart the task")
        engine = self.available.get()
        try:
            if self.failed.is_set():
                raise RuntimeError("Stockfish pool failed; restart the task")
            return key, self._search(engine, fen, multipv, moves)
        except BaseException:
            self.failed.set()
            close_engines(self.engines)
            raise
        finally:
            self.available.put(engine)

    def batch(
        self,
        requests: Iterable[tuple[str, int, tuple[str, ...]]],
    ) -> dict[tuple, list[dict]]:
        if self.failed.is_set():
            raise RuntimeError("Stockfish pool failed; restart the task")
        requested = list(dict.fromkeys(requests))
        result = {}
        missing = []
        with self.lock:
            for fen, multipv, moves in requested:
                key = (fen, multipv, moves)
                if key in self.cache:
                    result[key] = self.cache[key]
                    self.cache.move_to_end(key)
                else:
                    missing.append((key, fen, multipv, moves))

        # parallel oracle: distribute uncached requests over persistent engines.
        started = time.perf_counter()
        futures = [self.executor.submit(self._one, *request) for request in missing]
        for future in as_completed(futures):
            key, rows = future.result()
            result[key] = rows
            with self.lock:
                self.cache[key] = rows
                self.cache.move_to_end(key)
                while len(self.cache) > CACHE_SIZE:
                    self.cache.popitem(last=False)
        elapsed = time.perf_counter() - started
        with self.lock:
            self.query_count += len(missing)
            self.node_budget += sum(
                self.nodes_per_move * max(1, request[2]) for request in missing
            )
            self.wall_seconds += elapsed
        return result

    def suggestions(
        self,
        requests: Iterable[tuple[str, int]],
    ) -> dict[tuple[str, int], list[dict]]:
        source = list(dict.fromkeys(requests))
        raw = self.batch((fen, count, ()) for fen, count in source)
        return {(fen, count): raw[(fen, count, ())] for fen, count in source}

    def comparisons(
        self,
        requests: Iterable[tuple[str, tuple[str, ...]]],
    ) -> dict[tuple[str, tuple[str, ...]], dict[str, dict]]:
        source = [
            (fen, tuple(dict.fromkeys(moves))) for fen, moves in requests
        ]
        source = list(dict.fromkeys(source))
        raw = self.batch((fen, len(moves), moves) for fen, moves in source)
        result = {}
        for fen, moves in source:
            rows = raw[(fen, len(moves), moves)]
            if len(rows) != len(moves):
                raise RuntimeError(
                    f"restricted search returned {len(rows)} rows for {len(moves)} moves at {fen}"
                )
            result[(fen, moves)] = {
                move: {**row, "move": move} for move, row in zip(moves, rows)
            }
        return result

    def move_audits(
        self,
        requests: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], dict]:
        source = list(dict.fromkeys(requests))
        tops = self.suggestions((fen, 1) for fen, _ in source)
        comparisons = []
        for fen, played in source:
            best = tops[(fen, 1)][0]["move"]
            if best != played:
                comparisons.append((fen, (best, played)))
        compared = self.comparisons(comparisons)
        result = {}
        for fen, played in source:
            top = tops[(fen, 1)][0]
            best = top["move"]
            if best == played:
                best_cp = played_cp = top["cp"]
            else:
                rows = compared[(fen, (best, played))]
                best_cp = rows[best]["cp"]
                played_cp = rows[played]["cp"]
            result[(fen, played)] = {
                "best_uci": best,
                "best_cp": best_cp,
                "played_cp": played_cp,
                "winrate_drop": max(
                    0.0,
                    expected_winrate(best_cp) - expected_winrate(played_cp),
                ),
            }
        return result

    def close(self) -> None:
        self.failed.set()
        close_engines(self.engines)
        self.executor.shutdown(wait=True, cancel_futures=True)
