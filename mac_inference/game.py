"""One local live game; chess rules own the board, QUEEN supplies AI moves."""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import chess
import chess.pgn

from .common import BoardEncoder, decode_text, prompt_ids, request_seed, result


class GameError(ValueError):
    pass


class SupersededTurn(Exception):
    """A reset, takeback or resignation superseded an in-flight generation."""


class QueenEngine:
    def __init__(self, directory: Path, max_tokens=2048, temperature=0.6):
        from .mlx import MLXRunner

        self.tokenizer, self.ids, self.prompt = prompt_ids(directory)
        self.encoder = BoardEncoder(directory)
        self.runner = MLXRunner(directory)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def analyze(self, board, game_index, attempt, on_progress):
        previous = board.copy(stack=True)
        history = []
        while previous.move_stack:
            previous.pop()
            history.append(previous.fen())
        history.reverse()
        self.runner.set_board(self.encoder.encode(board.fen(), history))
        seed = request_seed(20260823 + attempt, game_index, board)
        raw, timing = self.runner.generate(
            self.tokenizer,
            self.ids,
            self.max_tokens,
            self.temperature,
            seed,
            on_progress=on_progress,
        )
        return result(board, raw, **timing)


class LiveGame:
    def __init__(self, engine_factory, board=None):
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="queen")
        self._engine_factory = engine_factory
        self._engine = None
        self._engine_error = None
        self._engine_loading = True
        self._closed = False
        self._future = None
        self._game_index = 0
        self._reset(chess.WHITE, board)
        # Load and use MLX on this same worker; HTTP threads never touch the model.
        self._worker.submit(self._load_engine)

    def _load_engine(self):
        try:
            self._engine = self._engine_factory()
        except Exception:
            logging.exception("QUEEN initialization failed")
            with self._lock:
                self._engine_error = "QUEEN couldn't load. Check the server terminal."
        finally:
            with self._lock:
                self._engine_loading = False

    def _reset(self, human, board=None):
        self.id = uuid.uuid4().hex
        self.version = 0
        self.human = human
        self.board = board.copy(stack=True) if board is not None else chess.Board()
        self.moves = []
        self.analysis = None
        self.thinking_text = ""
        self.error = None
        self.outcome = None
        self.phase = "playing"
        self._attempt = 0
        self._started = None
        self._finish_if_terminal()

    def _check_request(self, game_id, version):
        if game_id != self.id or type(version) is not int or version != self.version:
            raise GameError("The game changed. Refresh the position and try again.")

    def new_game(self, color):
        if color not in ("white", "black"):
            raise GameError("Choose White or Black.")
        with self._lock:
            if self._future:
                self._future.cancel()
            self._game_index += 1
            self._reset(color == "white")
            if self.board.turn != self.human:
                self._queue_ai()
            return self.snapshot()

    def _push(self, move, actor):
        self.moves.append(
            {
                "uci": move.uci(),
                "san": self.board.san(move),
                "color": "white" if self.board.turn else "black",
                "number": self.board.fullmove_number,
                "actor": actor,
            }
        )
        self.board.push(move)
        self.version += 1
        self._finish_if_terminal()

    def _finish_if_terminal(self):
        outcome = self.board.outcome()
        if outcome:
            self.outcome = {
                "result": outcome.result(),
                "winner": None
                if outcome.winner is None
                else ("white" if outcome.winner else "black"),
                "reason": outcome.termination.name.lower().replace("_", " "),
            }
            self.phase = "finished"

    def move(self, uci, game_id, version):
        with self._lock:
            self._check_request(game_id, version)
            if self.phase != "playing" or self.board.turn != self.human:
                raise GameError("Wait for your turn.")
            try:
                move = chess.Move.from_uci(uci)
            except (ValueError, TypeError):
                raise GameError("Choose a legal move.") from None
            if move not in self.board.legal_moves:
                raise GameError("That move isn't legal in this position.")
            self._push(move, "human")
            if self.phase != "finished":
                self._attempt = 0
                self._queue_ai()
            return self.snapshot()

    def _queue_ai(self):
        if self._closed:
            return
        self.phase = "thinking"
        self.error = None
        self.thinking_text = ""
        self._started = perf_counter()
        self._future = self._worker.submit(
            self._play_ai,
            self.id,
            self.version,
            self.board.copy(stack=True),
            self._game_index,
            self._attempt,
        )

    def _current(self, game_id, version):
        return (
            not self._closed
            and self.id == game_id
            and self.version == version
            and self.phase == "thinking"
        )

    def _play_ai(self, game_id, version, board, game_index, attempt):
        def progress(raw):
            with self._lock:
                if not self._current(game_id, version):
                    raise SupersededTurn()
                self.thinking_text = decode_text(board, raw)

        try:
            with self._lock:
                if not self._current(game_id, version):
                    return
            if self._engine is None:
                raise RuntimeError(self._engine_error or "QUEEN is unavailable")
            answer = self._engine.analyze(board, game_index, attempt, progress)
            with self._lock:
                if not self._current(game_id, version):
                    return
                self.analysis = {
                    **answer,
                    "move_number": board.fullmove_number,
                    "color": "white" if board.turn else "black",
                }
                uci = answer.get("best_move_uci")
                move = chess.Move.from_uci(uci) if isinstance(uci, str) else None
                if (
                    answer.get("fen") != self.board.fen()
                    or move not in self.board.legal_moves
                ):
                    self.phase = "error"
                    self.error = "QUEEN didn't return a legal move. Try again."
                    return
                self._push(move, "queen")
                if self.phase != "finished":
                    self.phase = "playing"
                self.thinking_text = ""
        except SupersededTurn:
            pass
        except Exception:
            logging.exception("QUEEN turn failed")
            with self._lock:
                if self._current(game_id, version):
                    self.phase = "error"
                    self.error = "QUEEN couldn't finish that turn. Try again."

    def retry(self, game_id, version):
        with self._lock:
            self._check_request(game_id, version)
            if self.phase != "error" or self.board.turn == self.human:
                raise GameError("There's no AI turn to retry.")
            self._attempt += 1
            self.version += 1
            self._queue_ai()
            return self.snapshot()

    def take_back(self, game_id, version):
        with self._lock:
            self._check_request(game_id, version)
            indexes = [
                i for i, move in enumerate(self.moves) if move["actor"] == "human"
            ]
            if not indexes:
                raise GameError("No move to take back yet.")
            index = indexes[-1]
            for _ in self.moves[index:]:
                self.board.pop()
            del self.moves[index:]
            self.version += 1
            self.phase = "playing"
            self.outcome = None
            self.analysis = None
            self.thinking_text = ""
            self.error = None
            if self._future:
                self._future.cancel()
            return self.snapshot()

    def resign(self, game_id, version):
        with self._lock:
            self._check_request(game_id, version)
            if self.phase == "finished":
                raise GameError("The game has already ended.")
            self.version += 1
            self.phase = "finished"
            self.thinking_text = ""
            self.error = None
            self.outcome = {
                "result": "0-1" if self.human else "1-0",
                "winner": "black" if self.human else "white",
                "reason": "resignation",
            }
            return self.snapshot()

    def claim_draw(self, game_id, version):
        with self._lock:
            self._check_request(game_id, version)
            if (
                self.phase != "playing"
                or self.board.turn != self.human
                or not self.board.can_claim_draw()
            ):
                raise GameError("A draw cannot be claimed in this position.")
            self.version += 1
            self.phase = "finished"
            self.outcome = {
                "result": "1/2-1/2",
                "winner": None,
                "reason": "threefold repetition"
                if self.board.can_claim_threefold_repetition()
                else "fifty-move rule",
            }
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            your_turn = self.phase == "playing" and self.board.turn == self.human
            return {
                "game_id": self.id,
                "version": self.version,
                "fen": self.board.fen(),
                "human": "white" if self.human else "black",
                "turn": "white" if self.board.turn else "black",
                "phase": self.phase,
                "legal_moves": [move.uci() for move in self.board.legal_moves]
                if your_turn
                else [],
                "moves": [dict(move) for move in self.moves],
                "last_move": self.moves[-1]["uci"] if self.moves else None,
                "in_check": self.board.is_check(),
                "outcome": self.outcome,
                "analysis": self.analysis,
                "thinking_text": self.thinking_text,
                "error": self.error or self._engine_error,
                "engine_loading": self._engine_loading,
                "thinking_seconds": perf_counter() - self._started
                if self.phase == "thinking"
                else 0,
                "can_take_back": any(move["actor"] == "human" for move in self.moves),
                "can_claim_draw": your_turn and self.board.can_claim_draw(),
            }

    def pgn(self):
        with self._lock:
            game = chess.pgn.Game.from_board(self.board)
            game.headers["Event"] = "Local game against QUEEN"
            game.headers["White"] = "You" if self.human else "QUEEN"
            game.headers["Black"] = "QUEEN" if self.human else "You"
            if self.outcome:
                game.headers["Result"] = self.outcome["result"]
            return str(game) + "\n"

    def close(self):
        with self._lock:
            self._closed = True
        self._worker.shutdown(wait=True, cancel_futures=True)
