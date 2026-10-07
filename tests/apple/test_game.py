import threading
from time import monotonic, sleep

import chess
import pytest

from mac_inference.game import GameError, LiveGame


class ControlledEngine:
    def __init__(self, blocked=False, invalid=False):
        self.started = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.invalid = invalid
        self.calls = []

    def analyze(self, board, game_index, attempt, progress):
        self.calls.append((board.fen(), list(board.move_stack), game_index, attempt))
        self.started.set()
        assert self.release.wait(3), "Test did not release the blocked engine"
        progress("ANALYSIS: A streamed test explanation.")
        preferred = chess.Move.from_uci("g1f3" if board.turn else "e7e5")
        move = (
            preferred
            if preferred in board.legal_moves
            else next(iter(board.legal_moves))
        )
        return {
            "fen": board.fen(),
            "text": "A completed test explanation.",
            "best_move_uci": None if self.invalid else move.uci(),
            "best_move_san": None if self.invalid else board.san(move),
            "generation_seconds": 0.1,
            "generated_tokens": 20,
        }


def wait_phase(game, phase):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        state = game.snapshot()
        if state["phase"] == phase and not state["engine_loading"]:
            return state
        sleep(0.01)
    pytest.fail(f"Expected {phase}, got {game.snapshot()}")


@pytest.fixture
def games():
    created = []

    def make(blocked=False, invalid=False, fen=None):
        engine = ControlledEngine(blocked, invalid)
        game = LiveGame(lambda: engine, chess.Board(fen) if fen else None)
        created.append((game, engine))
        return game, engine

    yield make
    for game, engine in created:
        engine.release.set()
        game.close()


def human_move(game, uci):
    state = game.snapshot()
    return game.move(uci, state["game_id"], state["version"])


def test_live_turns_preserve_history_and_reject_duplicate_moves(games):
    game, engine = games(blocked=True)
    initial = game.snapshot()
    thinking = human_move(game, "e2e4")
    assert thinking["phase"] == "thinking" and thinking["legal_moves"] == []
    assert engine.started.wait(3)
    with pytest.raises(GameError):
        game.move("d2d4", initial["game_id"], initial["version"])
    engine.release.set()
    complete = wait_phase(game, "playing")
    assert [move["uci"] for move in complete["moves"]] == ["e2e4", "e7e5"]
    assert complete["analysis"]["fen"] == engine.calls[0][0]
    human, queen = complete["moves"]
    assert human["analysis"] is None
    assert queen["analysis"]["text"] == "A completed test explanation."
    assert queen["analysis"]["best_move_san"] == "e5"
    assert engine.calls[0][1] == [chess.Move.from_uci("e2e4")]
    human_move(game, "g1f3")
    wait_phase(game, "playing")
    assert len(engine.calls[1][1]) == 3


def test_reset_supersedes_an_inflight_ai_turn(games):
    game, engine = games(blocked=True)
    human_move(game, "e2e4")
    assert engine.started.wait(3)
    old_id = game.snapshot()["game_id"]
    fresh = game.new_game("white")
    engine.release.set()
    game._worker.submit(lambda: None).result(timeout=3)
    state = game.snapshot()
    assert state["game_id"] != old_id and state["game_id"] == fresh["game_id"]
    assert state["fen"] == chess.STARTING_FEN
    assert state["moves"] == [] and state["analysis"] is None


def test_takeback_cancels_thinking_and_returns_the_human_turn(games):
    game, engine = games(blocked=True)
    human_move(game, "e2e4")
    assert engine.started.wait(3)
    state = game.snapshot()
    undone = game.take_back(state["game_id"], state["version"])
    engine.release.set()
    game._worker.submit(lambda: None).result(timeout=3)
    assert undone["fen"] == chess.STARTING_FEN
    assert game.snapshot()["moves"] == []
    assert "e2e4" in game.snapshot()["legal_moves"]


def test_takeback_removes_a_completed_human_and_ai_pair(games):
    game, _ = games()
    human_move(game, "e2e4")
    state = wait_phase(game, "playing")
    undone = game.take_back(state["game_id"], state["version"])
    assert undone["moves"] == [] and undone["fen"] == chess.STARTING_FEN


def test_playing_black_starts_with_a_real_ai_turn(games):
    game, engine = games()
    game.new_game("black")
    state = wait_phase(game, "playing")
    assert state["human"] == "black" and state["turn"] == "black"
    assert state["moves"][0]["actor"] == "queen"
    assert state["moves"][0]["uci"] == "g1f3"
    assert engine.calls[0][1] == []
    assert "e7e5" in state["legal_moves"]


def test_invalid_ai_response_keeps_the_position_and_retry_uses_a_new_attempt(games):
    game, engine = games(invalid=True)
    human_move(game, "e2e4")
    state = wait_phase(game, "error")
    assert len(state["moves"]) == 1
    assert state["turn"] == "black" and state["legal_moves"] == []
    engine.invalid = False
    game.retry(state["game_id"], state["version"])
    completed = wait_phase(game, "playing")
    assert len(completed["moves"]) == 2 and engine.calls[-1][3] == 1


def test_checkmate_ends_the_game_without_scheduling_ai(games):
    game, engine = games(fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    state = human_move(game, "f7g7")
    assert state["phase"] == "finished"
    assert state["outcome"] == {
        "result": "1-0",
        "winner": "white",
        "reason": "checkmate",
    }
    assert engine.calls == []


def test_promotion_requires_a_piece_choice_and_honors_underpromotion(games):
    game, _ = games(fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    with pytest.raises(GameError):
        human_move(game, "a7a8")
    state = human_move(game, "a7a8n")
    assert chess.Board(state["fen"]).piece_at(chess.A8) == chess.Piece(
        chess.KNIGHT, chess.WHITE
    )
    assert state["outcome"]["reason"] == "insufficient material"


@pytest.mark.parametrize(
    "fen,move,expected_square,absent_square",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "f1", "h1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "d6", "d5"),
    ],
)
def test_special_moves_are_applied_by_the_rules_engine(
    games, fen, move, expected_square, absent_square
):
    game, _ = games(blocked=True, fen=fen)
    state = human_move(game, move)
    board = chess.Board(state["fen"])
    assert board.piece_at(chess.parse_square(expected_square)) is not None
    assert board.piece_at(chess.parse_square(absent_square)) is None


def test_resignation_cancels_ai_and_exports_the_final_result(games):
    game, engine = games(blocked=True)
    human_move(game, "e2e4")
    assert engine.started.wait(3)
    state = game.snapshot()
    game.resign(state["game_id"], state["version"])
    engine.release.set()
    game._worker.submit(lambda: None).result(timeout=3)
    assert len(game.snapshot()["moves"]) == 1
    assert '[Result "0-1"]' in game.pgn() and "1. e4" in game.pgn()


def test_optional_threefold_draw_waits_for_a_claim():
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        board.push_uci(uci)
    game = LiveGame(lambda: ControlledEngine(), board)
    try:
        state = game.snapshot()
        assert state["phase"] == "playing" and state["can_claim_draw"]
        claimed = game.claim_draw(state["game_id"], state["version"])
        assert (
            claimed["phase"] == "finished" and claimed["outcome"]["result"] == "1/2-1/2"
        )
    finally:
        game.close()
