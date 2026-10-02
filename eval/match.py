"""Play fixed-opening games between two Players and score them.

A match is 10 games over a fixed schedule: 5 opening setups, each played twice
with colors swapped, so opening and color are balanced. Games run to a natural
result (``claim_draw=True``) or a 400-ply cap scored as a draw. An engine error
or illegal move forfeits that game for the offending side.
"""
from __future__ import annotations

from typing import Any

import chess
import chess.pgn

from .engines import Player

MAX_PLIES = 400

OPENINGS: list[dict[str, Any]] = [
    {"name": "italian_setup", "moves_san": ["e4", "e5", "Nf3", "Nc6"]},
    {"name": "open_sicilian_setup", "moves_san": ["e4", "c5", "Nf3", "d6"]},
    {"name": "qgd_setup", "moves_san": ["d4", "d5", "c4", "e6"]},
    {"name": "kings_indian_setup", "moves_san": ["d4", "Nf6", "c4", "g6"]},
    {"name": "english_four_knights", "moves_san": ["c4", "e5", "Nc3", "Nf6"]},
]
# (opening, A-plays-white): A sits on both sides of every opening => balanced.
SCHEDULE: list[tuple[dict[str, Any], bool]] = [
    (op, a_white) for op in OPENINGS for a_white in (True, False)
]


def play_game(
    white: Player,
    black: Player,
    opening_moves_san: list[str],
    opening_name: str = "",
    *,
    event: str = "benchmark",
) -> dict[str, Any]:
    """Play one game from an opening; return a record scored from White's POV."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update(
        Event=event, White=white.name, Black=black.name, Opening=opening_name
    )
    node: chess.pgn.GameNode = game
    white.new_game()
    black.new_game()

    for san in opening_moves_san:
        move = board.parse_san(san)
        node = node.add_main_variation(move)
        board.push(move)

    plies = len(opening_moves_san)
    capped = False
    while not board.is_game_over(claim_draw=True):
        if plies >= MAX_PLIES:
            capped = True
            break
        mover = white if board.turn == chess.WHITE else black
        try:
            move = mover.choose_move(board)
            if move not in board.legal_moves:
                raise ValueError(f"illegal move {move}")
        except Exception as exc:  # forfeit: the side to move loses
            loser_is_white = board.turn == chess.WHITE
            result = "0-1" if loser_is_white else "1-0"
            term = f"forfeit_{mover.name}:{type(exc).__name__}"
            return _finish(game, board, result, term, opening_name, plies)
        node = node.add_main_variation(move)
        board.push(move)
        plies += 1

    if capped:
        return _finish(game, board, "1/2-1/2", f"ply_cap_{MAX_PLIES}_drawn", opening_name, plies)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        term = "natural_draw" if outcome is None else f"natural_{outcome.termination.name.lower()}"
        return _finish(game, board, "1/2-1/2", term, opening_name, plies)
    result = "1-0" if outcome.winner == chess.WHITE else "0-1"
    return _finish(game, board, result, f"natural_{outcome.termination.name.lower()}", opening_name, plies)


def _finish(
    game: chess.pgn.Game,
    board: chess.Board,
    result: str,
    termination: str,
    opening_name: str,
    plies: int,
) -> dict[str, Any]:
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return {
        "opening": opening_name,
        "result": result,
        "termination": termination,
        "plies": plies,
        "final_fen": board.fen(),
        "pgn": game.accept(exporter).strip(),
    }


def score_white(result: str) -> float:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]


def play_match(a: Player, b: Player, *, schedule=SCHEDULE) -> dict[str, Any]:
    """Play the full schedule between A and B; return A's score and per-game records."""
    records: list[dict[str, Any]] = []
    score_a = 0.0
    for opening, a_white in schedule:
        white, black = (a, b) if a_white else (b, a)
        rec = play_game(white, black, opening["moves_san"], opening["name"])
        white_pov = score_white(rec["result"])
        rec["a_white"] = a_white
        rec["score_a"] = white_pov if a_white else 1.0 - white_pov
        score_a += rec["score_a"]
        records.append(rec)
    return {"a": a.name, "b": b.name, "games": len(records), "score_a": score_a, "records": records}
