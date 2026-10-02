"""Chess engine benchmarking: rate Stockfish (by nodes) and lc0 networks on a
shared Elo ladder, anchored to a random-legal floor.

Modular by design — ``Player`` (eval.engines) is the only extension point, so
LLM move-pickers can join the same ladder later without touching the match or
rating code.
"""
from .elo import Match, expected_score, fit_elos
from .engines import Player, RandomPlayer, UciPlayer, build_player, lc0, stockfish
from .match import OPENINGS, SCHEDULE, play_game, play_match, score_white

__all__ = [
    "Player",
    "RandomPlayer",
    "UciPlayer",
    "build_player",
    "stockfish",
    "lc0",
    "OPENINGS",
    "SCHEDULE",
    "play_game",
    "play_match",
    "score_white",
    "Match",
    "expected_score",
    "fit_elos",
]
