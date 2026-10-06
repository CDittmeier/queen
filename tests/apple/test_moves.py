import chess
import pytest

from mac_inference.common import best_move, request_seed


@pytest.mark.parametrize(
    "fen,raw,expected",
    [
        (chess.STARTING_FEN, "<PIECE_MN><SQUARE_7><SQUARE_22>", "g1f3"),
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "<PIECE_MP><SQUARE_13><SQUARE_29>",
            "e7e5",
        ),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "<PIECE_MP><SQUARE_49><SQUARE_57><PIECE_MN>",
            "a7a8n",
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "<PIECE_MP><SQUARE_37><PIECE_OP><SQUARE_44>",
            "e5d6",
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "<PIECE_MK><SQUARE_5><SQUARE_7>",
            "e1g1",
        ),
    ],
)
def test_recommendations_are_legal_and_use_root_pov(fen, raw, expected):
    board = chess.Board(fen)
    move = best_move(board, f"ANALYSIS: Example\nBEST_MOVE: {raw}\nEVALUATION: 0")
    assert move is not None and move.uci() == expected and move in board.legal_moves


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "ANALYSIS: <PIECE_MN><SQUARE_7><SQUARE_22>",
        "BEST_MOVE: <PIECE_MN><SQUARE_7>",
        "BEST_MOVE: <PIECE_MN><SQUARE_7><SQUARE_64>",
        "BEST_MOVE: <PIECE_OP><SQUARE_13><SQUARE_29>",
        "BEST_MOVE: <PIECE_MP><SQUARE_13><PIECE_ON><SQUARE_29>",
        "BEST_MOVE: <PIECE_MP><SQUARE_0><SQUARE_29>",
    ],
)
def test_unparseable_or_illegal_output_has_no_fallback(raw):
    assert best_move(chess.Board(), raw) is None


def test_evaluation_seed_matches_published_initial_request():
    assert request_seed(20260823, 0, chess.Board()) == 917225541
