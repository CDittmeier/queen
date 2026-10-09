import chess

from mac_inference.annotate import classify, critical_moments, moves_in, summary, to_san


def ply(**fields):
    base = {
        "ply": 0,
        "move": chess.Move.from_uci("e2e4"),
        "best": chess.Move.from_uci("e2e4"),
        "best_score": 0.5,
        "second_score": 0.5,
        "played_score": 0.5,
        "material": 0,
    }
    return {**base, **fields}


def test_classify_moments():
    assert classify(ply(played_score=0.2)) == ("blunder", "??")
    assert classify(ply(played_score=0.38)) == ("mistake", "?")
    assert classify(ply(second_score=0.3)) == ("only move", "!")
    assert classify(ply(material=-3, played_score=0.48)) == ("sacrifice", "!")
    assert classify(ply(material=-3, played_score=0.4)) == ("sacrifice", "!?")
    assert classify(ply()) == (None, "")


def test_decided_positions_are_not_moments():
    assert classify(ply(best_score=0.99, material=-3)) == (None, "")
    assert classify(ply(best_score=0.01, played_score=0.0)) == (None, "")


def test_critical_moments_keep_the_biggest_in_game_order():
    plies = [
        ply(ply=0, played_score=0.38),
        ply(ply=1, played_score=0.2),
        ply(ply=2, second_score=0.3),
    ]
    assert [p["ply"] for p in critical_moments(plies, 2)] == [1, 2]


def test_moves_in_replays_until_an_illegal_move():
    board = chess.Board()
    line = "white pawn e2-e4 black pawn e7-e5 white knight g1-f3"
    moves, legal = moves_in(line, board)
    assert legal and [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3"]
    moves, legal = moves_in("white pawn e2-e4 white pawn d2-d4", board)
    assert not legal and [m.uci() for m in moves] == ["e2e4"]


def test_to_san_and_summary():
    assert (
        to_san("After 3.white pawn d4 takes black pawn e5 white king e1-g1, fine.")
        == "After dxe5 O-O, fine."
    )
    text = (
        "ANALYSIS:\nThe position is tense.\n\n"
        "The best move is 1.white knight g1-f3, developing. It eyes e5. More.\n"
        "BEST_MOVE: white knight g1-f3"
    )
    assert summary(text) == "The best move is Nf3, developing. It eyes e5."
