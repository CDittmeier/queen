"""Stage-5 regret eval: score generated move-narratives against Stockfish.

For each ``{"fen", "generated"}`` sample we parse the final
``Therefore, I should play **...**`` recommendation and score it with Stockfish
@ 10k nodes as the oracle, in lichess win% (mover POV):

    regret = win%(best move) - win%(recommended move)   (>= 0)

The recommendation is taken as-is: an unparseable or illegal move is pinned to
played win% = 0 (so its regret is the full optimal win%) and counted toward the
illegal rate -- there is no fallback to another move.

Assumes the process runs from the repo root (STOCKFISH_BIN overridable via env).
"""
import math
import os
import re

import chess
import chess.engine

STOCKFISH_BIN = os.environ.get("STOCKFISH_BIN", "data/engines/stockfish_25080907_x64_avx2")
SF_NODES = 10000

_PROMO = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
_BEST_RE = re.compile(r"I should play\s+([^.\n]*)\.")   # final "...I should play **X**." clause
_SQUARE_RE = re.compile(r"<SQUARE_(\d+)>")
_PROMO_RE = re.compile(r"promoting to <PIECE_[MO]([QRBN])>")


def _win_pct(cp: int) -> float:
    """Lichess win% (0..100) from mover-POV centipawns."""
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def _square_to_board(token: int, white_to_move: bool) -> int:
    """POV square token (1..64) -> python-chess square (a1=0); black mirrors by ^56."""
    sq = token - 1
    return sq if white_to_move else sq ^ 56


def parse_move(generated: str, board: chess.Board) -> chess.Move | None:
    """The final 'I should play **...**' move as a legal move, else None.

    from = first square token, to = last (robust to captures/promotions that
    mention extra pieces but no extra square).
    """
    clauses = _BEST_RE.findall(generated)
    if not clauses:
        return None
    squares = _SQUARE_RE.findall(clauses[-1])
    if len(squares) < 2:
        return None
    white_to_move = board.turn == chess.WHITE
    frm = _square_to_board(int(squares[0]), white_to_move)
    to = _square_to_board(int(squares[-1]), white_to_move)
    promo = _PROMO_RE.search(clauses[-1])
    # Unspecified promotion defaults to queen.
    candidates = [_PROMO[promo.group(1)]] if promo else [None, chess.QUEEN]
    for promotion in candidates:
        move = chess.Move(frm, to, promotion=promotion)
        if move in board.legal_moves:
            return move
    return None


def compute_metrics(samples: list[dict]) -> dict:
    """Score ``{"fen", "generated"}`` samples -> regret / (il)legal-rate dict."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    try:
        engine.configure({"Threads": 1})
    except chess.engine.EngineError:
        pass
    limit = chess.engine.Limit(nodes=SF_NODES)

    regret_sum = regret_sum_legal = 0.0
    illegal = n = 0
    try:
        for sample in samples:
            board = chess.Board(sample["fen"])
            if board.is_game_over():   # no move to make
                continue
            n += 1
            mover = board.turn
            optimal = _win_pct(engine.analyse(board, limit)["score"].pov(mover).score(mate_score=10000))
            move = parse_move(sample["generated"], board)
            if move is None:
                illegal += 1
                played = 0.0
            else:
                board.push(move)
                played = _win_pct(engine.analyse(board, limit)["score"].pov(mover).score(mate_score=10000))
                regret_sum_legal += optimal - played
            regret_sum += optimal - played
    finally:
        engine.quit()

    if not n:
        return {}
    n_legal = n - illegal
    return {
        "regret": regret_sum / n,
        "illegal_rate": illegal / n,
        "legal_rate": n_legal / n,
        "regret_legal": regret_sum_legal / n_legal if n_legal else 0.0,
    }
