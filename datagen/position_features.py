"""Board-feature predicates and extractors over python-chess / BoardRepr.

Single source of truth for "does this position have property X" and "what's on
these squares". The position sampler's feature gates look predicates up in
`FEATURES`, and task feature-work shares the extractors here. Kept intentionally
lean — add features as tasks need them.
"""
from collections import defaultdict

import chess

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN


# ---------------------------------------------------------------------------
# Position-state predicates (chess.Board, side-to-move POV)
# ---------------------------------------------------------------------------

def in_check(board: chess.Board) -> bool:
    return board.is_check()


def is_checkmate(board: chess.Board) -> bool:
    return board.is_checkmate()


def is_stalemate(board: chess.Board) -> bool:
    return board.is_stalemate()


def has_check(board: chess.Board) -> bool:
    """True if the side to move has at least one legal move that gives check."""
    return any(board.gives_check(m) for m in board.legal_moves)


# Feature name -> predicate(chess.Board) -> bool, for config-driven filtering.
FEATURES = {
    "in_check":     in_check,
    "is_checkmate": is_checkmate,
    "is_stalemate": is_stalemate,
    "has_check":    has_check,
}


def mover_label(board: BoardRepr) -> str:
    """Word for the side to move. Under POV it is named relative to the POV anchor:
    'player' when the side to move is the anchor side, else 'opponent'. These
    differ only after an odd-length forward sequence, where the final side to move
    is the anchor's opponent. Without POV it is the absolute 'white' / 'black'."""
    if board.pov:
        return "player" if board.chess_board.turn == board._stm else "opponent"
    return "white" if board.chess_board.turn == chess.WHITE else "black"


# ---------------------------------------------------------------------------
# Square-set extractors (BoardRepr / token layer)
# ---------------------------------------------------------------------------

def line_piece_counts(line_sqs: tuple, board: BoardRepr) -> list[tuple[str, int]]:
    """[(piece_tok, n), ...] in canonical piece_tokens order, non-zero only, for
    the pieces occupying `line_sqs`."""
    counts: dict[str, int] = defaultdict(int)
    for sq in line_sqs:
        p = board.piece_at(sq)
        if p != EMPTY_TOKEN:
            counts[p] += 1
    return [(p, counts[p]) for p in board.piece_tokens if counts[p] > 0]


# ---------------------------------------------------------------------------
# Attack / defense (chess.Board level; control-based)
# ---------------------------------------------------------------------------

def get_attackers_and_defenders(board: chess.Board, square: int):
    """Control-based attackers and defenders of `square`.

    Returns (attackers, defenders), each a list of (piece, square) where `piece`
    is the chess.Piece doing the attacking and `square` is the board square it
    sits on. Every piece that attacks `square` is included regardless of legality
    — pinned pieces and kings count. A piece is a DEFENDER iff it is the same
    colour as the piece currently occupying `square` (and attacks it); otherwise
    it is an ATTACKER. (An empty square therefore has only attackers.)
    """
    occupant = board.piece_at(square)
    attackers: list[tuple[chess.Piece, int]] = []
    defenders: list[tuple[chess.Piece, int]] = []
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.attackers(color, square):
            piece = board.piece_at(sq)
            if occupant is not None and piece.color == occupant.color:
                defenders.append((piece, sq))
            else:
                attackers.append((piece, sq))
    return attackers, defenders


# ---------------------------------------------------------------------------
# Legal-move enumeration (BoardRepr level; returns token-resolved move dicts)
# ---------------------------------------------------------------------------

def _move_dict(board: BoardRepr, m: chess.Move) -> dict:
    """One legal move as a token-resolved dict:
      piece              moving piece's token
      from_sq, to_sq     square tokens; `to_sq` is the moving piece's LANDING
                         square (for en passant this is the destination, NOT the
                         captured pawn's square)
      captured_piece     captured piece's token, or None
      en_passant_square  captured pawn's square token (en passant only), else None
      castle_type        'kingside' | 'queenside' | None
      promotion_to       promotion piece's token, or None
      check_status       'check' | 'checkmate' | None  (prose suffix only — the
                         parse tag ignores it)
    """
    cb = board.chess_board
    frm, to = m.from_square, m.to_square
    captured = en_passant_square = castle_type = promotion_to = None
    if cb.is_castling(m):
        castle_type = "kingside" if cb.is_kingside_castling(m) else "queenside"
    elif cb.is_en_passant(m):
        cap_sq = chess.square(chess.square_file(to), chess.square_rank(frm))
        captured = board.piece_at(cap_sq)
        en_passant_square = board.sq_tok(cap_sq)
    elif cb.is_capture(m):
        captured = board.piece_at(to)
    if m.promotion is not None:
        # The mover is cb.turn (pre-push); under a fixed POV anchor that may be
        # the opponent (sequence moves), so resolve the token by the mover's
        # colour rather than assuming own == side-to-move.
        promotion_to = board.piece_tok_for(cb.turn, m.promotion)
    check_status = None
    if cb.gives_check(m):
        cb.push(m)
        check_status = "checkmate" if cb.is_checkmate() else "check"
        cb.pop()
    return {
        "piece":             board.piece_at(frm),
        "from_sq":           board.sq_tok(frm),
        "to_sq":             board.sq_tok(to),
        "captured_piece":    captured,
        "en_passant_square": en_passant_square,
        "castle_type":       castle_type,
        "promotion_to":      promotion_to,
        "check_status":      check_status,
    }


def moves_from(board: BoardRepr, square: int) -> list[dict]:
    """All legal moves of the side-to-move piece on `square`, as move dicts."""
    return [_move_dict(board, m) for m in list(board.chess_board.legal_moves)
            if m.from_square == square]


def all_moves(board: BoardRepr) -> list[dict]:
    """All legal moves of the side to move, as move dicts."""
    return [_move_dict(board, m) for m in list(board.chess_board.legal_moves)]


# ---------------------------------------------------------------------------
# Forward (stage-3/4) move-sequence helpers
# ---------------------------------------------------------------------------

def apply_move_sequence(board: chess.Board, moves) -> chess.Board:
    """The board after playing `moves` (chess.Move list) from `board` (a copy)."""
    b = board.copy(stack=False)
    for mv in moves:
        b.push(mv)
    return b


def origin_squares(initial: chess.Board, moves) -> dict[int, int]:
    """Map each square occupied on the board reached after `moves` back to the
    square that piece sat on in `initial` (surviving pieces only).

    Forward tasks pick a piece on the FINAL board but must name its square in the
    INITIAL position (the model only sees the initial board and must look ahead).
    The trace follows the moving piece through normal moves, captures, en passant,
    castling (the rook is relocated too), and promotions (the square is traced;
    the piece type changes). A piece captured along the way drops out of the map.
    """
    origin = {sq: sq for sq in chess.SQUARES if initial.piece_at(sq) is not None}
    b = initial.copy(stack=False)
    for mv in moves:
        if b.is_en_passant(mv):
            origin.pop(mv.to_square + (-8 if b.turn == chess.WHITE else 8), None)
        elif b.is_capture(mv):
            origin.pop(mv.to_square, None)
        if b.is_castling(mv):
            back = 0 if b.turn == chess.WHITE else 56
            if chess.square_file(mv.to_square) == 6:        # king-side (g-file)
                origin[chess.F1 + back] = origin.pop(chess.H1 + back)
            else:                                           # queen-side (c-file)
                origin[chess.D1 + back] = origin.pop(chess.A1 + back)
        origin[mv.to_square] = origin.pop(mv.from_square)
        b.push(mv)
    return origin


def final_board_repr(initial: BoardRepr, moves) -> BoardRepr:
    """The position reached after `moves`, as a BoardRepr whose POV is anchored
    to the INITIAL side to move — so its tokens read in the same perspective as
    `initial` (no mid-sequence M/O swap or square-mirror flip)."""
    final = apply_move_sequence(initial.chess_board, moves)
    return BoardRepr(final.fen(), initial.pov, pov_anchor=initial.chess_board.turn)


def sequence_move_dicts(initial: BoardRepr, moves) -> list[dict]:
    """Per-ply move dicts for the sequence, in chronological order. Each is built
    on its own pre-move state but tokenised under the initial side-to-move POV
    anchor, so squares/pieces stay in one perspective throughout the sequence."""
    anchor = initial.chess_board.turn
    cb = initial.chess_board.copy(stack=False)
    dicts = []
    for mv in moves:
        dicts.append(_move_dict(BoardRepr(cb.fen(), initial.pov, pov_anchor=anchor), mv))
        cb.push(mv)
    return dicts


def check_moves(board: BoardRepr) -> list[dict]:
    """All legal moves of the side to move that give check or checkmate, as move
    dicts (i.e. all_moves filtered to a non-None check_status)."""
    return [m for m in all_moves(board) if m["check_status"] is not None]


def capture_moves(board: BoardRepr) -> list[dict]:
    """All legal moves of the side to move that capture, as move dicts (i.e.
    all_moves filtered to a non-None captured_piece — en passant included)."""
    return [m for m in all_moves(board) if m["captured_piece"] is not None]
