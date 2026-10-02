"""Batch-time FEN to lc0 classical 112-plane encoding."""

from __future__ import annotations

from collections import defaultdict

import chess
import torch

INPUT_PLANES = 112
BOARD_SQUARES = 64


def _is_start_position(board: chess.Board) -> bool:
    return " ".join(board.fen(en_passant="legal").split()[:4]) == " ".join(
        chess.STARTING_FEN.split()[:4]
    )


def _position_key(board: chess.Board) -> str:
    return " ".join(board.fen(en_passant="legal").split()[:4])


def _oriented_square(square: int, pov: chess.Color) -> int:
    return square if pov == chess.WHITE else chess.square_mirror(square)


def _build_history_boards(
    start_fen: str,
    moves: list[str] | None = None,
) -> list[chess.Board]:
    board = chess.Board(start_fen)
    boards = [board.copy(stack=False)]
    for uci in moves or []:
        board.push(chess.Move.from_uci(uci))
        boards.append(board.copy(stack=False))
    return boards


def _looks_like_fen(s: str) -> bool:
    # A FEN always has rank separators ('/'); a UCI ply (e.g. "e2e4") never does.
    return "/" in s


def _build_boards(
    fen: str,
    history: list[str] | None = None,
) -> list[chess.Board]:
    """Return the board list whose **last** element is the current position.

    ``fen`` is the current/final position. ``history`` carries the prior
    context lc0 needs and may be either form (detected automatically):

      * ``None`` / ``[]``      — no prior context; boards = ``[fen]`` (history
        planes are then synthesized from the current board, as before).
      * list of prior **FENs**  — the boards leading up to ``fen``, oldest to
        newest, *excluding* ``fen`` itself. boards = ``[Board(h) ...] + [Board(fen)]``.
      * list of UCI **plies**   — legacy ``(start_fen, moves)`` form: ``fen`` is
        treated as the root and the plies are replayed forward (``fen`` ends up
        in the history, current = the post-plies board).
    """
    hist = [h for h in (history or []) if h]
    if not hist:
        return [chess.Board(fen)]
    if all(_looks_like_fen(h) for h in hist):
        return [chess.Board(h) for h in hist] + [chess.Board(fen)]
    # UCI plies applied to ``fen`` as the root (legacy start_fen+moves form).
    return _build_history_boards(fen, hist)


def _build_repetition_counts(boards: list[chess.Board]) -> list[int]:
    counts: defaultdict[str, int] = defaultdict(int)
    repetitions: list[int] = []
    for board in boards:
        key = _position_key(board)
        repetitions.append(counts[key])
        counts[key] += 1
    return repetitions


def _write_piece_plane(
    planes: torch.Tensor,
    plane_idx: int,
    board: chess.Board,
    *,
    piece_type: chess.PieceType,
    color: chess.Color,
    pov: chess.Color,
) -> None:
    for square in board.pieces(piece_type, color):
        planes[plane_idx, _oriented_square(square, pov)] = 1.0


def _apply_fen_only_en_passant_synthesis(
    planes: torch.Tensor,
    *,
    plane_base: int,
    board: chess.Board,
    pov: chess.Color,
) -> None:
    if board.ep_square is None:
        return

    if board.turn == chess.WHITE:
        moved_color = chess.BLACK
        current_square = board.ep_square - 8
        previous_square = board.ep_square + 8
    else:
        moved_color = chess.WHITE
        current_square = board.ep_square + 8
        previous_square = board.ep_square - 8

    plane_idx = plane_base + (0 if moved_color == pov else 6)
    planes[plane_idx, _oriented_square(current_square, pov)] = 0.0
    planes[plane_idx, _oriented_square(previous_square, pov)] = 1.0


def encode_classical_112_planes(
    fen: str,
    history: list[str] | None = None,
) -> torch.Tensor:
    """Encode the 112 lc0 classical input planes for the position ``fen``.

    ``fen`` is the current/final board. ``history`` (prior FENs or prior plies;
    see ``_build_boards``) supplies the up-to-7 preceding board states that fill
    lc0's history planes; when absent they are synthesized from ``fen``.
    """
    boards = _build_boards(fen, history)
    repetitions = _build_repetition_counts(boards)
    current_board = boards[-1]
    pov = current_board.turn

    planes = torch.zeros(INPUT_PLANES, BOARD_SQUARES, dtype=torch.float32)
    piece_layout = (
        (chess.PAWN, 0),
        (chess.KNIGHT, 1),
        (chess.BISHOP, 2),
        (chess.ROOK, 3),
        (chess.QUEEN, 4),
        (chess.KING, 5),
    )

    for history_slot in range(8):
        history_idx = len(boards) - 1 - history_slot
        if history_idx < 0:
            board = boards[0]
            repetition_count = repetitions[0]
            if _is_start_position(board):
                break
            synthetic_history = True
        else:
            board = boards[history_idx]
            repetition_count = repetitions[history_idx]
            synthetic_history = False

        base = history_slot * 13
        for piece_type, offset in piece_layout:
            _write_piece_plane(
                planes,
                base + offset,
                board,
                piece_type=piece_type,
                color=pov,
                pov=pov,
            )
            _write_piece_plane(
                planes,
                base + 6 + offset,
                board,
                piece_type=piece_type,
                color=not pov,
                pov=pov,
            )
        if repetition_count >= 1:
            planes[base + 12].fill_(1.0)
        if synthetic_history:
            _apply_fen_only_en_passant_synthesis(
                planes,
                plane_base=base,
                board=board,
                pov=pov,
            )

    aux = 13 * 8
    if current_board.has_queenside_castling_rights(pov):
        planes[aux + 0].fill_(1.0)
    if current_board.has_kingside_castling_rights(pov):
        planes[aux + 1].fill_(1.0)
    if current_board.has_queenside_castling_rights(not pov):
        planes[aux + 2].fill_(1.0)
    if current_board.has_kingside_castling_rights(not pov):
        planes[aux + 3].fill_(1.0)
    if pov == chess.BLACK:
        planes[aux + 4].fill_(1.0)
    planes[aux + 5].fill_(float(current_board.halfmove_clock))
    planes[aux + 7].fill_(1.0)
    return planes


def encode_fen_batch(
    fens: list[str],
    history_batch: list[list[str] | None] | None = None,
) -> torch.Tensor:
    """Batch ``encode_classical_112_planes``. ``fens`` are current/final boards;
    ``history_batch`` is the matching list of per-position prior contexts (prior
    FENs or plies; see ``_build_boards``)."""
    if history_batch is None:
        history_batch = [None] * len(fens)
    if len(fens) != len(history_batch):
        raise ValueError(
            f"Expected one history list per FEN, got {len(fens)} FENs and {len(history_batch)} histories"
        )
    return torch.stack(
        [encode_classical_112_planes(fen, hist) for fen, hist in zip(fens, history_batch)],
        dim=0,
    )
