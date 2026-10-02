"""Shared constants and utilities for data generation, training, and eval."""
import chess
import torch

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

# <SQUARE_A1> … <SQUARE_H8>  (64 tokens, indexed by chess.SQUARES order a1=0)
SQUARE_TOKENS: list[str] = [
    f"<SQUARE_{chess.square_name(sq).upper()}>"
    for sq in chess.SQUARES
]

# <SQUARE_1> … <SQUARE_64>  (64 POV-relative tokens, index i = LC0 hidden state i)
# For white to move: SQUARE_i corresponds to board square i (a1=1, b1=2, …)
# For black to move: SQUARE_i corresponds to board square i^56 (a8=1, b8=2, …)
POV_SQUARE_TOKENS: list[str] = [f"<SQUARE_{i + 1}>" for i in range(64)]

# <PIECE_WP> … <PIECE_BK>  (12 tokens, board-absolute)
PIECE_TOKENS: list[str] = [
    "<PIECE_WP>", "<PIECE_WN>", "<PIECE_WB>",
    "<PIECE_WR>", "<PIECE_WQ>", "<PIECE_WK>",
    "<PIECE_BP>", "<PIECE_BN>", "<PIECE_BB>",
    "<PIECE_BR>", "<PIECE_BQ>", "<PIECE_BK>",
]

# <PIECE_MP> … <PIECE_OK>  (12 POV-relative tokens, mine = side-to-move)
POV_PIECE_TOKENS: list[str] = [
    "<PIECE_MP>", "<PIECE_MN>", "<PIECE_MB>",
    "<PIECE_MR>", "<PIECE_MQ>", "<PIECE_MK>",
    "<PIECE_OP>", "<PIECE_ON>", "<PIECE_OB>",
    "<PIECE_OR>", "<PIECE_OQ>", "<PIECE_OK>",
]

EMPTY_TOKEN = "<EMPTY>"

# Board-absolute tokens (v2 / v2.1): 64 + 12 + 1 = 77
ANSWER_SPECIAL_TOKENS: list[str] = SQUARE_TOKENS + PIECE_TOKENS + [EMPTY_TOKEN]

# POV-relative tokens (v3): 64 + 12 + 1 = 77
POV_ANSWER_SPECIAL_TOKENS: list[str] = POV_SQUARE_TOKENS + POV_PIECE_TOKENS + [EMPTY_TOKEN]

# (color, piece_type) → token string  (board-absolute)
_PIECE_TO_TOKEN: dict[tuple[bool, int], str] = {
    (chess.WHITE, chess.PAWN):   "<PIECE_WP>",
    (chess.WHITE, chess.KNIGHT): "<PIECE_WN>",
    (chess.WHITE, chess.BISHOP): "<PIECE_WB>",
    (chess.WHITE, chess.ROOK):   "<PIECE_WR>",
    (chess.WHITE, chess.QUEEN):  "<PIECE_WQ>",
    (chess.WHITE, chess.KING):   "<PIECE_WK>",
    (chess.BLACK, chess.PAWN):   "<PIECE_BP>",
    (chess.BLACK, chess.KNIGHT): "<PIECE_BN>",
    (chess.BLACK, chess.BISHOP): "<PIECE_BB>",
    (chess.BLACK, chess.ROOK):   "<PIECE_BR>",
    (chess.BLACK, chess.QUEEN):  "<PIECE_BQ>",
    (chess.BLACK, chess.KING):   "<PIECE_BK>",
}

# (is_mine, piece_type) → token string  (POV-relative; True == side-to-move)
_PIECE_TO_POV_TOKEN: dict[tuple[bool, int], str] = {
    (True,  chess.PAWN):   "<PIECE_MP>",
    (True,  chess.KNIGHT): "<PIECE_MN>",
    (True,  chess.BISHOP): "<PIECE_MB>",
    (True,  chess.ROOK):   "<PIECE_MR>",
    (True,  chess.QUEEN):  "<PIECE_MQ>",
    (True,  chess.KING):   "<PIECE_MK>",
    (False, chess.PAWN):   "<PIECE_OP>",
    (False, chess.KNIGHT): "<PIECE_ON>",
    (False, chess.BISHOP): "<PIECE_OB>",
    (False, chess.ROOK):   "<PIECE_OR>",
    (False, chess.QUEEN):  "<PIECE_OQ>",
    (False, chess.KING):   "<PIECE_OK>",
}


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _generate_position_dict(fen: str) -> tuple[dict, dict]:
    """
    Returns:
        square_to_piece : {sq_name: piece_token | EMPTY_TOKEN}
        piece_to_squares : {piece_token: [sq_name, ...]}
    """
    board = chess.Board(fen)

    square_to_piece: dict[str, str] = {}
    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq)
        piece   = board.piece_at(sq)
        square_to_piece[sq_name] = (
            _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] if piece else EMPTY_TOKEN
        )

    piece_to_squares: dict[str, list[str]] = {tok: [] for tok in PIECE_TOKENS}
    for sq_name, tok in square_to_piece.items():
        if tok != EMPTY_TOKEN:
            piece_to_squares[tok].append(sq_name)

    return square_to_piece, piece_to_squares


def _generate_pov_position_dict(fen: str) -> tuple[dict, dict]:
    """POV-relative variant of _generate_position_dict.

    Indexes squares by their LC0 hidden-state index (0–63) rather than board name.
    For white to move: POV index i == board square i.
    For black to move: POV index i == board square i^56 (board is mirrored).

    Returns:
        pov_idx_to_piece  : {pov_idx (int): piece_token | EMPTY_TOKEN}
        piece_to_pov_idxs : {piece_token: [pov_idx, ...]}
    """
    board    = chess.Board(fen)
    is_black = board.turn == chess.BLACK

    pov_idx_to_piece: dict[int, str] = {}
    for pov_idx in range(64):
        board_sq = pov_idx ^ 56 if is_black else pov_idx
        piece    = board.piece_at(board_sq)
        pov_idx_to_piece[pov_idx] = (
            _PIECE_TO_TOKEN[(piece.color, piece.piece_type)] if piece else EMPTY_TOKEN
        )

    piece_to_pov_idxs: dict[str, list[int]] = {tok: [] for tok in PIECE_TOKENS}
    for pov_idx, tok in pov_idx_to_piece.items():
        if tok != EMPTY_TOKEN:
            piece_to_pov_idxs[tok].append(pov_idx)

    return pov_idx_to_piece, piece_to_pov_idxs


# ---------------------------------------------------------------------------
# Chat prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ChessLM, an AI assistant with deep chess knowledge. "
    "You can analyze board positions and identify piece locations precisely. "
    "/no_think /system_override"
)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

_BLACK_IDX = torch.tensor([sq ^ 56 for sq in range(64)])


def turn_tensor(fens: list[str]) -> torch.Tensor:
    """(B,) bool tensor, True where ``fens[i]`` is black-to-move.

    Paired with ``encode_planes(pov=False, turn=...)`` to drive the per-sample
    spatial un-flip in absolute mode. The collate / eval-batch loop builds it
    once from the FENs it's already parsing for plane construction.
    """
    return torch.tensor(
        [chess.Board(f).turn == chess.BLACK for f in fens],
        dtype=torch.bool,
    )


@torch.no_grad()
def encode_positions(
    encoder,
    start_fens: list[str],
    moves_list: list[list[str]],
    end_fens: list[str],
    device: torch.device,
    dtype: torch.dtype,
    pov: bool = False,
) -> torch.Tensor:
    """Returns encoder hidden states (B, 16, 64, 1024).

    pov=False (default): un-flip black positions so index 0 always = a1
                         (board-absolute, matches SQUARE_TOKENS).
    pov=True:            leave hidden states in LC0's native order so index i
                         always = "my bottom-left + i" from current player's POV
                         (matches POV_SQUARE_TOKENS).
    """
    planes = torch.stack([
        encoder.input_planes_from_fen(sf, mv)
        for sf, mv in zip(start_fens, moves_list)
    ]).to(device)

    out = encoder(planes, output_hidden_states=True)
    hidden = torch.stack(out.all_hidden_states, dim=1).to(dtype)  # (B, 16, 64, 1024)

    if not pov:
        black_idx = _BLACK_IDX.to(device)
        for i, (sf, mv) in enumerate(zip(start_fens, moves_list)):
            board = chess.Board(sf)
            for m in mv:
                board.push_uci(m)
            if board.turn == chess.BLACK:
                hidden[i] = hidden[i, :, black_idx, :]

    return hidden


def encode_planes(
    encoder,
    planes: torch.Tensor,
    dtype: torch.dtype,
    *,
    pov: bool,
    turn: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the lc0 encoder on pre-built planes -> hidden states (B, 16, 64, 1024).

    The planes are POV-oriented (lc0 always encodes from side-to-move's
    perspective). ``pov=True`` returns the hidden states unchanged. ``pov=False``
    (absolute mode) un-flips the spatial axis for black-to-move samples so
    index 0 always corresponds to absolute a1, matching ``<SQUARE_A1>..`` text
    references; ``turn`` must then be a (B,) bool/int tensor with True/1 for
    black-to-move samples. Feature semantics stay POV in both modes ("mine" =
    side-to-move) — only the spatial axis differs.

    Two encoder interfaces are supported. Our lc0 (src/lc0_torch) returns only
    last_hidden_state, so we capture each layer by wrapping ``_forward_encoder``
    (the input-embedding output + the 15 encoder-layer outputs = 16 states, same
    convention as chess_lm_base). An encoder that natively exposes per-layer
    states via ``output_hidden_states`` is used directly.
    """
    with torch.no_grad():
        if hasattr(encoder, "_forward_encoder"):
            captured: list[torch.Tensor] = []
            orig = encoder._forward_encoder

            def wrapped(x, layer):
                if not captured:
                    captured.append(x)          # input-embedding output (state 0)
                out = orig(x, layer)
                captured.append(out)            # encoder-layer outputs (states 1..15)
                return out

            encoder._forward_encoder = wrapped
            try:
                encoder(input_planes=planes, return_dict=True)
            finally:
                del encoder._forward_encoder    # restore the bound method
            hidden = torch.stack(captured, dim=1).to(dtype)
        else:
            out = encoder(planes, output_hidden_states=True)
            hidden = torch.stack(out.all_hidden_states, dim=1).to(dtype)

    if not pov:
        assert turn is not None, "absolute mode (pov=False) requires per-sample turn tensor"
        black_idx = _BLACK_IDX.to(hidden.device)
        is_black = turn.to(hidden.device).bool()
        if is_black.any():
            hidden[is_black] = hidden[is_black][:, :, black_idx, :]
    return hidden
