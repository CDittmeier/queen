from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import chess
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput


INPUT_PLANES = 112
BOARD_SQUARES = 64
POLICY_OUTPUT_SIZE = 1858
WDL_OUTPUT_SIZE = 3
INPUT_CLASSICAL_112_PLANE = 1


class Activation:
    DEFAULT = 0
    MISH = 1
    RELU = 2
    NONE = 3
    TANH = 4
    SIGMOID = 5
    SELU = 6
    SWISH = 7
    RELU_2 = 8
    SOFTMAX = 9


def _zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


def _copy_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    target = getattr(module, name)
    if target.shape != value.shape:
        raise ValueError(
            f"Shape mismatch for buffer {name!r}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
        )
    target.copy_(value.detach().to(device=target.device, dtype=target.dtype))


def _mish(x: torch.Tensor) -> torch.Tensor:
    e = torch.exp(x)
    n = e * e + 2.0 * e
    d = x / (n + 2.0)
    return torch.where(x <= -0.125, n * d, x - 2.0 * d)


def _selu(x: torch.Tensor) -> torch.Tensor:
    alpha = 1.67326324
    scale = 1.05070098
    return torch.where(x > 0, scale * x, scale * alpha * (torch.exp(x) - 1.0))


def _activate(x: torch.Tensor, activation: int) -> torch.Tensor:
    if activation == Activation.NONE:
        return x
    if activation == Activation.RELU:
        return torch.relu(x)
    if activation == Activation.RELU_2:
        relu = torch.relu(x)
        return relu * relu
    if activation == Activation.MISH:
        return _mish(x)
    if activation == Activation.TANH:
        return torch.tanh(x)
    if activation == Activation.SIGMOID:
        return torch.sigmoid(x)
    if activation == Activation.SELU:
        return _selu(x)
    if activation == Activation.SWISH:
        return x / (1.0 + torch.exp(-x))
    raise ValueError(f"Unsupported activation {activation}")


def _linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: int = Activation.NONE,
) -> torch.Tensor:
    out = F.linear(x, weight, bias)
    return _activate(out, activation)


def _layer_norm_with_skip(
    data: torch.Tensor,
    *,
    alpha: float,
    skip: torch.Tensor | None,
    gammas: torch.Tensor,
    betas: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    out = data * alpha
    if skip is not None:
        out = out + skip
    mean = out.mean(dim=-1, keepdim=True)
    var = ((out - mean) ** 2).mean(dim=-1, keepdim=True)
    return betas + gammas * (out - mean) / torch.sqrt(var + epsilon)


@lru_cache(maxsize=1)
def _rpe_map() -> torch.Tensor:
    out = torch.zeros((15 * 15, BOARD_SQUARES * BOARD_SQUARES), dtype=torch.float32)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    out[15 * (i - k + 7) + (j - l + 7), BOARD_SQUARES * (i * 8 + j) + k * 8 + l] = 1.0
    return out


class Lc0Bt4HFConfig(PretrainedConfig):
    model_type = "lc0_bt4"

    def __init__(
        self,
        *,
        embedding_dense_size: int = 0,
        embedding_size: int = 0,
        input_size: int = INPUT_PLANES,
        num_encoders: int = 0,
        num_heads: int = 1,
        head_dim: int = 1,
        dff_size: int = 0,
        use_smolgen: bool | None = None,
        use_rpe_q: bool | None = None,
        use_rpe_k: bool | None = None,
        use_rpe_v: bool | None = None,
        smolgen_hidden_channels: int = 0,
        smolgen_hidden_size: int = 0,
        smolgen_gen_size: int = 0,
        smolgen_gen_size_per_head: int = 0,
        policy_embedding_size: int = 0,
        policy_d_model: int = 0,
        value_embedding_size: int = 0,
        value_hidden_size: int = 0,
        moves_left_embedding_size: int = 0,
        moves_left_hidden_size: int = 0,
        default_activation: int = Activation.RELU,
        smolgen_activation: int = Activation.SWISH,
        ffn_activation: int = Activation.RELU,
        alpha: float = 1.0,
        encoder_eps: float = 1e-6,
        embedding_eps: float = 1e-3,
        input_format: int = INPUT_CLASSICAL_112_PLANE,
        input_planes: int = INPUT_PLANES,
        board_squares: int = BOARD_SQUARES,
        policy_output_size: int = POLICY_OUTPUT_SIZE,
        value_output_size: int = WDL_OUTPUT_SIZE,
        move_strings: list[str] | None = None,
        architectures: list[str] | None = None,
        **kwargs,
    ):
        self.embedding_dense_size = embedding_dense_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dff_size = dff_size
        self.use_smolgen = (
            smolgen_gen_size_per_head > 0 if use_smolgen is None else bool(use_smolgen)
        )
        self.use_rpe_q = bool(use_rpe_q) if use_rpe_q is not None else False
        self.use_rpe_k = bool(use_rpe_k) if use_rpe_k is not None else False
        self.use_rpe_v = bool(use_rpe_v) if use_rpe_v is not None else False
        self.smolgen_hidden_channels = smolgen_hidden_channels
        self.smolgen_hidden_size = smolgen_hidden_size
        self.smolgen_gen_size = smolgen_gen_size
        self.smolgen_gen_size_per_head = smolgen_gen_size_per_head
        self.policy_embedding_size = policy_embedding_size
        self.policy_d_model = policy_d_model
        self.value_embedding_size = value_embedding_size
        self.value_hidden_size = value_hidden_size
        self.moves_left_embedding_size = moves_left_embedding_size
        self.moves_left_hidden_size = moves_left_hidden_size
        self.default_activation = default_activation
        self.smolgen_activation = smolgen_activation
        self.ffn_activation = ffn_activation
        self.alpha = alpha
        self.encoder_eps = encoder_eps
        self.embedding_eps = embedding_eps
        self.input_format = input_format
        self.input_planes = input_planes
        self.board_squares = board_squares
        self.policy_output_size = policy_output_size
        self.value_output_size = value_output_size
        self.move_strings = list(move_strings or [])

        # Common Hugging Face aliases to make the config more discoverable.
        self.hidden_size = embedding_size
        self.intermediate_size = dff_size
        self.num_hidden_layers = num_encoders
        self.num_attention_heads = num_heads

        super().__init__(
            architectures=architectures or ["Lc0Bt4HFModel"],
            **kwargs,
        )

    @classmethod
    def from_bt4_config(
        cls,
        bt4_config: Any,
        **kwargs,
    ) -> "Lc0Bt4HFConfig":
        return cls(
            embedding_dense_size=bt4_config.embedding_dense_size,
            embedding_size=bt4_config.embedding_size,
            input_size=bt4_config.input_size,
            num_encoders=bt4_config.num_encoders,
            num_heads=bt4_config.num_heads,
            head_dim=bt4_config.head_dim,
            dff_size=bt4_config.dff_size,
            use_smolgen=getattr(bt4_config, "use_smolgen", None),
            use_rpe_q=getattr(bt4_config, "use_rpe_q", None),
            use_rpe_k=getattr(bt4_config, "use_rpe_k", None),
            use_rpe_v=getattr(bt4_config, "use_rpe_v", None),
            smolgen_hidden_channels=bt4_config.smolgen_hidden_channels,
            smolgen_hidden_size=bt4_config.smolgen_hidden_size,
            smolgen_gen_size=bt4_config.smolgen_gen_size,
            smolgen_gen_size_per_head=bt4_config.smolgen_gen_size_per_head,
            policy_embedding_size=bt4_config.policy_embedding_size,
            policy_d_model=bt4_config.policy_d_model,
            value_embedding_size=bt4_config.value_embedding_size,
            value_hidden_size=bt4_config.value_hidden_size,
            moves_left_embedding_size=bt4_config.moves_left_embedding_size,
            moves_left_hidden_size=bt4_config.moves_left_hidden_size,
            default_activation=bt4_config.default_activation,
            smolgen_activation=bt4_config.smolgen_activation,
            ffn_activation=bt4_config.ffn_activation,
            alpha=bt4_config.alpha,
            encoder_eps=bt4_config.encoder_eps,
            embedding_eps=bt4_config.embedding_eps,
            input_format=getattr(bt4_config, "input_format", INPUT_CLASSICAL_112_PLANE),
            **kwargs,
        )


@dataclass
class Lc0Bt4HFOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    policy_logits: torch.Tensor | None = None
    wdl_logits: torch.Tensor | None = None
    moves_left: torch.Tensor | None = None


_MOVE_STR_RE = re.compile(r'"([^"]+)"')


def _default_lc0_source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "lc0-src"


def _load_move_strings(source_root: str | Path | None = None) -> list[str]:
    root = Path(source_root) if source_root is not None else _default_lc0_source_root()
    encoder_path = root / "src" / "neural" / "encoder.cc"
    text = encoder_path.read_text(encoding="utf-8")
    start = text.index("const char* kMoveStrs[] = {")
    end = text.index("};", start)
    move_strings = _MOVE_STR_RE.findall(text[start:end])
    if len(move_strings) != POLICY_OUTPUT_SIZE:
        raise ValueError(
            f"Expected {POLICY_OUTPUT_SIZE} lc0 move strings, found {len(move_strings)} in {encoder_path}"
        )
    return move_strings


def _is_start_position(board: chess.Board) -> bool:
    return " ".join(board.fen(en_passant="legal").split()[:4]) == " ".join(
        chess.STARTING_FEN.split()[:4]
    )


def _position_key(board: chess.Board) -> str:
    return " ".join(board.fen(en_passant="legal").split()[:4])


def _oriented_square(square: int, pov: chess.Color) -> int:
    return square if pov == chess.WHITE else chess.square_mirror(square)


def _policy_move_key(move: chess.Move, pov: chess.Color) -> str:
    from_sq = _oriented_square(move.from_square, pov)
    to_sq = _oriented_square(move.to_square, pov)
    key = chess.square_name(from_sq) + chess.square_name(to_sq)
    if move.promotion is not None and move.promotion != chess.KNIGHT:
        promotion_map = {
            chess.QUEEN: "q",
            chess.ROOK: "r",
            chess.BISHOP: "b",
        }
        key += promotion_map[move.promotion]
    return key


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


def _encode_classical_112_planes(
    start_fen: str,
    moves: list[str] | None = None,
) -> torch.Tensor:
    boards = _build_history_boards(start_fen, moves)
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


class _BufferModule(nn.Module):
    def __init__(self, buffers: dict[str, torch.Tensor]):
        super().__init__()
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor)


class _SmolgenBuffers(_BufferModule):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(
            {
                "compress_w": _zeros(config.smolgen_hidden_channels, config.embedding_size),
                "dense1_w": _zeros(
                    config.smolgen_hidden_size,
                    BOARD_SQUARES * config.smolgen_hidden_channels,
                ),
                "dense1_b": _zeros(config.smolgen_hidden_size),
                "ln1_gammas": _zeros(config.smolgen_hidden_size),
                "ln1_betas": _zeros(config.smolgen_hidden_size),
                "dense2_w": _zeros(config.smolgen_gen_size, config.smolgen_hidden_size),
                "dense2_b": _zeros(config.smolgen_gen_size),
                "ln2_gammas": _zeros(config.smolgen_gen_size),
                "ln2_betas": _zeros(config.smolgen_gen_size),
            }
        )


class _EncoderLayerBuffers(_BufferModule):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(
            {
                "q_w": _zeros(config.embedding_size, config.embedding_size),
                "q_b": _zeros(config.embedding_size),
                "k_w": _zeros(config.embedding_size, config.embedding_size),
                "k_b": _zeros(config.embedding_size),
                "v_w": _zeros(config.embedding_size, config.embedding_size),
                "v_b": _zeros(config.embedding_size),
                "dense_w": _zeros(config.embedding_size, config.embedding_size),
                "dense_b": _zeros(config.embedding_size),
                "ln1_gammas": _zeros(config.embedding_size),
                "ln1_betas": _zeros(config.embedding_size),
                "ffn_dense1_w": _zeros(config.dff_size, config.embedding_size),
                "ffn_dense1_b": _zeros(config.dff_size),
                "ffn_dense2_w": _zeros(config.embedding_size, config.dff_size),
                "ffn_dense2_b": _zeros(config.embedding_size),
                "ln2_gammas": _zeros(config.embedding_size),
                "ln2_betas": _zeros(config.embedding_size),
                "rpe_q": _zeros(config.embedding_size, 15 * 15),
                "rpe_k": _zeros(config.embedding_size, 15 * 15),
                "rpe_v": _zeros(config.embedding_size, 15 * 15),
            }
        )
        self.smolgen = _SmolgenBuffers(config)


class _PolicyHeadBuffers(_BufferModule):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(
            {
                "ip_pol_w": _zeros(config.policy_embedding_size, config.embedding_size),
                "ip_pol_b": _zeros(config.policy_embedding_size),
                "ip2_pol_w": _zeros(config.policy_d_model, config.policy_embedding_size),
                "ip2_pol_b": _zeros(config.policy_d_model),
                "ip3_pol_w": _zeros(config.policy_d_model, config.policy_embedding_size),
                "ip3_pol_b": _zeros(config.policy_d_model),
                "ip4_pol_w": _zeros(4, config.policy_d_model),
            }
        )


class _ValueHeadBuffers(_BufferModule):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(
            {
                "ip_val_w": _zeros(config.value_embedding_size, config.embedding_size),
                "ip_val_b": _zeros(config.value_embedding_size),
                "ip1_val_w": _zeros(
                    config.value_hidden_size,
                    BOARD_SQUARES * config.value_embedding_size,
                ),
                "ip1_val_b": _zeros(config.value_hidden_size),
                "ip2_val_w": _zeros(config.value_output_size, config.value_hidden_size),
                "ip2_val_b": _zeros(config.value_output_size),
            }
        )


class _MovesLeftHeadBuffers(_BufferModule):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(
            {
                "ip_mov_w": _zeros(config.moves_left_embedding_size, config.embedding_size),
                "ip_mov_b": _zeros(config.moves_left_embedding_size),
                "ip1_mov_w": _zeros(
                    config.moves_left_hidden_size,
                    BOARD_SQUARES * config.moves_left_embedding_size,
                ),
                "ip1_mov_b": _zeros(config.moves_left_hidden_size),
                "ip2_mov_w": _zeros(1, config.moves_left_hidden_size),
                "ip2_mov_b": _zeros(1),
            }
        )


class Lc0Bt4HFPreTrainedModel(PreTrainedModel):
    config_class = Lc0Bt4HFConfig
    base_model_prefix = "lc0"
    main_input_name = "input_planes"
    supports_gradient_checkpointing = False
    _no_split_modules = ["_EncoderLayerBuffers"]

    def _init_weights(self, module: nn.Module) -> None:
        del module


class Lc0Bt4HFModel(Lc0Bt4HFPreTrainedModel):
    def __init__(self, config: Lc0Bt4HFConfig):
        super().__init__(config)
        self.register_buffer(
            "pos_encoding",
            _zeros(config.board_squares, config.board_squares),
        )
        self.register_buffer(
            "policy_index_map",
            _zeros(config.policy_output_size, dtype=torch.int64),
        )
        self.register_buffer(
            "ip_emb_pre_w",
            _zeros(
                config.board_squares * config.embedding_dense_size,
                config.board_squares * 12,
            ),
        )
        self.register_buffer(
            "ip_emb_pre_b",
            _zeros(config.board_squares * config.embedding_dense_size),
        )
        self.register_buffer(
            "ip_emb_w",
            _zeros(config.embedding_size, config.input_size),
        )
        self.register_buffer("ip_emb_b", _zeros(config.embedding_size))
        self.register_buffer("ip_emb_ln_gammas", _zeros(config.embedding_size))
        self.register_buffer("ip_emb_ln_betas", _zeros(config.embedding_size))
        self.register_buffer(
            "ip_mult_gate",
            _zeros(config.board_squares, config.embedding_size),
        )
        self.register_buffer(
            "ip_add_gate",
            _zeros(config.board_squares, config.embedding_size),
        )
        self.register_buffer(
            "ip_emb_ffn_dense1_w",
            _zeros(config.dff_size, config.embedding_size),
        )
        self.register_buffer("ip_emb_ffn_dense1_b", _zeros(config.dff_size))
        self.register_buffer(
            "ip_emb_ffn_dense2_w",
            _zeros(config.embedding_size, config.dff_size),
        )
        self.register_buffer("ip_emb_ffn_dense2_b", _zeros(config.embedding_size))
        self.register_buffer("ip_emb_ffn_ln_gammas", _zeros(config.embedding_size))
        self.register_buffer("ip_emb_ffn_ln_betas", _zeros(config.embedding_size))
        self.register_buffer(
            "smolgen_global_w",
            _zeros(
                config.board_squares * config.board_squares,
                config.smolgen_gen_size_per_head,
            ),
        )

        self.encoders = nn.ModuleList(
            [_EncoderLayerBuffers(config) for _ in range(config.num_encoders)]
        )
        self.policy_head = _PolicyHeadBuffers(config)
        self.value_head = _ValueHeadBuffers(config)
        self.moves_left_head = _MovesLeftHeadBuffers(config)
        self.post_init()

    @classmethod
    def from_weights_file(
        cls,
        weights_path: str | Path,
        *,
        source_root: str | Path | None = None,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "Lc0Bt4HFModel":
        from .weights import load_bt4_weights

        config_overrides.setdefault("source_weights_path", str(Path(weights_path).resolve()))
        config_overrides.setdefault(
            "lc0_source_root",
            str(Path(source_root).resolve()) if source_root is not None else None,
        )
        config_overrides.setdefault("move_strings", _load_move_strings(source_root))
        decoded = load_bt4_weights(weights_path, source_root=source_root, device="cpu")
        config = Lc0Bt4HFConfig.from_bt4_config(decoded.config, **config_overrides)
        model = cls(config)
        model.load_decoded_weights(decoded)
        return model.to(device)

    def load_decoded_weights(self, decoded_weights: Any) -> None:
        _copy_buffer(self, "pos_encoding", decoded_weights.pos_encoding)
        _copy_buffer(self, "policy_index_map", decoded_weights.policy_index_map)
        _copy_buffer(self, "ip_emb_pre_w", decoded_weights.ip_emb_pre_w)
        _copy_buffer(self, "ip_emb_pre_b", decoded_weights.ip_emb_pre_b)
        _copy_buffer(self, "ip_emb_w", decoded_weights.ip_emb_w)
        _copy_buffer(self, "ip_emb_b", decoded_weights.ip_emb_b)
        _copy_buffer(self, "ip_emb_ln_gammas", decoded_weights.ip_emb_ln_gammas)
        _copy_buffer(self, "ip_emb_ln_betas", decoded_weights.ip_emb_ln_betas)
        _copy_buffer(self, "ip_mult_gate", decoded_weights.ip_mult_gate)
        _copy_buffer(self, "ip_add_gate", decoded_weights.ip_add_gate)
        _copy_buffer(self, "ip_emb_ffn_dense1_w", decoded_weights.ip_emb_ffn_dense1_w)
        _copy_buffer(self, "ip_emb_ffn_dense1_b", decoded_weights.ip_emb_ffn_dense1_b)
        _copy_buffer(self, "ip_emb_ffn_dense2_w", decoded_weights.ip_emb_ffn_dense2_w)
        _copy_buffer(self, "ip_emb_ffn_dense2_b", decoded_weights.ip_emb_ffn_dense2_b)
        _copy_buffer(self, "ip_emb_ffn_ln_gammas", decoded_weights.ip_emb_ffn_ln_gammas)
        _copy_buffer(self, "ip_emb_ffn_ln_betas", decoded_weights.ip_emb_ffn_ln_betas)
        _copy_buffer(self, "smolgen_global_w", decoded_weights.smolgen_global_w)

        if len(decoded_weights.encoders) != len(self.encoders):
            raise ValueError(
                f"Expected {len(self.encoders)} encoders, got {len(decoded_weights.encoders)}"
            )
        for layer, layer_weights in zip(self.encoders, decoded_weights.encoders):
            _copy_buffer(layer, "q_w", layer_weights.q_w)
            _copy_buffer(layer, "q_b", layer_weights.q_b)
            _copy_buffer(layer, "k_w", layer_weights.k_w)
            _copy_buffer(layer, "k_b", layer_weights.k_b)
            _copy_buffer(layer, "v_w", layer_weights.v_w)
            _copy_buffer(layer, "v_b", layer_weights.v_b)
            _copy_buffer(layer, "dense_w", layer_weights.dense_w)
            _copy_buffer(layer, "dense_b", layer_weights.dense_b)
            _copy_buffer(layer, "ln1_gammas", layer_weights.ln1_gammas)
            _copy_buffer(layer, "ln1_betas", layer_weights.ln1_betas)
            _copy_buffer(layer, "ffn_dense1_w", layer_weights.ffn_dense1_w)
            _copy_buffer(layer, "ffn_dense1_b", layer_weights.ffn_dense1_b)
            _copy_buffer(layer, "ffn_dense2_w", layer_weights.ffn_dense2_w)
            _copy_buffer(layer, "ffn_dense2_b", layer_weights.ffn_dense2_b)
            _copy_buffer(layer, "ln2_gammas", layer_weights.ln2_gammas)
            _copy_buffer(layer, "ln2_betas", layer_weights.ln2_betas)
            _copy_buffer(layer, "rpe_q", layer_weights.rpe.q)
            _copy_buffer(layer, "rpe_k", layer_weights.rpe.k)
            _copy_buffer(layer, "rpe_v", layer_weights.rpe.v)

            _copy_buffer(layer.smolgen, "compress_w", layer_weights.smolgen.compress_w)
            _copy_buffer(layer.smolgen, "dense1_w", layer_weights.smolgen.dense1_w)
            _copy_buffer(layer.smolgen, "dense1_b", layer_weights.smolgen.dense1_b)
            _copy_buffer(layer.smolgen, "ln1_gammas", layer_weights.smolgen.ln1_gammas)
            _copy_buffer(layer.smolgen, "ln1_betas", layer_weights.smolgen.ln1_betas)
            _copy_buffer(layer.smolgen, "dense2_w", layer_weights.smolgen.dense2_w)
            _copy_buffer(layer.smolgen, "dense2_b", layer_weights.smolgen.dense2_b)
            _copy_buffer(layer.smolgen, "ln2_gammas", layer_weights.smolgen.ln2_gammas)
            _copy_buffer(layer.smolgen, "ln2_betas", layer_weights.smolgen.ln2_betas)

        _copy_buffer(self.policy_head, "ip_pol_w", decoded_weights.policy_head.ip_pol_w)
        _copy_buffer(self.policy_head, "ip_pol_b", decoded_weights.policy_head.ip_pol_b)
        _copy_buffer(self.policy_head, "ip2_pol_w", decoded_weights.policy_head.ip2_pol_w)
        _copy_buffer(self.policy_head, "ip2_pol_b", decoded_weights.policy_head.ip2_pol_b)
        _copy_buffer(self.policy_head, "ip3_pol_w", decoded_weights.policy_head.ip3_pol_w)
        _copy_buffer(self.policy_head, "ip3_pol_b", decoded_weights.policy_head.ip3_pol_b)
        _copy_buffer(self.policy_head, "ip4_pol_w", decoded_weights.policy_head.ip4_pol_w)

        _copy_buffer(self.value_head, "ip_val_w", decoded_weights.value_head.ip_val_w)
        _copy_buffer(self.value_head, "ip_val_b", decoded_weights.value_head.ip_val_b)
        _copy_buffer(self.value_head, "ip1_val_w", decoded_weights.value_head.ip1_val_w)
        _copy_buffer(self.value_head, "ip1_val_b", decoded_weights.value_head.ip1_val_b)
        _copy_buffer(self.value_head, "ip2_val_w", decoded_weights.value_head.ip2_val_w)
        _copy_buffer(self.value_head, "ip2_val_b", decoded_weights.value_head.ip2_val_b)

        _copy_buffer(self.moves_left_head, "ip_mov_w", decoded_weights.moves_left_head.ip_mov_w)
        _copy_buffer(self.moves_left_head, "ip_mov_b", decoded_weights.moves_left_head.ip_mov_b)
        _copy_buffer(self.moves_left_head, "ip1_mov_w", decoded_weights.moves_left_head.ip1_mov_w)
        _copy_buffer(self.moves_left_head, "ip1_mov_b", decoded_weights.moves_left_head.ip1_mov_b)
        _copy_buffer(self.moves_left_head, "ip2_mov_w", decoded_weights.moves_left_head.ip2_mov_w)
        _copy_buffer(self.moves_left_head, "ip2_mov_b", decoded_weights.moves_left_head.ip2_mov_b)

    def _smolgen_qk_bias(self, x: torch.Tensor, layer: _EncoderLayerBuffers) -> torch.Tensor:
        batch_size = x.shape[0]
        cfg = self.config
        compressed = _linear(x, layer.smolgen.compress_w, None, Activation.NONE)
        hidden = compressed.reshape(batch_size, BOARD_SQUARES * cfg.smolgen_hidden_channels)
        hidden = _linear(
            hidden,
            layer.smolgen.dense1_w,
            layer.smolgen.dense1_b,
            cfg.smolgen_activation,
        )
        hidden = _layer_norm_with_skip(
            hidden,
            alpha=1.0,
            skip=None,
            gammas=layer.smolgen.ln1_gammas,
            betas=layer.smolgen.ln1_betas,
            epsilon=1e-3,
        )
        out = _linear(
            hidden,
            layer.smolgen.dense2_w,
            layer.smolgen.dense2_b,
            cfg.smolgen_activation,
        )
        out = _layer_norm_with_skip(
            out,
            alpha=1.0,
            skip=None,
            gammas=layer.smolgen.ln2_gammas,
            betas=layer.smolgen.ln2_betas,
            epsilon=1e-3,
        )
        out = out.reshape(batch_size * cfg.num_heads, cfg.smolgen_gen_size_per_head)
        qk = _linear(out, self.smolgen_global_w, None, Activation.NONE)
        return qk.reshape(batch_size, cfg.num_heads, BOARD_SQUARES, BOARD_SQUARES)

    def _rpe_logits(self, x: torch.Tensor, rpe_weights: torch.Tensor, *, rpe_type: str) -> torch.Tensor:
        factorizer = _rpe_map().to(device=rpe_weights.device, dtype=rpe_weights.dtype)
        rpe = torch.matmul(rpe_weights, factorizer).reshape(
            self.config.head_dim,
            self.config.num_heads,
            BOARD_SQUARES,
            BOARD_SQUARES,
        )
        if rpe_type == "q":
            return torch.einsum("bhqd,dhqk->bhqk", x, rpe)
        if rpe_type == "k":
            return torch.einsum("bhkd,dhqk->bhqk", x, rpe)
        raise ValueError(f"Unsupported RPE type {rpe_type}")

    def _rpe_values(self, attention_weights: torch.Tensor, rpe_weights: torch.Tensor) -> torch.Tensor:
        factorizer = _rpe_map().to(device=rpe_weights.device, dtype=rpe_weights.dtype)
        rpe = torch.matmul(rpe_weights, factorizer).reshape(
            self.config.head_dim,
            self.config.num_heads,
            BOARD_SQUARES,
            BOARD_SQUARES,
        )
        return torch.einsum("bhqk,dhqk->bhqd", attention_weights, rpe)

    def _forward_encoder(self, x: torch.Tensor, layer: _EncoderLayerBuffers) -> torch.Tensor:
        cfg = self.config
        batch_size = x.shape[0]
        q = _linear(x, layer.q_w, layer.q_b, Activation.NONE)
        k = _linear(x, layer.k_w, layer.k_b, Activation.NONE)
        v = _linear(x, layer.v_w, layer.v_b, Activation.NONE)

        q = q.reshape(batch_size, BOARD_SQUARES, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, BOARD_SQUARES, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, BOARD_SQUARES, cfg.num_heads, cfg.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2))
        if cfg.use_rpe_q:
            attn = attn + self._rpe_logits(q, layer.rpe_q, rpe_type="q")
        if cfg.use_rpe_k:
            attn = attn + self._rpe_logits(k, layer.rpe_k, rpe_type="k")
        attn = attn * (1.0 / (cfg.head_dim ** 0.5))
        if cfg.use_smolgen:
            attn = attn + self._smolgen_qk_bias(x, layer)
        attention_weights = torch.softmax(attn, dim=-1)

        out = torch.matmul(attention_weights, v)
        if cfg.use_rpe_v:
            out = out + self._rpe_values(attention_weights, layer.rpe_v)
        out = out.transpose(1, 2).reshape(batch_size, BOARD_SQUARES, cfg.embedding_size)
        out = _linear(out, layer.dense_w, layer.dense_b, Activation.NONE)
        x = _layer_norm_with_skip(
            out,
            alpha=cfg.alpha,
            skip=x,
            gammas=layer.ln1_gammas,
            betas=layer.ln1_betas,
            epsilon=cfg.encoder_eps,
        )

        ffn = _linear(x, layer.ffn_dense1_w, layer.ffn_dense1_b, cfg.ffn_activation)
        ffn = _linear(ffn, layer.ffn_dense2_w, layer.ffn_dense2_b, Activation.NONE)
        return _layer_norm_with_skip(
            ffn,
            alpha=cfg.alpha,
            skip=x,
            gammas=layer.ln2_gammas,
            betas=layer.ln2_betas,
            epsilon=cfg.encoder_eps,
        )

    def _forward_policy_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        head = self.policy_head
        cfg = self.config

        emb = _linear(x, head.ip_pol_w, head.ip_pol_b, cfg.default_activation)
        q = _linear(emb, head.ip2_pol_w, head.ip2_pol_b, Activation.NONE)
        k = _linear(emb, head.ip3_pol_w, head.ip3_pol_b, Activation.NONE)

        flow = torch.matmul(q, k.transpose(-1, -2))
        flow = flow * (1.0 / (cfg.policy_d_model ** 0.5))

        prom_offsets = torch.matmul(k[:, 56:64, :], head.ip4_pol_w.transpose(0, 1))
        prom_offsets = prom_offsets.transpose(1, 2)
        prom_offsets = prom_offsets[:, :3, :] + prom_offsets[:, 3:4, :]
        prom = prom_offsets.transpose(1, 2).reshape(batch_size, 1, 24)

        sl = flow[:, 48:56, 56:64].reshape(batch_size, 64, 1)
        sl = sl.expand(-1, -1, 3).reshape(batch_size, 8, 24)
        prom = (sl + prom).reshape(batch_size, 3, 64)

        flow = torch.cat([flow, prom], dim=1).reshape(batch_size, 67 * 64)
        return flow.index_select(1, self.policy_index_map)

    def _forward_value_head(self, x: torch.Tensor) -> torch.Tensor:
        head = self.value_head
        val = _linear(x, head.ip_val_w, head.ip_val_b, self.config.default_activation)
        val = _linear(
            val.reshape(x.shape[0], -1),
            head.ip1_val_w,
            head.ip1_val_b,
            self.config.default_activation,
        )
        return _linear(val, head.ip2_val_w, head.ip2_val_b, Activation.NONE)

    def _forward_moves_left_head(self, x: torch.Tensor) -> torch.Tensor:
        head = self.moves_left_head
        mov = _linear(x, head.ip_mov_w, head.ip_mov_b, self.config.default_activation)
        mov = _linear(
            mov.reshape(x.shape[0], -1),
            head.ip1_mov_w,
            head.ip1_mov_b,
            self.config.default_activation,
        )
        return _linear(mov, head.ip2_mov_w, head.ip2_mov_b, Activation.RELU).squeeze(-1)

    def forward(
        self,
        input_planes: torch.Tensor | None = None,
        *,
        planes: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> Lc0Bt4HFOutput | tuple[torch.Tensor, ...]:
        if input_planes is None:
            input_planes = planes
        if input_planes is None:
            raise ValueError("input_planes must be provided.")

        if input_planes.ndim == 2:
            input_planes = input_planes.unsqueeze(0)
        if input_planes.ndim != 3 or input_planes.shape[1:] != (
            self.config.input_planes,
            self.config.board_squares,
        ):
            raise ValueError(
                "Expected input planes with shape "
                f"[batch, {self.config.input_planes}, {self.config.board_squares}], "
                f"got {tuple(input_planes.shape)}"
            )

        input_planes = input_planes.to(dtype=self.ip_emb_w.dtype)
        batch_size = input_planes.shape[0]
        cfg = self.config
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        pos_info = input_planes[:, :12, :].transpose(1, 2).reshape(batch_size, BOARD_SQUARES * 12)
        pos_info = _linear(pos_info, self.ip_emb_pre_w, self.ip_emb_pre_b, Activation.NONE)
        pos_info = pos_info.reshape(batch_size, BOARD_SQUARES, cfg.embedding_dense_size)

        board = input_planes.transpose(1, 2)
        x = torch.cat([board, pos_info], dim=-1)
        x = _linear(x, self.ip_emb_w, self.ip_emb_b, cfg.default_activation)
        x = _layer_norm_with_skip(
            x,
            alpha=1.0,
            skip=None,
            gammas=self.ip_emb_ln_gammas,
            betas=self.ip_emb_ln_betas,
            epsilon=cfg.embedding_eps,
        )
        x = x * self.ip_mult_gate.unsqueeze(0) + self.ip_add_gate.unsqueeze(0)

        ffn = _linear(
            x,
            self.ip_emb_ffn_dense1_w,
            self.ip_emb_ffn_dense1_b,
            cfg.ffn_activation,
        )
        ffn = _linear(
            ffn,
            self.ip_emb_ffn_dense2_w,
            self.ip_emb_ffn_dense2_b,
            Activation.NONE,
        )
        x = _layer_norm_with_skip(
            ffn,
            alpha=cfg.alpha,
            skip=x,
            gammas=self.ip_emb_ffn_ln_gammas,
            betas=self.ip_emb_ffn_ln_betas,
            epsilon=cfg.embedding_eps,
        )

        for layer in self.encoders:
            x = self._forward_encoder(x, layer)

        policy_logits = self._forward_policy_head(x)
        wdl_logits = self._forward_value_head(x)
        moves_left = self._forward_moves_left_head(x)

        if not return_dict:
            return x, policy_logits, wdl_logits, moves_left
        return Lc0Bt4HFOutput(
            last_hidden_state=x,
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            moves_left=moves_left,
        )

    def best_legal_move(
        self,
        input_planes: torch.Tensor,
        legal_moves: list[str],
        legal_policy_indices: list[int],
    ) -> str:
        logits = self(input_planes=input_planes).policy_logits[0]
        best_idx = max(
            range(len(legal_moves)),
            key=lambda idx: float(logits[legal_policy_indices[idx]]),
        )
        return legal_moves[best_idx]

    def _move_to_policy_index(self) -> dict[str, int]:
        lookup = getattr(self, "_move_to_policy_index_cache", None)
        if lookup is None:
            if not self.config.move_strings:
                source_root = getattr(self.config, "lc0_source_root", None)
                if source_root is None:
                    raise ValueError(
                        "This checkpoint does not contain lc0 move_strings metadata."
                    )
                self.config.move_strings = _load_move_strings(source_root)
            lookup = {
                move_str: idx for idx, move_str in enumerate(self.config.move_strings)
            }
            self._move_to_policy_index_cache = lookup
        return lookup

    def input_planes_from_fen(
        self,
        fen: str,
        moves: list[str] | None = None,
    ) -> torch.Tensor:
        input_format = getattr(self.config, "input_format", INPUT_CLASSICAL_112_PLANE)
        if input_format != INPUT_CLASSICAL_112_PLANE:
            raise NotImplementedError(
                f"Pure-Python encoding currently supports only input_format={INPUT_CLASSICAL_112_PLANE}, "
                f"got {input_format}"
            )
        return _encode_classical_112_planes(fen, moves)

    def legal_moves_and_policy_indices(
        self,
        fen: str,
        moves: list[str] | None = None,
    ) -> tuple[list[str], list[int]]:
        board = _build_history_boards(fen, moves)[-1]
        lookup = self._move_to_policy_index()
        move_and_index: list[tuple[str, int]] = []
        for move in board.legal_moves:
            move_key = _policy_move_key(move, board.turn)
            if move_key not in lookup:
                raise KeyError(f"Could not map move {move.uci()} to an lc0 policy index.")
            move_and_index.append((move.uci(), lookup[move_key]))
        move_and_index.sort(key=lambda item: item[1])
        legal_moves = [move for move, _ in move_and_index]
        policy_indices = [index for _, index in move_and_index]
        return legal_moves, policy_indices

    def best_legal_move_from_fen(
        self,
        fen: str,
        moves: list[str] | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> str:
        legal_moves, legal_policy_indices = self.legal_moves_and_policy_indices(fen, moves)
        input_planes = self.input_planes_from_fen(fen, moves)
        if device is None:
            device = self.ip_emb_w.device
        input_planes = input_planes.unsqueeze(0).to(device=device)
        return self.best_legal_move(input_planes, legal_moves, legal_policy_indices)


def convert_lc0_checkpoint_to_hf(
    weights_path: str | Path,
    save_directory: str | Path,
    *,
    source_root: str | Path | None = None,
    safe_serialization: bool = True,
) -> Path:
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    model = Lc0Bt4HFModel.from_weights_file(
        weights_path,
        source_root=source_root,
        device="cpu",
        source_weights_path=str(Path(weights_path).resolve()),
        lc0_source_root=str(Path(source_root).resolve()) if source_root is not None else None,
    )
    model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    return save_directory


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an lc0 BT4 protobuf weights file into a Transformers save_pretrained directory."
    )
    parser.add_argument("--weights", required=True, help="Path to the lc0 .pb.gz weights file.")
    parser.add_argument("--save-dir", required=True, help="Directory to write the HF checkpoint into.")
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional path to the lc0 source tree containing attention_policy_map.h.",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Write PyTorch .bin weights instead of safetensors.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_dir = convert_lc0_checkpoint_to_hf(
        args.weights,
        args.save_dir,
        source_root=args.source_root,
        safe_serialization=not args.no_safe_serialization,
    )
    print(output_dir)


Lc0Bt4HFConfig.register_for_auto_class("AutoConfig")
Lc0Bt4HFModel.register_for_auto_class("AutoModel")


if __name__ == "__main__":
    main()
