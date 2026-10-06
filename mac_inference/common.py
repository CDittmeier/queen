import hashlib
import json
import re
from pathlib import Path

import chess
import torch
from transformers import AutoTokenizer

from models.encoder import Lc0Bt4HFModel
from utils.lc0_planes import encode_fen_batch
from utils.translate_helpers import Translator
from utils.utils import encode_planes

MODEL_REVISIONS = {
    "pawn-8": (
        "princeton-nlp/queen_pawn-8",
        "5027687fac08b64b2403b13aca5b198e68294fe9",
    ),
}


def verify_release(directory: Path) -> None:
    """Check the pinned download against its published release manifest."""
    manifest = json.loads((directory / "release.json").read_text())
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or incomplete model file: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")


def download_model(directory: Path, model: str) -> None:
    from huggingface_hub import snapshot_download

    repo_id, revision = MODEL_REVISIONS[model]
    snapshot_download(repo_id, revision=revision, local_dir=directory, token=False)
    verify_release(directory)


def prompt_ids(directory: Path, prompt: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    settings = json.loads((directory / "inference_config.json").read_text())
    prompt = settings["prompt"] if prompt is None else prompt
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer, ids, prompt


class BoardEncoder:
    def __init__(self, directory: Path, device: str = "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU is unavailable; use an Apple silicon Mac.")
        self.device = device
        # Run the upstream LC0 encoder in FP32; decoder and bridges use BF16.
        self.model = (
            Lc0Bt4HFModel.from_pretrained(
                directory / "lc0",
                local_files_only=True,
            )
            .float()
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def encode(self, fen: str, history: list[str]) -> torch.Tensor:
        planes = encode_fen_batch([fen], [history]).to(self.device)
        states = encode_planes(self.model, planes, torch.float32, pov=True)
        if states.shape != (1, 16, 64, 1024) or not torch.isfinite(states).all():
            raise RuntimeError("Invalid LC0 encoder states")
        return states


def request_seed(seed: int, game_id: int, board: chess.Board) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{game_id}|{board.ply()}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


# The released model emits POV tokens, anchored to the ROOT side to move.
_BEST = re.compile(r"(?m)^BEST_MOVE:\s*(.*)$")
_SQUARE = re.compile(r"<SQUARE_(\d+)>")
_MOVE = re.compile(
    r"<PIECE_([MO])([PNBRQK])>\s*<SQUARE_(\d+)>\s*"
    r"(?:<PIECE_([MO])([PNBRQK])>\s*)?<SQUARE_(\d+)>"
    r"(?:\s*<PIECE_([MO])([QRBN])>)?"
)
_PIECES = dict(zip("PNBRQK", chess.PIECE_TYPES, strict=True))


def best_move(board: chess.Board, raw: str) -> chess.Move | None:
    """Only report an explicit, legal BEST_MOVE. Never invent a random fallback."""
    field = _BEST.search(raw)
    match = _MOVE.fullmatch(field[1].strip()) if field else None
    if match is None:
        return None
    side, piece_type, source, capture_side, capture_type, target, promo_side, promo = (
        match.groups()
    )
    source, target = int(source) - 1, int(target) - 1
    if source not in range(64) or target not in range(64) or side != "M":
        return None
    if board.turn == chess.BLACK:
        source, target = source ^ 56, target ^ 56
    piece = board.piece_at(source)
    if piece != chess.Piece(_PIECES[piece_type], board.turn):
        return None
    if promo and (piece.piece_type != chess.PAWN or promo_side != "M"):
        return None
    promotion = _PIECES[promo] if promo else None
    if piece.piece_type == chess.PAWN and chess.square_rank(target) in (0, 7):
        if promotion is None:
            return None  # An unfinished promotion may have meant an underpromotion.
    move = chess.Move(source, target, promotion=promotion)
    if move not in board.legal_moves:
        return None
    captured = (
        chess.Piece(chess.PAWN, not board.turn)
        if board.is_en_passant(move)
        else board.piece_at(target)
    )
    if capture_type:
        if capture_side != "O" or captured != chess.Piece(
            _PIECES[capture_type],
            not board.turn,
        ):
            return None
    elif captured is not None:
        return None
    return move


def result(board: chess.Board, raw: str, **metadata) -> dict:
    move = best_move(board, raw)
    # The upstream translator assumes every square is in its 64-token vocabulary.
    # Mark malformed square references in displayed prose; retain raw text exactly.
    display_raw = _SQUARE.sub(
        lambda match: (
            match[0] if 1 <= int(match[1]) <= 64 else f"[invalid square {match[1]}]"
        ),
        raw,
    )
    return {
        "fen": board.fen(),
        "text": Translator(board.turn).decode_absolute(display_raw),
        "raw_text": raw,
        "best_move_uci": move.uci() if move else None,
        "best_move_san": board.san(move) if move else None,
        "move_source": "best_move" if move else "missing_or_invalid_best_move",
        **metadata,
    }
