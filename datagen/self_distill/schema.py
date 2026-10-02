"""Shared position identities and seed-record normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chess


HISTORY_PLIES = 7


def normalized_fen(fen: str) -> str:
    """Return the position fields of a FEN without its move counters."""
    board = chess.Board(fen)
    return " ".join(board.fen().split()[:4])


def stable_digest(value: str, size: int = 16) -> str:
    return hashlib.blake2b(value.encode(), digest_size=size).hexdigest()


def _legacy_history(start_fen: str, moves: list[str], end_fen: str) -> list[str]:
    board = chess.Board(start_fen)
    history = []
    for move in moves:
        history.append(board.fen())
        board.push_uci(move)
    if normalized_fen(board.fen()) != normalized_fen(end_fen):
        raise ValueError("legacy move history does not reach its end FEN")
    return history[-HISTORY_PLIES:]


def normalize_seed(value: Any, source_path: Path, line_number: int) -> dict:
    """Normalize named and legacy-list position records to the seed schema."""
    if isinstance(value, list):
        if len(value) < 3:
            raise ValueError(f"legacy row {source_path}:{line_number} has fewer than 3 fields")
        start_fen, moves, fen = value[:3]
        if not isinstance(moves, list):
            raise ValueError(f"legacy moves are not a list at {source_path}:{line_number}")
        history = _legacy_history(start_fen, moves, fen)
        extra = {
            "source_kind": "game",
            "source_path": str(source_path),
            "source_line": line_number,
            "legacy_start_fen": start_fen,
            "legacy_moves": moves,
        }
        if len(value) > 3:
            extra["legacy_payload"] = value[3:]
        record = {"fen": fen, "history": history, "extra": extra}
    elif isinstance(value, dict):
        if "fen" not in value:
            raise ValueError(f"row {source_path}:{line_number} has no fen")
        record = dict(value)
        record["history"] = list(record.get("history") or [])[-HISTORY_PLIES:]
        record["extra"] = dict(record.get("extra") or {})
        record["extra"].setdefault("source_path", str(source_path))
        record["extra"].setdefault("source_line", line_number)
    else:
        raise ValueError(f"row {source_path}:{line_number} is not an object or legacy list")

    position = normalized_fen(record["fen"])
    for history_fen in record["history"]:
        chess.Board(history_fen)
    record["extra"]["normalized_fen"] = position
    record.setdefault("record_id", f"position_{stable_digest(position)}")
    return record


def position_identity(record: dict) -> str:
    return stable_digest(normalized_fen(record["fen"]))


def source_identity(record: dict) -> str:
    """Return a stable source identity used in manifests and diagnostics."""
    extra = record.get("extra") or {}
    source = {
        "record_id": record.get("record_id") or position_identity(record),
        "source_kind": extra.get("source_kind"),
        "puzzle_id": extra.get("puzzle_id"),
        "game": extra.get("game"),
        "ply": extra.get("ply"),
    }
    return stable_digest(json.dumps(source, sort_keys=True, separators=(",", ":")))


def accepted_identity(record: dict) -> str:
    """Identify an accepted lineage independently of shard-local numbering."""
    source = record.get("source") or {}
    start_fen = record.get("start_fen") or source.get("fen")
    if start_fen is None:
        raise ValueError("accepted record has no start position")
    return stable_digest(
        f"{position_identity({'fen': start_fen})}|{source_identity(source)}"
    )


def partition_for(identity: str, parts: int) -> int:
    if parts <= 0:
        raise ValueError("parts must be positive")
    return int(identity, 16) % parts
