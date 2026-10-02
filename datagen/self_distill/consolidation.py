"""Batched, resumable Qwen consolidation of accepted recursive records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import time
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import chess

from datagen.self_distill.analysis import field_map
from datagen.self_distill.io import atomic_json
from datagen.self_distill.play_storage import fingerprint
from datagen.self_distill.schema import accepted_identity


PIECE_NAMES = {
    "P": "PAWN",
    "N": "KNIGHT",
    "B": "BISHOP",
    "R": "ROOK",
    "Q": "QUEEN",
    "K": "KING",
}
PIECE_TYPES = {
    "PAWN": chess.PAWN,
    "KNIGHT": chess.KNIGHT,
    "BISHOP": chess.BISHOP,
    "ROOK": chess.ROOK,
    "QUEEN": chess.QUEEN,
    "KING": chess.KING,
}
PROMOTIONS = {
    "QUEEN": chess.QUEEN,
    "ROOK": chess.ROOK,
    "BISHOP": chess.BISHOP,
    "KNIGHT": chess.KNIGHT,
}
POV_SQUARE = re.compile(r"<SQUARE_(\d+)>")
POV_PIECE = re.compile(r"<PIECE_([MO])([PNBRQK])>")
ABS_SQUARE_PAIR = re.compile(
    r"<SQUARE_([A-H][1-8])>\s*[-–—]\s*<SQUARE_([A-H][1-8])>"
)
ABS_MOVE = re.compile(
    r"<(?P<mover_colour>WHITE|BLACK)_"
    r"(?P<mover_piece>PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)>\s*"
    r"<SQUARE_(?P<source>[A-H][1-8])>\s*"
    r"(?:<(?P<captured_colour>WHITE|BLACK)_"
    r"(?P<captured_piece>PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)>\s*)?"
    r"<SQUARE_(?P<target>[A-H][1-8])>"
)
PROMOTION_TOKEN = re.compile(r"\s*<(WHITE|BLACK)_(QUEEN|ROOK|BISHOP|KNIGHT)>")


# prompt: preserve the exact faithful-editor recipe used for Sol-full-4.
COMMON_INSTRUCTIONS = """You are consolidating three recursive evaluations of a
single root chess position. You are a faithful editor and synthesizer, not an
independent chess analyst. Every conclusion, positional claim, candidate, tactic,
and move sequence in your answer must be grounded in the three supplied child
analyses. Do not add chess knowledge or repair their analysis yourself.

Each child position was reached by playing the labeled candidate from the root.
The analysis model then began afresh: its move numbers restart at 1 or 1..., and
its side to move is the root player's opponent. Every child evaluation is therefore
from the OPPONENT'S perspective, not the root player's perspective.

The numerical selection rule supplied in the input is mandatory. Choose the root
candidate with the LOWEST signed numerical evaluation for the opponent. For
example, -2.25 for the opponent is better for the root player than +1.03 or +2.68
for the opponent. Equivalently, negate the child evaluation to obtain the root
player's evaluation, then choose the highest result. Treat -M as lower than every
pawn value and +M as higher than every pawn value. Do not override this arithmetic
rule using the prose, even if a child's narrative sounds more favorable. BEST_MOVE
must be exactly the candidate marked MANDATORY BEST ROOT CANDIDATE in the input.

In the consolidated prose and structured fields, restore the root player's point
of view and restart move numbering from the root. A continuation through a child
must begin with the labeled root candidate before the child's line.

Preserve the established explanatory style and order. Begin with general
considerations and important features of the root position and its candidates.
Use later paragraphs for increasingly concrete plans, comparisons, tactics,
logic, and variations. Condense repeated motifs such as a common fork, outpost,
or maneuver, but preserve decisive tactics and forcing lines at full useful
length, even when they are long. Emphasize the best candidate more than inferior
ones. The result is a root analysis, not a broad summary of three positions.

Use only this absolute machine vocabulary:
- squares: <SQUARE_E4>;
- pieces: <WHITE_PAWN>, <BLACK_BISHOP>, and analogous colored piece tokens;
- players: <WHITE> and <BLACK>;
- files, ranks, and diagonals: <FILE_E>, <RANK_4>, <DIAGONAL_A1_H8>.
Never emit POV tokens such as <PIECE_MN> or numbered squares such as <SQUARE_29>.

Every complete move is written as adjacent tokens. A non-capture has exactly
three tokens, <PIECE><FROM_SQUARE><TO_SQUARE>. A capture has exactly four,
<PIECE><FROM_SQUARE><CAPTURED_PIECE><TO_SQUARE>. A promotion appends the promoted
piece as a rare fifth token. Apply this convention both in prose and in the
structured fields.

Return exactly these five fields and no surrounding commentary:
ANALYSIS:
<fluent grounded root analysis>
BEST_MOVE: <one tokenized root move>
CRITICAL_LINE: <one tokenized root line>
PROMISING_MOVES: <one to four tokenized legal root candidates, best first>
EVALUATION: <one sentence with a pawn or mate evaluation from the root player's perspective>

You have the complete three child analyses. Consolidate them to approximately
the length of one individual child analysis: compress overlap and inferior
branches to make room for the most important concrete detail."""

BOUNDARY_INSTRUCTIONS = """

Every CHILD CRITICAL_LINE below has been mechanically truncated immediately
before its first illegal move or its first move with a Stockfish-100k win-rate
drop of at least 10 percentage points. Each child block states the exact
boundary. When a line was truncated, omit the rejected move and every
continuation that depends on it from both the consolidated prose and structured
CRITICAL_LINE. Do not repair, reconstruct, extend, or independently analyze
that rejected continuation. You may retain and use the supplied clean prefix.

The consolidated CRITICAL_LINE must begin with the MANDATORY BEST ROOT
CANDIDATE and may continue only with a prefix of that candidate's supplied
clean child CRITICAL_LINE. If no child continuation remains, output only the
root candidate. PROMISING_MOVES may contain only the supplied legal root
candidates. The original root analysis is deliberately not supplied: combine
only the recursive child evidence.
"""

MATERIAL_HINT_INSTRUCTIONS = """

The root material inventory supplied in the input is a mechanical validation
fact. Preserve that inventory and minimally correct any material claim that
conflicts with it. This fact is an editing constraint; it is not permission to
introduce independent analysis.
"""


@dataclass(frozen=True)
class ConsolidationConfig:
    input: Path
    output: Path
    model: Path
    material_hints: bool = False
    chunk_size: int = 64
    max_model_len: int = 16_384
    max_output_tokens: int = 8_192
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    reasoning_effort: str = "low"

    def validate(self) -> None:
        if not self.input.is_file():
            raise FileNotFoundError(self.input)
        if not self.model.is_dir():
            raise FileNotFoundError(self.model)
        if self.chunk_size <= 0 or self.max_num_seqs <= 0:
            raise ValueError("chunk_size and max_num_seqs must be positive")
        if self.max_model_len <= 0 or self.max_output_tokens <= 0:
            raise ValueError("model length and output length must be positive")
        if self.max_output_tokens >= self.max_model_len:
            raise ValueError("max_output_tokens must be smaller than max_model_len")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def system_prompt(self) -> str:
        suffix = MATERIAL_HINT_INSTRUCTIONS if self.material_hints else ""
        return COMMON_INSTRUCTIONS + BOUNDARY_INSTRUCTIONS + suffix

    def semantic_settings(self) -> dict:
        return {
            "model": str(self.model.resolve()),
            "model_sha256": fingerprint(self.model),
            "material_hints": self.material_hints,
            "system_prompt_sha256": hashlib.sha256(
                self.system_prompt().encode()
            ).hexdigest(),
            "chunk_size": self.chunk_size,
            "max_model_len": self.max_model_len,
            "max_output_tokens": self.max_output_tokens,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.reasoning_effort,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_token(match: re.Match[str]) -> str:
    first, second = match.group(1), match.group(2)
    first_file, first_rank = first[0], int(first[1])
    second_file, second_rank = second[0], int(second[1])
    if first_file == second_file and {first_rank, second_rank} == {1, 8}:
        return f"<FILE_{first_file}>"
    if first_rank == second_rank and {first_file, second_file} == {"A", "H"}:
        return f"<RANK_{first_rank}>"
    first_square = chess.parse_square(first.lower())
    second_square = chess.parse_square(second.lower())
    file_distance = abs(chess.square_file(first_square) - chess.square_file(second_square))
    rank_distance = abs(chess.square_rank(first_square) - chess.square_rank(second_square))
    if file_distance == rank_distance:
        ends = []
        for delta in (-9, -7, 7, 9):
            square = first_square
            while True:
                following = square + delta
                if (
                    not 0 <= following < 64
                    or abs(chess.square_file(following) - chess.square_file(square)) != 1
                ):
                    break
                square = following
            ends.append(square)
        if second_square in ends:
            low, high = sorted((first, second))
            return f"<DIAGONAL_{low}_{high}>"
    return match.group(0)


def absolute_tokens(text: str, pov: chess.Color) -> str:
    """Translate POV model vocabulary into Qwen's absolute intermediate form."""

    def square(match: re.Match[str]) -> str:
        value = int(match.group(1)) - 1
        absolute = value if pov == chess.WHITE else value ^ 56
        return f"<SQUARE_{chess.square_name(absolute).upper()}>"

    def piece(match: re.Match[str]) -> str:
        colour = pov if match.group(1) == "M" else not pov
        side = "WHITE" if colour else "BLACK"
        return f"<{side}_{PIECE_NAMES[match.group(2)]}>"

    mover = "<WHITE>" if pov == chess.WHITE else "<BLACK>"
    opponent = "<BLACK>" if pov == chess.WHITE else "<WHITE>"
    translated = POV_SQUARE.sub(square, text)
    translated = POV_PIECE.sub(piece, translated)
    translated = translated.replace("<PLAYER>", mover).replace("<OPPONENT>", opponent)
    translated = re.sub(
        r"\b(?:the\s+)?player\b", mover, translated, flags=re.IGNORECASE
    )
    translated = re.sub(
        r"\b(?:the\s+)?opponent\b", opponent, translated, flags=re.IGNORECASE
    )
    return ABS_SQUARE_PAIR.sub(_line_token, translated)


def absolute_move_tokens(board: chess.Board, move: chess.Move) -> str:
    """Render one legal move using adjacent absolute piece and square tokens."""
    mover = board.piece_at(move.from_square)
    if mover is None:
        raise ValueError(f"no piece at {chess.square_name(move.from_square)}")
    side = "WHITE" if mover.color else "BLACK"
    parts = [
        f"<{side}_{PIECE_NAMES[mover.symbol().upper()] }>",
        f"<SQUARE_{chess.square_name(move.from_square).upper()}>",
    ]
    if board.is_en_passant(move):
        captured_side = "BLACK" if mover.color else "WHITE"
        parts.append(f"<{captured_side}_PAWN>")
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square)
        if victim is None:
            raise ValueError(f"capture has no victim: {move.uci()}")
        captured_side = "WHITE" if victim.color else "BLACK"
        parts.append(f"<{captured_side}_{PIECE_NAMES[victim.symbol().upper()]}>")
    parts.append(f"<SQUARE_{chess.square_name(move.to_square).upper()}>")
    if move.promotion:
        promoted = PIECE_NAMES[chess.piece_symbol(move.promotion).upper()]
        parts.append(f"<{side}_{promoted}>")
    return "".join(parts)


def material_inventory(board: chess.Board) -> str:
    """Describe exact root material and a conventional point-count balance."""
    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
    }
    sides = []
    for colour, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        counts = []
        for piece_type, name in names.items():
            count = len(board.pieces(piece_type, colour))
            counts.append(f"{count} {name if count == 1 else name + 's'}")
        sides.append(f"{label} has " + ", ".join(counts))
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    totals = {
        colour: sum(
            values[piece_type] * len(board.pieces(piece_type, colour))
            for piece_type in values
        )
        for colour in (chess.WHITE, chess.BLACK)
    }
    difference = totals[chess.WHITE] - totals[chess.BLACK]
    if difference == 0:
        balance = "equal material points"
    elif difference > 0:
        balance = f"White is ahead by {difference} material points"
    else:
        balance = f"Black is ahead by {-difference} material points"
    return "; ".join(sides) + f"; {balance}."


def _blocking_move(child: dict) -> dict | None:
    sanitation = child["sanitation"]
    candidates = [
        row
        for row in (sanitation.get("illegal_at"), sanitation.get("first_mistake"))
        if row is not None
    ]
    return min(candidates, key=lambda row: row["ply"]) if candidates else None


def child_status(child: dict) -> str:
    sanitation = child["sanitation"]
    stopped = _blocking_move(child)
    if stopped is None:
        return (
            "CRITICAL_LINE STATUS: complete supplied line; "
            f"{sanitation['accepted_ply']} clean ply retained."
        )
    rejected = stopped.get("uci") or stopped.get("token_text") or "unparseable move"
    if stopped["reason"] == "mistake":
        detail = f"a {100.0 * stopped['winrate_drop']:.2f}-point win-rate mistake"
    else:
        detail = "illegal"
    return (
        f"CRITICAL_LINE STATUS: truncated after {sanitation['accepted_ply']} clean ply; "
        f"the next move ({rejected}) was {detail}. Omit that rejected move and all "
        "later continuation; use only the retained clean prefix."
    )


def child_mate_note(text: str) -> str:
    value = field_map(text).get("EVALUATION", "")
    mate = re.search(r"(?i)([-+])\s*M(?:ATE)?\s*(\d+)", value)
    if mate is not None and mate.group(1) == "-":
        return (
            "\nMATE SCORE NOTE: The child side to move is getting mated. "
            "Remember to add 1 to the mate score at the parent."
        )
    return ""


def input_record_identity(record: dict) -> str:
    return accepted_identity(record)


def output_record_identity(record: dict) -> str:
    identity = record.get("record_id")
    if not identity:
        raise ValueError("consolidated record has no record_id")
    return identity


def build_case(record: dict, input_line: int, material_hints: bool) -> dict:
    """Render one accepted record as a terminal passthrough or Qwen prompt."""
    step = record["trajectory"][-1]
    board = chess.Board(step["fen"])
    parent = absolute_tokens(step["root_generation"], board.turn)
    base = {
        "input_line": input_line,
        "record_id": input_record_identity(record),
        "lineage_id": record["lineage_id"],
        "source_kind": record.get("source_kind", "unknown"),
        "fen": step["fen"],
        "recursion_depth": step["depth"],
        "terminal": bool(step.get("terminal") or not step.get("children")),
        "original_start_fen": record["start_fen"],
        "original_explanation": parent,
    }
    if base["terminal"]:
        return base

    selected_label = step["decision"]["selected_label"]
    selected = next(
        child for child in step["children"] if child["label"] == selected_label
    )
    blocks = []
    allowed_lines = {}
    child_records = []

    # children: translate each opponent-POV analysis and expose its clean line boundary.
    for child in sorted(step["children"], key=lambda row: "xyz".index(row["label"])):
        move = chess.Move.from_uci(child["move_uci"])
        move_tokens = absolute_move_tokens(board, move)
        child_pov = chess.Board(child["fen"]).turn
        generation = absolute_tokens(child["generation"], child_pov)
        status = absolute_tokens(child_status(child), child_pov)
        blocks.append(
            f"CHILD {child['label'].upper()}\n"
            f"ROOT CANDIDATE: {move_tokens} ({child['move_uci']})\n"
            f"CHILD FEN: {child['fen']}\n"
            "MODEL-IMPLIED ROOT WIN RATE: "
            f"{100.0 * child['predicted_root_winrate']:.2f}%\n"
            f"{status}\n"
            "CHILD ANALYSIS (opponent POV; move numbering refreshed):\n"
            f"{generation}{child_mate_note(generation)}"
        )
        accepted_uci = child["sanitation"]["accepted_uci"]
        allowed_lines[child["move_uci"]] = [child["move_uci"], *accepted_uci]
        child_records.append(
            {
                "label": child["label"],
                "move_uci": child["move_uci"],
                "fen": child["fen"],
                "predicted_root_winrate": child["predicted_root_winrate"],
                "critical_line_status": status,
                "accepted_uci": accepted_uci,
                "explanation": generation,
            }
        )

    # root: mandate the recursively selected move and optionally expose exact material.
    mandatory = chess.Move.from_uci(selected["move_uci"])
    mandatory_tokens = absolute_move_tokens(board, mandatory)
    side = "White" if board.turn == chess.WHITE else "Black"
    material = (
        f"ROOT MATERIAL INVENTORY: {material_inventory(board)}\n"
        if material_hints
        else ""
    )
    user_prompt = (
        f"ROOT FEN: {step['fen']}\nROOT SIDE TO MOVE: {side}\n"
        f"{material}\n"
        f"MANDATORY BEST ROOT CANDIDATE: {mandatory_tokens} "
        f"({selected['move_uci']})\n\n"
        + "\n\n".join(blocks)
    )
    return {
        **base,
        "acceptance_reason": step["decision"]["reason"],
        "mandatory_best_uci": selected["move_uci"],
        "mandatory_best_tokens": mandatory_tokens,
        "allowed_lines": allowed_lines,
        "children": child_records,
        "user_prompt": user_prompt,
    }


def _capture_error(
    board: chess.Board,
    move: chess.Move,
    match: re.Match[str],
) -> str | None:
    actual = None
    if board.is_en_passant(move):
        actual = chess.Piece(chess.PAWN, not board.turn)
    elif board.is_capture(move):
        actual = board.piece_at(move.to_square)
    declared_piece = match.group("captured_piece")
    if actual is None:
        return "capture token on a non-capture" if declared_piece else None
    if declared_piece is None:
        return "capture omitted its captured-piece token"
    declared_colour = match.group("captured_colour") == "WHITE"
    if (
        actual.color != declared_colour
        or actual.piece_type != PIECE_TYPES[declared_piece]
    ):
        return "captured-piece token does not match the board"
    return None


def _parsed_move(
    match: re.Match[str],
    board: chess.Board,
    text: str,
) -> tuple[chess.Move | None, str | None]:
    source = chess.parse_square(match.group("source").lower())
    target = chess.parse_square(match.group("target").lower())
    piece = board.piece_at(source)
    colour = match.group("mover_colour") == "WHITE"
    if (
        piece is None
        or piece.color != colour
        or piece.piece_type != PIECE_TYPES[match.group("mover_piece")]
    ):
        return None, "mover token does not match the board"
    promotion = None
    if piece.piece_type == chess.PAWN and chess.square_rank(target) in (0, 7):
        suffix = PROMOTION_TOKEN.match(text, match.end())
        promotion = PROMOTIONS[suffix.group(2)] if suffix else chess.QUEEN
    move = chess.Move(source, target, promotion=promotion)
    if move not in board.legal_moves:
        return None, "move is illegal"
    error = _capture_error(board, move, match)
    return (None, error) if error else (move, None)


def replay_line(fen: str, text: str) -> tuple[list[str], str | None]:
    board = chess.Board(fen)
    observed = []
    for match in ABS_MOVE.finditer(text):
        move, error = _parsed_move(match, board, text)
        if move is None:
            return observed, error
        observed.append(move.uci())
        board.push(move)
    return observed, None if observed else "no parseable moves"


def root_moves(fen: str, text: str) -> tuple[list[str], list[str]]:
    board = chess.Board(fen)
    moves = []
    errors = []
    for index, match in enumerate(ABS_MOVE.finditer(text), 1):
        move, error = _parsed_move(match, board.copy(stack=False), text)
        if move is None:
            errors.append(f"candidate {index}: {error}")
        else:
            moves.append(move.uci())
    if not moves and not errors:
        errors.append("no parseable moves")
    return moves, errors


def audit(case: dict, output: str) -> list[str]:
    """Validate structure, legality, supplied candidates, and clean-line fidelity."""
    issues = []
    fields = field_map(output)
    required = {"ANALYSIS", "BEST_MOVE", "CRITICAL_LINE", "PROMISING_MOVES", "EVALUATION"}
    missing = sorted(required - fields.keys())
    if missing:
        issues.append(f"missing fields: {', '.join(missing)}")
    if re.search(r"<PIECE_[MO][PNBRQK]>|<SQUARE_\d+>", output):
        issues.append("response contains POV vocabulary")

    # critical line: require the mandatory move followed only by its supplied clean prefix.
    critical, critical_error = replay_line(case["fen"], fields.get("CRITICAL_LINE", ""))
    if critical_error:
        issues.append(f"critical line: {critical_error}")
    allowed = case["allowed_lines"].get(critical[0]) if critical else None
    if allowed is None or critical != allowed[: len(critical)]:
        issues.append(f"critical line {critical} is not a supplied clean prefix")
    if not critical or critical[0] != case["mandatory_best_uci"]:
        issues.append("critical line does not begin with the mandatory candidate")

    # root fields: require the selected best move and only supplied legal candidates.
    best, best_errors = root_moves(case["fen"], fields.get("BEST_MOVE", ""))
    if best_errors or best[:1] != [case["mandatory_best_uci"]]:
        issues.append("BEST_MOVE does not match the mandatory candidate")
    promising, promising_errors = root_moves(
        case["fen"], fields.get("PROMISING_MOVES", "")
    )
    issues.extend(f"promising moves: {error}" for error in promising_errors)
    allowed_roots = set(case["allowed_lines"])
    if any(move not in allowed_roots for move in promising):
        issues.append("PROMISING_MOVES contains an unsupplied candidate")
    if promising and promising[0] != case["mandatory_best_uci"]:
        issues.append("PROMISING_MOVES does not put the mandatory candidate first")
    return issues


def strip_thinking(text: str) -> str:
    return text.split("</think>", 1)[1].lstrip() if "</think>" in text else text


def _input_inventory(path: Path) -> tuple[list[str], int, str]:
    order = []
    rows = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            order.append(input_record_identity(json.loads(line)))
    if len(set(order)) != rows:
        raise ValueError(f"input contains only {len(set(order))} identities for {rows} rows")
    return order, rows, _file_sha256(path)


def _pending_cases(
    path: Path,
    completed: set[str],
    material_hints: bool,
) -> Iterator[dict]:
    with path.open() as handle:
        for input_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if input_record_identity(record) not in completed:
                yield build_case(record, input_line, material_hints)


def _write_chunk(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _chunk_index(chunks: Path) -> tuple[dict, dict, dict]:
    index = {}
    digests = {}
    statistics = {
        "records": 0,
        "generated_records": 0,
        "terminal_records": 0,
        "output_tokens": 0,
        "truncated_records": 0,
        "records_with_audit_issues": 0,
        "audit_issues": 0,
        "audit_issue_categories": Counter(),
    }

    # Reconstruct the in-memory index once on startup/resume.
    for path in sorted(chunks.glob("chunk_*.jsonl")):
        _index_chunk(path, index, statistics, digests)
    return index, statistics, digests


def _index_chunk(path: Path, index: dict, statistics: dict, digests: dict) -> None:
    """Add one published chunk, checking duplicates and counting new records only."""
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            row = json.loads(raw)
            identity = output_record_identity(row)
            digest = hashlib.sha256(raw).hexdigest()
            if identity in digests and digests[identity] != digest:
                raise ValueError(f"conflicting completed chunks for {identity}")
            if identity in index:
                continue
            digests[identity] = digest
            index[identity] = (path, offset, len(raw))
            statistics["records"] += 1
            generated = row.get("mode") == "qwen_consolidation"
            statistics["generated_records"] += generated
            statistics["terminal_records"] += not generated
            statistics["output_tokens"] += row.get("output_tokens", 0)
            statistics["truncated_records"] += row.get("finish_reason") == "length"
            issues = row.get("audit_issues") or []
            statistics["records_with_audit_issues"] += bool(issues)
            statistics["audit_issues"] += len(issues)
            statistics["audit_issue_categories"].update(
                issue.split(":", 1)[0] for issue in issues
            )


def _assemble(
    index: dict[str, tuple[Path, int, int]],
    output: Path,
    required_order: list[str],
) -> str:
    missing = [identity for identity in required_order if identity not in index]
    if missing:
        raise RuntimeError(f"assembly is missing {len(missing)} required records")
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    digest = hashlib.sha256()
    current_path = None
    source = None
    with temporary.open("wb") as destination:
        for identity in required_order:
            path, offset, length = index[identity]
            if path != current_path:
                if source is not None:
                    source.close()
                source = path.open("rb")
                current_path = path
            source.seek(offset)
            raw = source.read(length)
            destination.write(raw)
            digest.update(raw)
        if source is not None:
            source.close()
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(output)
    return digest.hexdigest()


class Consolidator:
    def __init__(self, config: ConsolidationConfig):
        config.validate()
        self.config = config
        self.system_prompt = config.system_prompt()
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self.stop_requested = False

    def _request_stop(self, _signum, _frame) -> None:
        self.stop_requested = True
        print("[signal] stopping after the current atomic chunk", flush=True)

    def _load_model(self) -> float:
        if self.llm is not None:
            return 0.0
        from vllm import LLM, SamplingParams

        # model setup: reproduce the Qwen3.8 Sol-full-4 inference settings.
        started = time.perf_counter()
        self.llm = LLM(
            model=str(self.config.model),
            tensor_parallel_size=1,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
            language_model_only=True,
            enable_prefix_caching=False,
            enforce_eager=self.config.enforce_eager,
            gdn_prefill_backend="triton",
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_output_tokens,
        )
        return time.perf_counter() - started

    def _generate(self, cases: list[dict]) -> tuple[dict[int, Any], float, float]:
        if not cases:
            return {}, 0.0, 0.0
        load_seconds = self._load_model()
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": case["user_prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=self.config.reasoning_effort,
            )
            for case in cases
        ]
        started = time.perf_counter()
        generations = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        elapsed = time.perf_counter() - started
        return (
            dict(
                zip(
                    (case["input_line"] for case in cases),
                    generations,
                    strict=True,
                )
            ),
            load_seconds,
            elapsed,
        )

    def _rows(self, batch: list[dict], generations: dict[int, Any]) -> list[dict]:
        rows = []
        for case in batch:
            if case["terminal"]:
                rows.append(
                    {
                        **case,
                        "mode": "terminal_passthrough",
                        "consolidated_explanation": case["original_explanation"],
                        "raw_generation": None,
                        "finish_reason": "terminal",
                        "output_tokens": 0,
                        "audit_issues": [],
                    }
                )
                continue
            result = generations[case["input_line"]]
            raw = result.outputs[0].text
            output = strip_thinking(raw)
            rows.append(
                {
                    **case,
                    "mode": "qwen_consolidation",
                    "consolidated_explanation": output,
                    "raw_generation": raw,
                    "finish_reason": result.outputs[0].finish_reason,
                    "output_tokens": len(result.outputs[0].token_ids),
                    "audit_issues": audit(case, output),
                }
            )
        return rows

    def run(self) -> dict:
        order, total, input_sha256 = _input_inventory(self.config.input)
        required = set(order)
        output = self.config.output
        chunks = output / "chunks"
        output.mkdir(parents=True, exist_ok=True)
        chunks.mkdir(exist_ok=True)
        manifest_path = output / "manifest.json"
        settings = self.config.semantic_settings()
        run_identity = hashlib.sha256(
            json.dumps(
                {"input_sha256": input_sha256, "settings": settings},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        # resume: require the same immutable input and complete generation settings.
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if prior and prior.get("run_identity") != run_identity:
            raise RuntimeError("existing consolidation manifest does not match this run")
        if not prior and any(chunks.iterdir()):
            raise RuntimeError("consolidation chunks exist without a matching manifest")
        model_load_seconds = prior.get("model_load_seconds", 0.0)
        generation_seconds = prior.get("generation_seconds", 0.0)
        wall_seconds = prior.get("wall_seconds", 0.0)
        index, statistics, digests = _chunk_index(chunks)
        completed = required & index.keys()

        def publish_manifest(complete: bool, output_sha256: str | None = None) -> dict:
            generated = statistics["generated_records"]
            tokens = statistics["output_tokens"]
            manifest = {
                "schema_version": 1,
                "mode": "consolidate",
                "run_identity": run_identity,
                "input": {
                    "path": str(self.config.input.resolve()),
                    "sha256": input_sha256,
                    "records": total,
                },
                "output": str(output.resolve()),
                "model": str(self.config.model.resolve()),
                "prompt_profile": (
                    "sol-full-4+material-hints"
                    if self.config.material_hints
                    else "sol-full-4"
                ),
                "system_prompt": self.system_prompt,
                "settings": settings,
                "completed_records": len(completed),
                "complete": complete,
                "model_load_seconds": model_load_seconds,
                "generation_seconds": generation_seconds,
                "wall_seconds": wall_seconds,
                "statistics": {
                    **statistics,
                    "truncation_rate": (
                        statistics["truncated_records"] / generated if generated else 0.0
                    ),
                    "records_per_generation_second": (
                        generated / generation_seconds if generation_seconds else 0.0
                    ),
                    "tokens_per_generation_second": (
                        tokens / generation_seconds if generation_seconds else 0.0
                    ),
                },
            }
            if output_sha256 is not None:
                manifest["consolidated_sha256"] = output_sha256
            atomic_json(manifest_path, manifest)
            return manifest

        publish_manifest(required <= index.keys())
        if required <= index.keys():
            output_sha256 = _assemble(index, output / "consolidated.jsonl", order)
            manifest = publish_manifest(True, output_sha256)
            print(f"[done] {total}/{total} records already complete", flush=True)
            return manifest

        signal.signal(signal.SIGTERM, self._request_stop)
        iterator = _pending_cases(
            self.config.input,
            set(completed),
            self.config.material_hints,
        )
        started = time.perf_counter()

        # generation: write each maximum-sized batch as one immutable atomic chunk.
        while not self.stop_requested:
            batch = list(islice(iterator, self.config.chunk_size))
            if not batch:
                break
            generated_cases = [case for case in batch if not case["terminal"]]
            generations, load_elapsed, generation_elapsed = self._generate(generated_cases)
            model_load_seconds += load_elapsed
            generation_seconds += generation_elapsed
            rows = self._rows(batch, generations)
            first, last = batch[0]["input_line"], batch[-1]["input_line"]
            identity_digest = hashlib.sha256(
                "".join(row["record_id"] for row in rows).encode()
            ).hexdigest()[:12]
            chunk_path = chunks / f"chunk_{first:08d}_{last:08d}_{identity_digest}.jsonl"
            _write_chunk(chunk_path, rows)
            _index_chunk(chunk_path, index, statistics, digests)
            completed = required & index.keys()
            wall_seconds = prior.get("wall_seconds", 0.0) + time.perf_counter() - started
            publish_manifest(required <= index.keys())
            print(
                f"[progress] {len(completed)}/{total} wrote {chunk_path.name}",
                flush=True,
            )

        wall_seconds = prior.get("wall_seconds", 0.0) + time.perf_counter() - started
        completed = required & index.keys()
        if required <= index.keys():
            output_sha256 = _assemble(index, output / "consolidated.jsonl", order)
            manifest = publish_manifest(True, output_sha256)
            print(f"[done] wrote {output / 'consolidated.jsonl'}", flush=True)
            return manifest
        manifest = publish_manifest(False)
        print(f"[resume] stopped safely at {len(completed)}/{total}", flush=True)
        return manifest
