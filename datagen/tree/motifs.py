"""Normalized positional, material, and tactical motif detection.

This module is the data-facing facade for the stage-5 motif task.  Detectors
return chess-native dictionaries only: colors are absolute, squares are
algebraic, and moves are UCI.  POV conversion and prose belong to the task
builder, so mined JSONL remains reusable when wording changes.
"""
from __future__ import annotations

from collections.abc import Callable

import chess

from datagen.tree.primitives import activity
from datagen.tree.primitives.core.context import Context, get_context
from datagen.tree.primitives.core.score import SCORE_ZERO
from datagen.tree.primitives.mobility import MOBILITY_BONUS
from datagen.tree.primitives.passed import passed_pawn_score
from datagen.tree.primitives.pawns import pawn_score
from datagen.tree.primitives.pieces import piece_score


_COLORS = (chess.WHITE, chess.BLACK)
_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_MATERIAL_TYPES = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)
_MATERIAL_VALUES = {
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.PAWN: 1,
}
_TACTIC_NAMES = {
    "attraction": "attraction",
    "deflection": "deflection",
    "hangingPiece": "hanging_piece",
    "trappedPiece": "trapped_piece",
    "skewer": "skewer",
    "interference": "interference",
    "intermezzo": "intermezzo",
    "pin": "pin",
    "xRayAttack": "x_ray_attack",
    "collinearMove": "collinear_move",
    "backRankMate": "back_rank_mate",
}


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _square_names(squares) -> list[str]:
    return [chess.square_name(square) for square in sorted(squares)]


def _pieces(board: chess.Board, color: chess.Color, piece_type: int,
            squares) -> list[dict]:
    return [
        {
            "color": _color_name(color),
            "piece": _PIECE_NAMES[piece_type],
            "square": chess.square_name(square),
        }
        for square in sorted(squares)
        if (board.piece_type_at(square) == piece_type
            and board.color_at(square) == color)
    ]


def _record(name: str, source: str, metadata: dict) -> dict:
    return {"motif": name, "metadata": {"source": source, **metadata}}


def _pawn_structure(board: chess.Board, ctx: Context,
                    color: chess.Color) -> tuple[float, dict]:
    net = activity.pawn_structure(board, ctx)
    weaker = chess.BLACK if net > 0 else chess.WHITE
    if not net or color != weaker:
        return 0.0, {}
    factors: dict[str, list[int]] = {}
    for rec in ctx.pawn[color].pawns:
        for name, score in pawn_score(rec)[1]:
            if name in ("isolated", "doubled"):
                factors.setdefault(name, []).append(rec.sq)
    if not factors:
        return 0.0, {}
    return abs(net), {
        "net_score_white": net,
        "factors": [
            {"name": name, "squares": _square_names(factors[name])}
            for name in ("isolated", "doubled")
            if name in factors
        ],
    }


def _knight_outpost(board: chess.Board, ctx: Context,
                    color: chess.Color) -> tuple[float, dict]:
    found = [(square, score) for side, square, score
             in activity.outpost_knights(board, ctx) if side == color]
    value = sum(max(0.0, activity.felt(board, ctx, score))
                for _square, score in found)
    squares = [square for square, _score in found]
    return value, {
        "pieces": _pieces(board, color, chess.KNIGHT, squares),
        "squares": _square_names(squares),
    }


def _active_bishop_pair(board: chess.Board, ctx: Context,
                        color: chess.Color) -> tuple[float, dict]:
    recs = activity.bishop_pair(ctx, color) or []
    value = sum(max(0.0, activity.felt(
        board, ctx, MOBILITY_BONUS[chess.BISHOP][rec.mob_count]
    )) for rec in recs)
    squares = [rec.sq for rec in recs]
    return value, {
        "pieces": _pieces(board, color, chess.BISHOP, squares),
        "squares": _square_names(squares),
    }


def _active_bishop(board: chess.Board, ctx: Context,
                   color: chess.Color) -> tuple[float, dict]:
    recs = list(ctx.piece_recs(color, chess.BISHOP))
    value = activity.lone_bishop_activity(board, ctx, color)
    squares = [rec.sq for rec in recs] if value else []
    return value, {
        "pieces": _pieces(board, color, chess.BISHOP, squares),
        "squares": _square_names(squares),
    }


def _bad_bishop(board: chess.Board, ctx: Context,
                color: chess.Color) -> tuple[float, dict]:
    found = activity.bad_bishops(board, ctx, color)
    value = activity.bishop_badness(board, ctx, color)
    squares = [square for square, _count, _score in found]
    return value, {
        "pieces": _pieces(board, color, chess.BISHOP, squares),
        "squares": _square_names(squares),
        "own_color_pawns": {
            chess.square_name(square): count
            for square, count, _score in found
        },
    }


def _bad_knight(board: chess.Board, ctx: Context,
                color: chess.Color) -> tuple[float, dict]:
    found = activity.bad_knights(board, ctx, color)
    value = activity.knight_badness(board, ctx, color)
    squares = [square for square, _mobility in found]
    return value, {
        "pieces": _pieces(board, color, chess.KNIGHT, squares),
        "squares": _square_names(squares),
        "mobility": {
            chess.square_name(square): mobility
            for square, mobility in found
        },
    }


def _bad_rook(board: chess.Board, ctx: Context,
              color: chess.Color) -> tuple[float, dict]:
    found = activity.bad_rooks(board, ctx, color)
    value = activity.rook_badness(board, ctx, color)
    squares = [square for square, _mobility in found]
    return value, {
        "pieces": _pieces(board, color, chess.ROOK, squares),
        "squares": _square_names(squares),
        "mobility": {
            chess.square_name(square): mobility
            for square, mobility in found
        },
    }


def _active_rook(board: chess.Board, ctx: Context,
                 color: chess.Color) -> tuple[float, dict]:
    value = activity.rook_activity(board, ctx, color)
    recs = list(ctx.piece_recs(color, chess.ROOK)) if value else []
    squares = [rec.sq for rec in recs]
    files = []
    for rec in recs:
        bonus = activity.rook_file_bonus(board, color, rec.sq)
        if bonus is not None:
            enemy_pawns = board.pawns & board.occupied_co[not color]
            kind = (
                "semi-open"
                if enemy_pawns & chess.BB_FILES[rec.sq & 7]
                else "open"
            )
            files.append({"square": chess.square_name(rec.sq), "kind": kind})
    return value, {
        "pieces": _pieces(board, color, chess.ROOK, squares),
        "squares": _square_names(squares),
        "files": files,
    }


def _colour_complexion(board: chess.Board, ctx: Context,
                       color: chess.Color) -> tuple[float, dict]:
    total = SCORE_ZERO
    for rec in ctx.piece_recs(color, chess.BISHOP):
        for name, score in piece_score(board, ctx, rec)[1]:
            if name == "bishop pawns":
                total = total + score
    value = max(0.0, -activity.felt(board, ctx, total))
    light = activity.weak_colour(board, ctx, color)
    return value, {
        "complex": None if light is None else ("light" if light else "dark"),
    }


def _space(board: chess.Board, ctx: Context,
           color: chess.Color) -> tuple[float, dict]:
    net = activity.space(board, ctx)
    favored = chess.WHITE if net > 0 else chess.BLACK
    if not net or color != favored:
        return 0.0, {}
    regions = {
        phrase.removeprefix("on the ").removeprefix("in the "): activity.space_region_count(
            board, ctx, color, files
        )
        for phrase, files in activity.SPACE_REGIONS
    }
    return abs(net), {"net_score_white": net, "regions": regions}


def _passed_pawns(board: chess.Board, ctx: Context,
                  color: chess.Color) -> tuple[float, dict]:
    squares = activity.passed_squares(board, ctx, color)
    value = sum(max(0.0, activity.felt(
        board, ctx, passed_pawn_score(board, ctx, color, square)
    )) for square in squares)
    return value, {
        "pieces": _pieces(board, color, chess.PAWN, squares),
        "squares": _square_names(squares),
    }


_POSITION_DETECTORS: tuple[tuple[str, str, Callable], ...] = (
    ("pawn_structure", "pawn structure", _pawn_structure),
    ("knight_outpost", "knight outpost", _knight_outpost),
    ("active_bishop_pair", "active bishop pair", _active_bishop_pair),
    ("active_bishop", "active lone bishop", _active_bishop),
    ("bad_bishop", "bad bishop", _bad_bishop),
    ("bad_knight", "bad knight", _bad_knight),
    ("bad_rook", "bad rook", _bad_rook),
    ("active_rook", "active rook", _active_rook),
    ("colour_complexion", "colour complexion", _colour_complexion),
    ("space_advantage", "space", _space),
    ("passed_pawns", "passed pawns", _passed_pawns),
)


def material_motifs(board: chess.Board) -> list[dict]:
    """Return a meaningful, like-for-like-cancelled material imbalance.

    A conventional value difference is always meaningful. Equal-valued
    residuals are retained only when their composition remains different after
    bishop and knight are treated as one interchangeable minor-piece class.
    Thus bishop for knight is omitted, while rook and pawn for two minors is
    retained. Saved inventories contain only each side's uncancelled surplus.
    """
    inventories = {
        color: {
            _PIECE_NAMES[piece_type]: len(board.pieces(piece_type, color))
            for piece_type in _MATERIAL_TYPES
        }
        for color in _COLORS
    }
    if inventories[chess.WHITE] == inventories[chess.BLACK]:
        return []
    residual = {chess.WHITE: {}, chess.BLACK: {}}
    for piece_type in _MATERIAL_TYPES:
        name = _PIECE_NAMES[piece_type]
        white = inventories[chess.WHITE][name]
        black = inventories[chess.BLACK][name]
        if white > black:
            residual[chess.WHITE][name] = white - black
        elif black > white:
            residual[chess.BLACK][name] = black - white
    approximate = sum(
        _MATERIAL_VALUES[piece_type]
        * (residual[chess.WHITE].get(_PIECE_NAMES[piece_type], 0)
           - residual[chess.BLACK].get(_PIECE_NAMES[piece_type], 0))
        for piece_type in _MATERIAL_TYPES
    )

    def composition(color: chess.Color) -> dict[str, int]:
        side = residual[color]
        result = {
            "queen": side.get("queen", 0),
            "rook": side.get("rook", 0),
            "minor": side.get("bishop", 0) + side.get("knight", 0),
            "pawn": side.get("pawn", 0),
        }
        return {name: count for name, count in result.items() if count}

    composition_differs = (
        composition(chess.WHITE) != composition(chess.BLACK)
    )
    if approximate == 0 and not composition_differs:
        return []
    return [_record("material_imbalance", "material", {
        "white": residual[chess.WHITE],
        "black": residual[chess.BLACK],
        "approximate_pawn_difference": approximate,
        "classification": "numerical" if approximate else "composition",
    })]


def position_motifs(board: chess.Board) -> list[dict]:
    """Return every threshold-strong positional motif, separately by side."""
    records = material_motifs(board)
    if board.is_check():
        return records
    ctx = get_context(board)
    thresholds = activity.position_thresholds()
    for motif, feature, detector in _POSITION_DETECTORS:
        threshold = thresholds[feature]
        for color in _COLORS:
            score, metadata = detector(board, ctx, color)
            if score < threshold:
                continue
            records.append(_record(motif, "position", {
                "color": _color_name(color),
                "score": score,
                "threshold": threshold,
                "threshold_feature": feature,
                **metadata,
            }))
    return records


def _tactic_metadata(board: chess.Board, line: list[chess.Move], cp: int,
                     gap: float) -> dict:
    return {
        "color": _color_name(board.turn),
        "after": [move.uci() for move in line],
        "best_move": line[0].uci(),
        "eval_mover_cp": cp,
        "expected_score_gap": gap,
        "nodes": 10_000,
    }


def tactical_motifs(engines, before: chess.Board,
                    setup: chess.Move) -> list[dict]:
    """Return every accepted tactic after the human-game ``setup`` move.

    The common 1k/10k clear-best filters are identical to the verdict data.
    The saved continuation starts with the tactic side's first move.
    """
    from datagen.sim import tactics

    if setup not in before.legal_moves:
        return []
    board = before.copy(stack=False)
    board.push(setup)
    if board.is_game_over(claim_draw=False):
        return []

    lines = engines.screen.search2(board.fen())
    if not lines:
        return []
    cp1k = lines[0][1]
    if cp1k < tactics.EVAL_MIN - 30:
        return []
    if len(lines) > 1 and (
        tactics.es(cp1k) - tactics.es(lines[1][1]) < tactics.GAP_MIN - 0.02
    ):
        return []

    clear = tactics.clear_best(engines, board.fen())
    if clear is None:
        return []
    best_uci, cp10k, gap = clear
    line = tactics.tactic_line(engines, board, best_uci)
    if not line:
        return []
    puzzle = tactics.build_puzzle(before, setup, line, cp10k)
    tags = tactics.cook_tags(puzzle)

    first = line[0]
    raw_fork = tactics.fork_targets(board, first, raw=True)
    tags = [tag for tag in tags if tag in tactics.COOK_TAGS
            and not (tag == "deflection" and raw_fork)]
    base = _tactic_metadata(board, line, cp10k, gap)
    records = []
    seen = set()

    targets = tactics.fork_targets(board, first)
    our_fork = False
    if targets and len(line) >= 3:
        tracked = set(targets)
        if line[1].from_square in tracked:
            tracked.discard(line[1].from_square)
            tracked.add(line[1].to_square)
        after = board.copy(stack=False)
        after.push(line[0])
        after.push(line[1])
        our_fork = (
            after.is_capture(line[2])
            and line[2].to_square in tracked
            and line[2].from_square == first.to_square
        )
    if our_fork:
        moved = board.piece_at(first.from_square)
        records.append(_record("fork", "tactical", {
            **base,
            "attacker": {
                "color": _color_name(board.turn),
                "piece": _PIECE_NAMES[moved.piece_type],
                "square": chess.square_name(first.to_square),
            },
            "target_squares": _square_names(targets),
        }))
        seen.add("fork")

    if tactics.discovered_check(board, line):
        records.append(_record("discovered_check", "tactical", base))
        seen.add("discovered_check")
    elif tactics.discovered_prose(board, line) is not None:
        records.append(_record("discovered_attack", "tactical", base))
        seen.add("discovered_attack")

    for tag in tags:
        if tag.startswith("mateIn"):
            name = f"mate_in_{tag.removeprefix('mateIn')}"
        else:
            name = _TACTIC_NAMES.get(tag)
        if name is None or name in seen:
            continue
        records.append(_record(name, "tactical", base))
        seen.add(name)
    return records


def motifs_for_game_ply(engines, before: chess.Board,
                        setup: chess.Move) -> list[dict]:
    """Return all motifs in the position reached by the legal ``setup`` move."""
    if setup not in before.legal_moves:
        return []
    board = before.copy(stack=False)
    board.push(setup)
    return position_motifs(board) + tactical_motifs(engines, before, setup)
