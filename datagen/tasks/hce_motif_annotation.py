"""Rendering, parsing, and grading for the stage-5 motif annotation task."""
from __future__ import annotations

import re

import chess

from datagen.position_features import sequence_move_dicts
from datagen.prose import (
    MATERIAL_MOTIF_TEMPLATES,
    MOTIF_TEMPLATES,
    POSITION_MOTIF_TEMPLATES,
    TACTICAL_MOTIF_TEMPLATES,
    format_move_sequence_tag,
    join_and,
    render_motif_prose,
)
from utils.board_representation import BoardRepr
from utils.translate_helpers import Translator


NAME = "hce_motif_annotation"
MAX_UNIQUE_QUERIES = 1
MINED_RECORDS = True

MATERIAL_MOTIFS = frozenset(MATERIAL_MOTIF_TEMPLATES)
POSITION_MOTIF_ORDER = tuple(POSITION_MOTIF_TEMPLATES)
POSITION_MOTIFS = frozenset(POSITION_MOTIF_TEMPLATES)
TACTICAL_MOTIFS = frozenset(TACTICAL_MOTIF_TEMPLATES)
MOTIF_ORDER = (
    *MATERIAL_MOTIF_TEMPLATES,
    *POSITION_MOTIF_TEMPLATES,
    *TACTICAL_MOTIF_TEMPLATES,
)
MOTIF_ORDER_INDEX = {motif: index for index, motif in enumerate(MOTIF_ORDER)}

CURRENT_PROMPT = "Identify the positional and tactical motifs in the current position."
SEQUENCE_PROMPT = (
    "Identify the positional and tactical motifs in the position after the sequence: {sequence}."
)

_ENTRY_RE = re.compile(r"^\s*-\s+([^:]+):\s*(.+?)\s*$")
_SIDE_RE = re.compile(r"<(PLAYER|OPPONENT)>")


def number_agreement(count: int) -> dict[str, str]:
    """Template fields for a motif whose named subject has ``count`` pieces."""
    if count < 1:
        raise ValueError("motif number agreement requires at least one piece")
    singular = count == 1
    return {
        "be": "is" if singular else "are",
        "have": "has" if singular else "have",
        "occupy": "occupies" if singular else "occupy",
        "enjoy": "enjoys" if singular else "enjoy",
        "lack": "lacks" if singular else "lack",
        "give": "gives" if singular else "give",
        "active_piece_noun": "an active piece" if singular else "active pieces",
        "passed_pawn_noun": "a passed pawn" if singular else "passed pawns",
        "possessive": "its" if singular else "their",
    }


_PIECE_TYPES = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}


def _color(name: str) -> chess.Color:
    return chess.WHITE if name == "white" else chess.BLACK


def _side(name: str, anchor: chess.Color) -> str:
    return "<PLAYER>" if _color(name) == anchor else "<OPPONENT>"


def _piece(board: chess.Board, square: int, anchor: chess.Color) -> str:
    return Translator(anchor).piece(board, square).machine


def _move(board: chess.Board, move: chess.Move, anchor: chess.Color) -> str:
    return Translator(anchor).move(board, move).machine


def _squares(names: list[str], anchor: chess.Color) -> str:
    translator = Translator(anchor)
    return join_and([translator.sqtok(chess.parse_square(name)) for name in names])


def _species(pieces: list[dict], anchor: chess.Color) -> str:
    translator = Translator(anchor)
    return join_and([
        translator.ptok(_color(piece["color"]), _PIECE_TYPES[piece["piece"]])
        for piece in pieces
    ])


def _line_boards(board: chess.Board,
                 moves: list[chess.Move]) -> list[chess.Board]:
    boards = [board.copy(stack=False)]
    for move in moves:
        child = boards[-1].copy(stack=False)
        if move not in child.legal_moves:
            raise ValueError(f"illegal mined tactic move {move.uci()} in {child.fen()}")
        child.push(move)
        boards.append(child)
    return boards


def _line_text(board: chess.Board, moves: list[chess.Move],
               anchor: chess.Color) -> str:
    move_dicts = sequence_move_dicts(
        BoardRepr(board.fen(), pov=True, pov_anchor=anchor), moves
    )
    return format_move_sequence_tag(move_dicts)


def _position_values(motif: str, metadata: dict,
                     anchor: chess.Color) -> dict:
    if motif == "material_imbalance":
        translator = Translator(anchor)

        def inventory(color_name: str) -> str:
            color = _color(color_name)
            parts = []
            for name in ("queen", "rook", "bishop", "knight", "pawn"):
                count = metadata[color_name].get(name, 0)
                if not count:
                    continue
                token = translator.ptok(color, _PIECE_TYPES[name])
                parts.append(f"{count} {token}{'' if count == 1 else 's'}")
            return join_and(parts) or "no unmatched material"

        difference = metadata["approximate_pawn_difference"]
        if difference:
            relation = (
                f"an approximate {abs(difference)}-pawn advantage remains for "
                f"{'White' if difference > 0 else 'Black'}"
            )
        else:
            relation = "the residuals are equal in conventional pawn value"
        return {
            "white_clause": f"White has {inventory('white')}",
            "black_clause": f"Black has {inventory('black')}",
            "material_relation": relation,
        }

    side = _side(metadata["color"], anchor)
    pieces = metadata.get("pieces", [])
    squares = metadata.get("squares", [])
    values = {
        "side": side,
        "knights": _species(pieces, anchor),
        "bishops": _species(pieces, anchor),
        "rooks": _species(pieces, anchor),
        "squares": _squares(squares, anchor),
        **number_agreement(max(1, len(pieces) or len(squares))),
    }
    if motif == "pawn_structure":
        translator = Translator(anchor)
        pawn = translator.ptok(_color(metadata["color"]), chess.PAWN)
        features = []
        for factor in metadata["factors"]:
            count = len(factor["squares"])
            article = "an" if factor["name"] == "isolated" else "a"
            noun = (f"{article} {factor['name']} {pawn}" if count == 1
                    else f"{factor['name']} {pawn}s")
            features.append(
                f"{noun} on {_squares(factor['squares'], anchor)}"
            )
        values.update({
            "structure_effect": "weakened",
            "structure_result": "weaker",
            "structure_verb": "suffers",
            "features": join_and(features),
        })
    elif motif == "colour_complexion":
        values["complex"] = metadata["complex"]
    elif motif == "space_advantage":
        regions = metadata["regions"]
        maximum = max(regions.values())
        region = join_and([name for name, value in regions.items()
                           if value == maximum])
        values["region"] = f"in the {region}"
    return values


def _tactical_values(motif: str, metadata: dict, board: chess.Board,
                     anchor: chess.Color) -> dict:
    """Recover exact template fields from the accepted tactic line."""
    from datagen.tree.primitives import lichess_util as util

    moves = [chess.Move.from_uci(uci) for uci in metadata["after"]]
    boards = _line_boards(board, moves)
    values = {"moves": _line_text(board, moves, anchor)}

    def move_at(index: int) -> str:
        return _move(boards[index], moves[index], anchor)

    if motif.startswith("mate_in_"):
        return {
            **values,
            "side": _side(metadata["color"], anchor),
            "mating_line": values["moves"],
        }
    if motif == "hanging_piece":
        move = moves[0]
        square = (chess.square(chess.square_file(move.to_square),
                               chess.square_rank(move.from_square))
                  if board.is_en_passant(move) else move.to_square)
        return {**values, "piece": _piece(board, square, anchor),
                "move": move_at(0)}
    if motif == "fork":
        first = moves[0]
        after = boards[1]
        targets = [
            _piece(after, chess.parse_square(name), anchor)
            for name in metadata["target_squares"]
            if after.piece_at(chess.parse_square(name)) is not None
        ]
        return {
            **values,
            "move": move_at(0),
            "attacker": _piece(after, first.to_square, anchor),
            "targets": join_and(targets),
            "square": Translator(anchor).sqtok(first.to_square),
        }
    if motif == "back_rank_mate":
        king_square = boards[-1].king(boards[-1].turn)
        if king_square is None:
            raise ValueError("back-rank mate has no mated king")
        return {**values, "mating_move": move_at(len(moves) - 1),
                "king": _piece(boards[-1], king_square, anchor)}
    if motif == "attraction":
        for index in range(0, len(moves) - 2, 2):
            first, reply, follow = moves[index:index + 3]
            if reply.to_square != first.to_square:
                continue
            attracted = boards[index + 1].piece_at(reply.from_square)
            if attracted is None or attracted.piece_type not in (
                    chess.KING, chess.QUEEN, chess.ROOK):
                continue
            attackers = boards[index + 3].attackers(
                board.turn, reply.to_square
            )
            if follow.to_square in attackers:
                return {
                    **values,
                    "move": move_at(index),
                    "piece": _piece(boards[index + 1], reply.from_square, anchor),
                    "square": Translator(anchor).sqtok(reply.to_square),
                    "follow_up": move_at(index + 2),
                }
    if motif == "deflection":
        for index in range(2, len(moves), 2):
            winning = moves[index]
            captured = boards[index].piece_at(winning.to_square)
            if captured is None and not winning.promotion:
                continue
            defender_move = moves[index - 1]
            lure = moves[index - 2]
            square = winning.to_square
            target = boards[index - 1].piece_at(square)
            defended_before = square in boards[index - 1].attacks(
                defender_move.from_square
            )
            # Lichess also calls a promotion a deflection when the defender
            # controlled the promotion file from behind: after it is lured
            # away, the pawn can advance/promote along that file.
            promotion_file_defense = (
                winning.promotion is not None
                and chess.square_file(winning.to_square)
                == chess.square_file(defender_move.from_square)
                and winning.from_square in boards[index - 1].attacks(
                    defender_move.from_square
                )
            )
            if (square in (defender_move.to_square, lure.to_square)
                    or not (defender_move.to_square == lure.to_square
                            or boards[index - 1].is_check())
                    or not (defended_before or promotion_file_defense)
                    or square in boards[index].attacks(defender_move.to_square)):
                continue
            return {
                **values,
                "piece": _piece(boards[index], defender_move.to_square, anchor),
                "target": (_piece(boards[index - 1], square, anchor)
                           if target is not None
                           else Translator(anchor).sqtok(square)),
                "winning_move": move_at(index),
                "move": move_at(index - 2),
            }
    if motif == "trapped_piece":
        for index in range(2, len(moves), 2):
            square = moves[index].to_square
            captured = boards[index].piece_at(square)
            if captured is None or captured.piece_type == chess.PAWN:
                continue
            previous = moves[index - 1]
            trapped_square = (previous.from_square
                              if previous.to_square == square else square)
            if util.is_trapped(boards[index - 1], trapped_square):
                return {
                    **values,
                    "piece": _piece(boards[index - 1], trapped_square, anchor),
                    "move": move_at(index),
                }
    if motif == "skewer":
        for index in range(2, len(moves), 2):
            winning = moves[index]
            current = boards[index]
            rear = current.piece_at(winning.to_square)
            front_move = moves[index - 1]
            if (rear is None
                    or current.piece_at(winning.from_square).piece_type
                    not in util.ray_piece_types
                    or front_move.to_square == winning.to_square
                    or front_move.from_square not in chess.SquareSet.between(
                        winning.from_square, winning.to_square)):
                continue
            return {
                **values,
                "attacker": _piece(current, winning.from_square, anchor),
                "front_target": _piece(current, front_move.to_square, anchor),
                "rear_target": _piece(current, winning.to_square, anchor),
                "winning_move": move_at(index),
            }
    if motif == "interference":
        for index in range(2, len(moves), 2):
            target_square = moves[index].to_square
            target = boards[index].piece_at(target_square)
            if target is None or not util.is_hanging(
                    boards[index], target, target_square):
                continue
            for blocking_index in (index - 1, index - 2):
                blocking = moves[blocking_index]
                initial = boards[blocking_index]
                for defender_square in initial.attackers(
                        target.color, target_square):
                    defender = initial.piece_at(defender_square)
                    if (defender is None
                            or defender.piece_type not in util.ray_piece_types
                            or blocking.to_square not in chess.SquareSet.between(
                                target_square, defender_square)):
                        continue
                    return {
                        **values,
                        "blocking_move": move_at(blocking_index),
                        "defender": _piece(initial, defender_square, anchor),
                        "target": _piece(initial, target_square, anchor),
                        "winning_move": move_at(index),
                    }
    if motif == "intermezzo":
        for index in range(2, len(moves), 2):
            recapture = moves[index]
            if not boards[index].is_capture(recapture):
                continue
            square = recapture.to_square
            intermediate_index = index - 2
            if moves[intermediate_index].to_square != square:
                return {
                    **values,
                    "square": Translator(anchor).sqtok(square),
                    "intermediate_move": move_at(intermediate_index),
                    "recapture": move_at(index),
                    "side": _side(metadata["color"], anchor),
                }
    if motif == "pin":
        for index in range(0, len(moves), 2):
            after = boards[index + 1]
            for square, piece in after.piece_map().items():
                if piece.color == board.turn:
                    continue
                pin_ray = after.pin(piece.color, square)
                if pin_ray == chess.BB_ALL:
                    continue
                pinners = [sq for sq in after.attackers(board.turn, square)
                           if sq in pin_ray]
                if not pinners:
                    continue
                king_square = after.king(piece.color)
                if king_square is None:
                    continue
                return {
                    **values,
                    "pinner": _piece(after, pinners[0], anchor),
                    "pinned_piece": _piece(after, square, anchor),
                    "king": _piece(after, king_square, anchor),
                    "move": move_at(index),
                }
    if motif == "x_ray_attack":
        for index in range(2, len(moves), 2):
            winning = moves[index]
            screen = moves[index - 1]
            if (not boards[index].is_capture(winning)
                    or screen.to_square != winning.to_square
                    or screen.from_square not in chess.SquareSet.between(
                        winning.from_square, winning.to_square)):
                continue
            return {
                **values,
                "attacker": _piece(boards[index], winning.from_square, anchor),
                "screening_piece": _piece(
                    boards[index - 1], screen.from_square, anchor
                ),
                "screening_move": move_at(index - 1),
                "move": move_at(index),
            }
    if motif == "collinear_move":
        for index in range(0, len(moves), 2):
            move = moves[index]
            before = boards[index]
            moving = before.piece_at(move.from_square)
            if (moving is None or moving.piece_type not in util.ray_piece_types
                    or before.is_capture(move)):
                continue
            for square in before.attacks(move.from_square):
                enemy = before.piece_at(square)
                if (enemy is None or enemy.color == moving.color
                        or enemy.piece_type not in util.ray_piece_types
                        or not util.squares_are_collinear(
                            move.from_square, square, move.to_square)
                        or chess.Move(move.from_square, square)
                        not in before.legal_moves):
                    continue
                same_file = chess.square_file(move.from_square) == chess.square_file(square)
                same_rank = chess.square_rank(move.from_square) == chess.square_rank(square)
                return {
                    **values,
                    "move": move_at(index),
                    "piece": _piece(before, move.from_square, anchor),
                    "enemy_piece": _piece(before, square, anchor),
                    "line": "file" if same_file else "rank" if same_rank else "diagonal",
                }
    if motif in ("discovered_check", "discovered_attack"):
        for index in range(0, len(moves), 2):
            after = boards[index + 1]
            checkers = [sq for sq in after.checkers()
                        if sq != moves[index].to_square]
            if motif == "discovered_check" and checkers:
                king_square = after.king(after.turn)
                return {
                    **values,
                    "uncovering_move": move_at(index),
                    "piece": _piece(boards[index], moves[index].from_square, anchor),
                    "revealed_checker": _piece(after, checkers[0], anchor),
                    "king": _piece(after, king_square, anchor),
                }
            if (motif == "discovered_attack" and index >= 2
                    and boards[index].is_capture(moves[index])):
                prior, reply = moves[index - 2], moves[index - 1]
                between = chess.SquareSet.between(
                    moves[index].from_square, moves[index].to_square
                )
                if (prior.from_square in between
                        and moves[index].to_square != prior.to_square
                        and moves[index].from_square != prior.to_square
                        and reply.to_square != moves[index].to_square):
                    return {
                        **values,
                        "uncovering_move": move_at(index - 2),
                        "piece": _piece(
                            boards[index - 2], prior.from_square, anchor
                        ),
                        "revealed_attacker": _piece(
                            boards[index], moves[index].from_square, anchor
                        ),
                        "target": _piece(
                            boards[index], moves[index].to_square, anchor
                        ),
                        "winning_move": move_at(index),
                    }
    raise ValueError(f"could not reconstruct {motif} from {board.fen()}")


def payloads_from_record(record: dict, prompt_fen: str) -> list[dict]:
    """Convert one mined chess-native record to render-ready motif payloads."""
    board = chess.Board(record["fen"])
    anchor = chess.Board(prompt_fen).turn
    payloads = []
    for item in record["motifs"]:
        motif, metadata = item["motif"], item["metadata"]
        values = (_tactical_values(motif, metadata, board, anchor)
                  if metadata["source"] == "tactical"
                  else _position_values(motif, metadata, anchor))
        payloads.append({"motif": motif, "values": values})
    return payloads


def render_mined_record(record: dict, prompt_fen: str,
                        prompt_moves: list[str], rng) -> dict:
    """Render a mined motif record at a chosen 0--8-ply prompt root."""
    board = BoardRepr(prompt_fen, pov=True)
    moves = [chess.Move.from_uci(uci) for uci in prompt_moves]
    return compose_motif_annotation(
        sequence_move_dicts(board, moves),
        payloads_from_record(record, prompt_fen),
        rng,
    )


def _lead_in(template: str) -> str:
    """The controlled natural-English alias before ``after`` / the colon."""
    head = template.split(":", 1)[0]
    return head.split(" after ", 1)[0].strip().casefold()


MOTIF_ALIASES = {
    _lead_in(template): motif
    for motif, templates in MOTIF_TEMPLATES.items()
    for template in templates
}
if len(MOTIF_ALIASES) != sum(len(v) for v in MOTIF_TEMPLATES.values()):
    raise RuntimeError("motif prose lead-ins must be unique")


def parse_motif_response(text: str) -> list[dict]:
    """Parse a bullet response into canonical motif dictionaries.

    Position records retain their explicit ``PLAYER`` / ``OPPONENT`` side so a
    response cannot receive credit for identifying the right motif for the
    wrong side. Tactical records use ``None``. Any prose outside the controlled
    list grammar is rejected rather than partially credited.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty motif response")
    parsed = []
    for line in lines:
        match = _ENTRY_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed motif entry: {line!r}")
        head, body = match.groups()

        if " after " in head:
            alias, moves = head.split(" after ", 1)
            if not moves.strip():
                raise ValueError(f"tactical motif has an empty move line: {line!r}")
        else:
            alias = head
        motif = MOTIF_ALIASES.get(alias.strip().casefold())
        if motif is None:
            raise ValueError(f"unknown motif lead-in: {alias!r}")

        metadata = {}
        if motif in POSITION_MOTIFS:
            side_match = _SIDE_RE.search(body)
            if side_match is None:
                raise ValueError(f"position motif does not name its side: {line!r}")
            metadata["side"] = side_match.group(1).lower()
        parsed.append({"motif": motif, "metadata": metadata})

    def order_key(item: dict) -> tuple[int, int]:
        side = item["metadata"].get("side")
        return (MOTIF_ORDER_INDEX[item["motif"]],
                1 if side == "opponent" else 0)

    keys = [order_key(item) for item in parsed]
    if keys != sorted(keys):
        raise ValueError("motif bullets are not in canonical order")
    identities = [
        (item["motif"], item["metadata"].get("side")) for item in parsed
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate motif bullet")
    return parsed


def extract_motif_dicts(text: str) -> list[dict]:
    """Public name for strict response-to-structured-motif extraction."""
    return parse_motif_response(text)


def grade_motif_response(prediction: str, gold: str) -> bool:
    """Exact motif-and-side accuracy; malformed generations are incorrect."""
    try:
        return parse_motif_response(prediction) == parse_motif_response(gold)
    except ValueError:
        return False


def _ordered_payloads(payloads: list[dict]) -> list[dict]:
    """One global motif order; sided motifs put PLAYER before OPPONENT."""
    indexed = list(enumerate(payloads))

    def key(pair):
        index, payload = pair
        motif = payload["motif"]
        if motif not in MOTIF_ORDER_INDEX:
            raise ValueError(f"unknown motif in payload: {motif!r}")
        side_rank = 0
        if motif in POSITION_MOTIFS:
            side = payload.get("values", {}).get("side")
            if side not in ("<PLAYER>", "<OPPONENT>"):
                raise ValueError(
                    f"position motif {motif!r} needs a PLAYER/OPPONENT side"
                )
            side_rank = 0 if side == "<PLAYER>" else 1
        return (MOTIF_ORDER_INDEX[motif], side_rank, index)

    return [payload for _index, payload in sorted(indexed, key=key)]


def compose_motif_annotation(move_dicts: list[dict], payloads: list[dict], rng) -> dict:
    """Render already-mined motif payloads into one prompt/response record."""
    if not payloads:
        raise ValueError("motif annotation needs at least one motif")
    if move_dicts:
        question = SEQUENCE_PROMPT.format(
            sequence=format_move_sequence_tag(move_dicts)
        )
    else:
        question = CURRENT_PROMPT

    ordered = _ordered_payloads(payloads)
    entries = []
    for payload in ordered:
        motif = payload["motif"]
        values = payload["values"]
        prose = render_motif_prose(
            motif, values, rng, variant=payload.get("variant")
        )
        entries.append(f"- {prose}")

    return {
        "question": question,
        "answer": "\n".join(entries),
        "question_type": NAME,
        "answer_class": [
            payload["motif"] + (
                ":" + payload["values"]["side"]
                if payload["motif"] in POSITION_MOTIFS else ""
            )
            for payload in ordered
        ],
    }
