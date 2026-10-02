"""Translate between human chess prose and POV-token model prose.

``Translator`` builds paired human/machine sentences in lockstep and decodes
the stage-1--5 POV vocabulary for inspection or parsing. Its ``pov`` is the
side to move at the position that anchors the text.
"""
import re

import chess

from utils.utils import POV_SQUARE_TOKENS, _PIECE_TO_POV_TOKEN


PIECE_NAME = {"P": "pawn", "N": "knight", "B": "bishop",
              "R": "rook", "Q": "queen", "K": "king"}
ORDINAL = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th",
           "5": "5th", "6": "6th", "7": "7th", "8": "8th"}

_CAP = re.compile(
    r"<PIECE_([MO])([PNBRQK])><SQUARE_(\d+)><PIECE_([MO])([PNBRQK])><SQUARE_(\d+)>")
_TRIPLE = re.compile(r"<PIECE_([MO])([PNBRQK])><SQUARE_(\d+)><SQUARE_(\d+)>")
_LINE = re.compile(r"<SQUARE_(\d+)><SQUARE_(\d+)>\s*[-–—]?\s*(file|rank|diagonal)")
_SQ = re.compile(r"<SQUARE_(\d+)>")
_PIECE = re.compile(r"<PIECE_([MO])([PNBRQK])>")
_SEAM = re.compile(r"(?<=>)(?=<)")


class Translator:
    """Build and decode prose anchored at one side's point of view."""

    def __init__(self, pov: chess.Color):
        self.pov = pov
        self.h: list[str] = []
        self.m: list[str] = []

    # -------------------------------------------------------------- decoding

    def _square_name(self, idx: str) -> str:
        square = int(idx) - 1
        return chess.square_name(
            square if self.pov == chess.WHITE else square ^ 56)

    def _colour_name(self, side: str) -> str:
        colour = self.pov if side == "M" else not self.pov
        return "white" if colour == chess.WHITE else "black"

    def decode(self, text: str) -> str:
        """Decode POV tokens to absolute human prose at this anchor."""
        def capture(match):
            return (f"{self._colour_name(match.group(1))} "
                    f"{PIECE_NAME[match.group(2)]} "
                    f"{self._square_name(match.group(3))} takes "
                    f"{self._colour_name(match.group(4))} "
                    f"{PIECE_NAME[match.group(5)]} "
                    f"{self._square_name(match.group(6))}")

        def move(match):
            return (f"{self._colour_name(match.group(1))} "
                    f"{PIECE_NAME[match.group(2)]} "
                    f"{self._square_name(match.group(3))}-"
                    f"{self._square_name(match.group(4))}")

        def line(match):
            a = self._square_name(match.group(1))
            b = self._square_name(match.group(2))
            kind = match.group(3)
            if kind == "file" and a[0] == b[0]:
                return f"{a[0]}-file"
            if kind == "rank" and a[1] == b[1]:
                return f"{ORDINAL[a[1]]} rank"
            return f"{a}-{b} {kind}"

        text = _CAP.sub(capture, text)
        text = _TRIPLE.sub(move, text)
        text = _LINE.sub(line, text)
        text = _SEAM.sub(" ", text)
        text = _SQ.sub(lambda m: self._square_name(m.group(1)), text)
        text = _PIECE.sub(
            lambda m: (f"{self._colour_name(m.group(1))} "
                       f"{PIECE_NAME[m.group(2)]}"), text)
        return text.replace("<EMPTY>", "empty")

    def decode_absolute(self, raw: str) -> str:
        """Decode POV tokens and replace player placeholders with colours."""
        mover, other = (("White", "Black") if self.pov == chess.WHITE
                        else ("Black", "White"))
        return (self.decode(raw).replace("<PLAYER>", mover)
                .replace("<OPPONENT>", other))

    @staticmethod
    def reanchor(text: str, from_pov: chess.Color,
                 to_pov: chess.Color) -> str:
        """Re-anchor POV-token text from one side-to-move anchor to another."""
        if from_pov == to_pov:
            return text
        text = _SQ.sub(
            lambda m: f"<SQUARE_{((int(m.group(1)) - 1) ^ 56) + 1}>", text)
        text = _PIECE.sub(
            lambda m: f"<PIECE_{'O' if m.group(1) == 'M' else 'M'}{m.group(2)}>",
            text)
        return (text.replace("<PLAYER>", "\x00")
                .replace("<OPPONENT>", "<PLAYER>")
                .replace("\x00", "<OPPONENT>"))

    # --------------------------------------------------------------- building

    def txt(self, text: str) -> "Translator":
        self.h.append(text)
        self.m.append(text)
        return self

    def sqtok(self, square: int) -> str:
        return POV_SQUARE_TOKENS[
            square ^ 56 if self.pov == chess.BLACK else square]

    def sq(self, square: int) -> "Translator":
        self.h.append(chess.square_name(square))
        self.m.append(self.sqtok(square))
        return self

    def side(self, colour: chess.Color) -> "Translator":
        self.h.append("White" if colour else "Black")
        self.m.append("<PLAYER>" if colour == self.pov else "<OPPONENT>")
        return self

    def ptok(self, colour: chess.Color, piece_type: int) -> str:
        return _PIECE_TO_POV_TOKEN[(colour == self.pov, piece_type)]

    def piece(self, board: chess.Board, square: int) -> "Translator":
        piece = board.piece_at(square)
        self.h.append(f"{'white' if piece.color else 'black'} "
                      f"{chess.piece_name(piece.piece_type)} on "
                      f"{chess.square_name(square)}")
        self.m.append(
            self.ptok(piece.color, piece.piece_type) + self.sqtok(square))
        return self

    def move(self, board_before: chess.Board,
             move: chess.Move) -> "Translator":
        self.h.append(board_before.san(move))
        piece = board_before.piece_at(move.from_square)
        tokens = (self.ptok(piece.color, piece.piece_type)
                  + self.sqtok(move.from_square))
        if board_before.is_en_passant(move):
            tokens += self.ptok(not piece.color, chess.PAWN)
        elif board_before.is_capture(move):
            victim = board_before.piece_at(move.to_square)
            tokens += self.ptok(victim.color, victim.piece_type)
        tokens += self.sqtok(move.to_square)
        if move.promotion:
            tokens += self.ptok(piece.color, move.promotion)
        self.m.append(tokens)
        return self

    def node(self, node) -> "Translator":
        return self.move(node.parent.board(), node.move)

    @property
    def human(self) -> str:
        return "".join(self.h)

    @property
    def machine(self) -> str:
        return "".join(self.m)
