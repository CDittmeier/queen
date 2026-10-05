"""Historical compact-PV benchmark parsing (structured and legacy HCE formats).

This intentionally retains the reported benchmark's move-geometry parser;
it is not a structural-hallucination detector for the surrounding prose.
"""
import math
import re
import chess

_ATOM = re.compile(
    r"<PIECE_[MO][PNBRQK]><SQUARE_(\d+)>"
    r"(?:<PIECE_[MO][PNBRQK]>)?<SQUARE_(\d+)>"
    r"(?:<PIECE_[MO]([QRBN])>)?"
)

_PROMOTION = {"Q": chess.QUEEN, "R": chess.ROOK,
              "B": chess.BISHOP, "N": chess.KNIGHT}

def _pov_square(token, root_white):
    square = int(token) - 1
    return square if root_white else square ^ 56

def extract_line(raw, fen):
    """Extract and legally replay the final compact critical/best line."""
    source = None
    if "Critical line:" in raw:
        section = raw.rsplit("Critical line:", 1)[1].split("—", 1)[0]
        source = "critical_line"
    else:
        matches = re.findall(r"it runs\s+(.*?)\s*Therefore,", raw,
                             flags=re.IGNORECASE | re.DOTALL)
        section = matches[-1] if matches else ""
        source = "best_variation" if matches else "none"
    atoms = list(_ATOM.finditer(section))
    if not atoms:
        return {"uci": [], "valid": False, "source": source,
                "error": "no compact line found", "atoms": 0}

    board = chess.Board(fen)
    root_white = board.turn == chess.WHITE
    moves = []
    for ply, atom in enumerate(atoms):
        frm = _pov_square(atom.group(1), root_white)
        to = _pov_square(atom.group(2), root_white)
        promotion = _PROMOTION.get(atom.group(3))
        candidates = [promotion] if promotion else [None, chess.QUEEN]
        move = next((chess.Move(frm, to, promotion=p)
                     for p in candidates
                     if chess.Move(frm, to, promotion=p) in board.legal_moves), None)
        if move is None:
            return {"uci": moves, "valid": False, "source": source,
                    "error": f"illegal/unparseable atom at ply {ply}: {atom.group(0)}",
                    "atoms": len(atoms)}
        moves.append(move.uci())
        board.push(move)
    return {"uci": moves, "valid": True, "source": source,
            "error": None, "atoms": len(atoms)}

def win_rate(cp):
    cp = max(-1500, min(1500, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)

def _prefix_suffix(model_line, solution):
    if not model_line:
        return False, False, False
    exact = model_line == solution
    prefix = (len(model_line) <= len(solution)
              and model_line == solution[:len(model_line)])
    suffix = (len(model_line) <= len(solution)
              and model_line == solution[-len(model_line):])
    return prefix, suffix, exact
FIELD = re.compile(
    r"(?ms)^(?P<name>ANALYSIS|BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):"
    r"\s*(?P<body>.*?)(?=^(?:ANALYSIS|BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION):|\Z)"
)

def parse_critical_line(text: str, fen: str) -> dict:
    fields = {match.group("name"): match.group("body").strip()
              for match in FIELD.finditer(text)}
    atoms = list(_ATOM.finditer(fields.get("CRITICAL_LINE", "")))
    if not atoms:
        # HCE-full-1 predates the structured uppercase fields and ends with
        # either ``Critical line:`` or ``it runs ... Therefore`` instead.
        return extract_line(text, fen)

    board = chess.Board(fen)
    root_white = board.turn == chess.WHITE
    moves: list[str] = []
    for ply, atom in enumerate(atoms):
        source = _pov_square(atom.group(1), root_white)
        target = _pov_square(atom.group(2), root_white)
        promotion = _PROMOTION.get(atom.group(3))
        promotions = [promotion] if promotion else [None, chess.QUEEN]
        move = next((candidate for candidate in (
            chess.Move(source, target, promotion=value) for value in promotions
        ) if candidate in board.legal_moves), None)
        if move is None:
            return {"uci": moves, "valid": False, "source": "CRITICAL_LINE",
                    "error": f"illegal/unparseable atom at ply {ply}: {atom.group(0)}",
                    "atoms": len(atoms)}
        moves.append(move.uci())
        board.push(move)
    return {"uci": moves, "valid": True, "source": "CRITICAL_LINE",
            "error": None, "atoms": len(atoms)}
