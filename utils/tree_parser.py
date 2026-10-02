"""Parse a stage-5 move-tree narrative (token prose) back into a move tree.

Inverts the datagen serialization (datagen/tree/glue.py): each move mention,
introduced by a frame ("Consider", "The opponent meets this with", "Then", ...),
is a child of the *current* node; backtrack markers reset the current node to an
earlier position. Squares are POV-anchored to the ROOT side to move (constant for
the whole narrative), matching the generator.

Illegal / unparseable handling (best-effort, for inspecting model output):
  * A well-formed but illegal move — its two squares read fine, but the move is
    not legal on the current board — is kept as a node labeled ``[illegal: <uci>]``.
    Its resulting position is unknown, so every descendant in that branch is
    likewise ``[illegal: ...]``.
  * A wholly unparseable move — its squares can't be read — becomes a ``???``
    node. Its resulting position is unknown, so rather than guess, we skip forward
    until an explicit position anchor ("...after {line}") snaps us back to a
    precise node in the tree, and resume there.
"""
import re

import chess

_SQ = re.compile(r"<SQUARE_(\d+)>")
_PROMO_PROSE = re.compile(r"promoting to <PIECE_[MO]([QRBN])>")
_PROMO = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
_NUM = re.compile(r"^\d+\.+$")   # move numbers in a compact path: "1." / "1..."

# A move descriptor never contains '.', ',' or '?' (those punctuate the eval that
# follows), so capture the run up to the first such mark. It always opens with a
# piece token, which keeps a sentence-initial "Now"/"Then" in prose from matching.
_FRAME = re.compile(
    r"(?:Let's consider|What about|Let's look at|We continue with|"
    r"The opponent meets this with|The opponent replies|In reply,|"
    r"Consider|Then|Now)\s+(<PIECE_[MO][PNBRQK]>[^.,?]*)")
_BT_PATH = re.compile(
    r"(?:Let me return to the position after|Backing up to the position after|"
    r"Let me see if the opponent has another response after|"
    r"Does the opponent have another try after)\s+(.*?)[.?](?=\s+[A-Z]|\s*\Z)")
_BT_THERE = re.compile(r"Another try from there|Also worth a look there")
_BT_START = re.compile(r"Let me go back to the starting position|Returning to the initial position")
_BEST = re.compile(r"I should play\s+\*\*([^*]+?)\*\*")


def _pov_sq(n: int, root_wtm: bool) -> int:
    """POV square token (1..64) -> python-chess square (a1=0), anchored to the root."""
    return (n - 1) if root_wtm else ((n - 1) ^ 56)


class Node:
    """A parsed move. ``board`` is None when the move is illegal/unknown (so its
    descendants can't be verified). ``label`` is SAN / ``[illegal: <uci>]`` / ``???``."""
    __slots__ = ("board", "move", "label", "illegal", "parent", "children")

    def __init__(self, board, move, label, illegal, parent):
        self.board, self.move, self.label = board, move, label
        self.illegal, self.parent, self.children = illegal, parent, []

    def add(self, board, move, label, illegal):
        child = Node(board, move, label, illegal, self)
        self.children.append(child)
        return child


def _read_move(clause: str, root_wtm: bool):
    """(from_sq, to_sq, promo) from a prose move clause, or None if < 2 squares."""
    sqs = _SQ.findall(clause)
    if len(sqs) < 2:
        return None
    pm = _PROMO_PROSE.search(clause)
    return (_pov_sq(int(sqs[0]), root_wtm), _pov_sq(int(sqs[-1]), root_wtm),
            _PROMO.get(pm.group(1)) if pm else None)


def _legal(frm, to, promo, board):
    """The legal chess.Move for (frm, to, promo) on ``board``, or None.
    Unspecified promotions fall back to a queen."""
    for pr in ([promo] if promo else [None, chess.QUEEN]):
        mv = chess.Move(frm, to, promotion=pr)
        if mv in board.legal_moves:
            return mv
    return None


def parse_tree(narrative: str, root_fen: str, trace: list | None = None):
    """Parse ``narrative`` into a move tree rooted at ``root_fen``.

    Returns ``(root, stats, best_move)`` where ``root`` is a Node, ``stats`` counts
    legal/illegal/unparseable moves and unresolved backtracks, and ``best_move`` is
    the recommended move parsed from the "I should play **...**" conclusion (or None).

    When ``trace`` is a list, ``(narrative_offset, node)`` is appended for every node
    created from a move event, in creation order.
    """
    root = Node(chess.Board(root_fen), None, None, False, None)
    root_wtm = root.board.turn == chess.WHITE
    current = last_ret = root
    recovering = False
    stats = {"legal": 0, "illegal": 0, "unparseable": 0, "unresolved_bt": 0}

    events = []
    for m in _BT_PATH.finditer(narrative):
        events.append((m.start(), "bt_path", m.group(1)))
    for m in _BT_THERE.finditer(narrative):
        events.append((m.start(), "bt_there", None))
    for m in _BT_START.finditer(narrative):
        events.append((m.start(), "bt_start", None))
    for m in _FRAME.finditer(narrative):
        events.append((m.start(), "move", m.group(1)))
    events.sort(key=lambda e: e[0])

    def resolve(path: str):
        """Navigate the built tree by a compact move path ("1... <tag> <tag>")."""
        node = root
        for tag in (t for t in path.split() if not _NUM.match(t)):
            mv = _read_move(tag, root_wtm)
            if mv is None:
                return None
            node = next((c for c in node.children if c.move is not None
                         and c.move.from_square == mv[0] and c.move.to_square == mv[1]), None)
            if node is None:
                return None
        return node

    def _made(off, node):
        if trace is not None:
            trace.append((off, node))
        return node

    for off, kind, payload in events:
        if kind == "bt_start":
            current = last_ret = root
            recovering = False
            continue
        if kind == "bt_there":
            current = last_ret
            recovering = False
            continue
        if kind == "bt_path":
            tgt = resolve(payload)
            if tgt is None or tgt.board is None:
                stats["unresolved_bt"] += 1   # leave `recovering` set: wait for a resolvable anchor
            else:
                current = last_ret = tgt
                recovering = False            # snapped to a precise position
            continue

        if recovering:
            # After a '???' the resulting position is unknown. Rather than guess by
            # attaching the next legal-looking move, we skip forward until an explicit
            # position anchor ("...after {line}", handled above) snaps us to a precise
            # node — then normal parsing resumes there.
            continue

        mv = _read_move(payload, root_wtm)

        if current.board is None:
            # Inside an illegal branch — nothing can be verified; inherit illegal.
            if mv is None:
                current = _made(off, current.add(None, None, "???", True))
                stats["unparseable"] += 1
            else:
                m = chess.Move(mv[0], mv[1], promotion=mv[2])
                current = _made(off, current.add(None, m, f"[illegal: {m.uci()}]", True))
                stats["illegal"] += 1
            continue

        if mv is None:
            # Unparseable at a valid node: mark the gap, then fast-forward.
            _made(off, current.add(None, None, "???", True))
            stats["unparseable"] += 1
            recovering = True
            continue

        legal = _legal(*mv, current.board)
        if legal is not None:
            b2 = current.board.copy(); label = b2.san(legal); b2.push(legal)
            current = _made(off, current.add(b2, legal, label, False))
            stats["legal"] += 1
        else:
            m = chess.Move(mv[0], mv[1], promotion=mv[2])
            current = _made(off, current.add(None, m, f"[illegal: {m.uci()}]", True))
            stats["illegal"] += 1

    best = None
    bm = _BEST.findall(narrative)
    if bm:
        r = _read_move(bm[-1], root_wtm)
        if r is not None:
            best = _legal(*r, root.board) or chess.Move(r[0], r[1], promotion=r[2])
    return root, stats, best


def render(root: Node) -> str:
    """ASCII rendering of the parsed tree (SAN / [illegal: uci] / ??? per node)."""
    lines = []

    def walk(node, prefix):
        for i, c in enumerate(node.children):
            last = i == len(node.children) - 1
            lines.append(prefix + ("└─ " if last else "├─ ") + c.label)
            walk(c, prefix + ("   " if last else "│  "))

    walk(root, "")
    return "\n".join(lines)
