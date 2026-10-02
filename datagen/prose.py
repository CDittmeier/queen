"""String-formatting + line-task QA helpers shared across `datagen/tasks/*.py`.

The prose helpers (`join_and`, `plural`, `encode_piece_count`,
`format_piece_counts`, `format_square_breakdown`) are pure string ops and
have no knowledge of BoardRepr / chess.Board / FEN.

The line-task helper (`line_facts`) takes a tuple of board squares + a
BoardRepr and emits the shared (start_tok, end_tok, ordered, parse_tag,
answer_class) bundle used by piece_on_{file, rank, diagonal}. Its square
composition comes from `position_features.line_piece_counts`.

Sources:
- `format_piece_counts`, `format_square_breakdown` ported from the old
  `utils/build_qa_dataset.py` (now superseded by `datagen/`).
- `plural`, `join_and` ported from
  `depr/ab_chesslm/src/data/curriculum/tasks/_common.py`.
"""
from typing import Optional

import chess

from utils.board_representation import BoardRepr
from utils.utils import EMPTY_TOKEN
from datagen.position_features import line_piece_counts


# ---------------------------------------------------------------------------
# List joiner
# ---------------------------------------------------------------------------

def join_and(items: list[str]) -> str:
    """Oxford-comma join: '' / 'a' / 'a and b' / 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ---------------------------------------------------------------------------
# Token formatting
# ---------------------------------------------------------------------------

def plural(token: str, n: int) -> str:
    """'<PIECE_WB>' -> '<PIECE_WB>s' for n != 1. Cosmetic only — never tokenized."""
    return f"{token}s" if n != 1 else token


def _sq_num(sq_tok: str) -> int:
    """The 1..64 index inside a '<SQUARE_n>' token."""
    return int(sq_tok[len("<SQUARE_"):-1])


# King-first piece ordering for the attack/defense (piece, square) multiset tag.
_PIECE_TOK_ORDER = "KQRBNP"


def _piece_square_key(item: tuple[str, str]) -> tuple[int, int]:
    """Canonical (piece, square) order: heavier piece first (king > queen > rook >
    bishop > knight > pawn), then ascending square index (1..64). Shared by the
    parse-tag flattener and the CoT prose flattener so both order identically."""
    piece_tok, sq_tok = item
    return (_PIECE_TOK_ORDER.index(piece_tok[-2]), int(sq_tok[len("<SQUARE_"):-1]))


def format_piece_squares(items: list[tuple[str, str]]) -> str:
    """Format (piece_tok, square_tok) pairs into a spaceless `<piece><square>...`
    multiset tag, in canonical `_piece_square_key` order (heavier piece first,
    then ascending square). Assumes already POV-reflected square tokens.

    Callers compose a section's parse tag from two of these strings — call once
    with the attacker (piece, square) tokens and once with the defenders.
    """
    return "".join(f"{p}{s}" for p, s in sorted(items, key=_piece_square_key))


def format_piece_square_list(items: list[tuple[str, str]]) -> str:
    """CoT prose for (piece_tok, square_tok) pairs: '<piece> on <square>' entries,
    'and'-joined, in the same canonical `_piece_square_key` order as
    format_piece_squares (so the prose and the parse tag list pieces identically)."""
    ordered = sorted(items, key=_piece_square_key)
    return join_and([f"{p} on {s}" for p, s in ordered])


# ---------------------------------------------------------------------------
# Move formatting (token-resolved move dict from position_features.*_moves)
# ---------------------------------------------------------------------------

def _directional_verb(from_tok: str, to_tok: str) -> str:
    """advances/retreats/slides 'to', by POV rank (already-reflected, so a higher
    rank is toward the opponent). Assumes POV square tokens (<SQUARE_1..64>)."""
    fr = (int(from_tok[len("<SQUARE_"):-1]) - 1) // 8
    tr = (int(to_tok[len("<SQUARE_"):-1]) - 1) // 8
    if tr > fr:
        return "advances to"
    if tr < fr:
        return "retreats to"
    return "slides to"


def format_move_cot(move: dict, rng) -> str:
    """Chain-of-thought (prose) rendering of a move dict — a core plus ordered
    modifiers. Takes the move dict and an rng (for the phrasing samplers).

    Core (equal-probability verb samplers):
        non-capture : '<piece> on <from> {verb} <to>'  with verb from
                      'moves to' / 'to' / directional (advances/retreats/slides to)
        capture     : '<piece> on <from> {verb} <piece> on <sq>'  with verb from
                      'captures' / 'takes'; <sq> is the captured pawn's square for
                      en passant, else the destination.

    Modifiers, in importance order castling > promotion > en passant > check:
        castling    'castling <side>'   (side from queen-side/long or king-side/short)
        promotion   'promoting to <piece>'
        en passant  'en passant and lands on <to>'
        check       'with check' / 'with checkmate'
    A non-check modifier that follows another modifier is prefixed with 'and '.
    """
    piece, frm, to = move["piece"], move["from_sq"], move["to_sq"]
    captured, ep_sq = move["captured_piece"], move["en_passant_square"]

    if captured is not None:
        cap_sq = ep_sq if ep_sq is not None else to
        verb = rng.choice(["captures", "takes"])
        core = f"{piece} on {frm} {verb} {captured} on {cap_sq}"
    else:
        verb = rng.choice(["moves to", "to", _directional_verb(frm, to)])
        core = f"{piece} on {frm} {verb} {to}"

    mods: list[str] = []
    if move["castle_type"] is not None:
        side = rng.choice(["queen-side", "long"] if move["castle_type"] == "queenside"
                          else ["king-side", "short"])
        mods.append(f"castling {side}")
    if move["promotion_to"] is not None:
        mods.append(f"promoting to {move['promotion_to']}")
    if ep_sq is not None:
        mods.append(f"en passant and lands on {to}")

    parts = [core]
    for i, text in enumerate(mods):
        parts.append(text if i == 0 else f"and {text}")
    if move["check_status"] == "check":
        parts.append("with check")
    elif move["check_status"] == "checkmate":
        parts.append("with checkmate")
    return " ".join(parts)


def format_move_tag(move: dict) -> str:
    """Parse-tag rendering of a move dict. Uniform across move kinds:

        <piece><from><to>                              quiet / castle
        <piece><from><captured><to>                    capture (incl. en passant)
        <piece><from><to><new_piece>                   promotion
        <piece><from><captured><to><new_piece>         capture-promotion

    The captured piece goes before <to>; the promotion piece after. <to> is always
    the moving piece's landing square (so for en passant it is the destination, not
    the captured pawn's square)."""
    parts = [move["piece"], move["from_sq"]]
    if move["captured_piece"] is not None:
        parts.append(move["captured_piece"])
    parts.append(move["to_sq"])
    if move["promotion_to"] is not None:
        parts.append(move["promotion_to"])
    return "".join(parts)


def _move_class_rank(move: dict) -> int:
    """Coarsest move-ordering bucket — checks before captures before quiet moves:
    0 checkmate, 1 check-capture, 2 check, 3 capture, 4 quiet. (En passant counts
    as a capture, since its move dict carries a captured_piece.)"""
    capture = move["captured_piece"] is not None
    if move["check_status"] == "checkmate":
        return 0
    if move["check_status"] == "check":
        return 1 if capture else 2
    return 3 if capture else 4


def _move_order_key(move: dict) -> tuple[int, int, int, int]:
    """Sort key for a move list, four ascending levels:
      (a) move class  — checkmate, check-capture, check, capture, quiet
      (b) piece value — king highest, then Q > R > B > N > P
      (c) starting-square index
      (d) destination-square index
    The same key orders both the CoT list and the parse-tag list."""
    return (_move_class_rank(move),
            _PIECE_TOK_ORDER.index(move["piece"][-2]),
            int(move["from_sq"][len("<SQUARE_"):-1]),
            int(move["to_sq"][len("<SQUARE_"):-1]))


def format_move_tag_list(moves: list[dict]) -> str:
    """Comma-separated parse-tag list of moves (each via format_move_tag), ordered
    by `_move_order_key`. The caller prepends any question-type prefix (e.g. the
    opponent king + square for checks), space-separated, before this list."""
    return ",".join(format_move_tag(m) for m in sorted(moves, key=_move_order_key))


def format_move_cot_list(moves: list[dict], rng) -> str:
    """', '-separated CoT list of moves (each via format_move_cot), same ordering
    as the parse-tag list. Call once per move category the caller wants to list
    separately (e.g. capturing the checker vs. moving the king away)."""
    return ", ".join(format_move_cot(m, rng) for m in sorted(moves, key=_move_order_key))


# ---------------------------------------------------------------------------
# Forward (stage-3/4) move-sequence prose
# ---------------------------------------------------------------------------

# Prompt frame (command form): {question} is spliced in mid-sentence with its
# first character lowercased (see compose_forward). That is safe for every task:
# questions begin either with a plain word ("What"/"In"/"Where"/"How") or with a
# square token ("<SQUARE_n> contains..."), whose leading "<" has no case so
# lowercasing is a no-op and the token is preserved.
AFTER_SEQUENCE_TEMPLATES = [
    "After the moves {sequence}, {question}",
    "After playing {sequence}, {question}",
    "Once the moves {sequence} have been played, {question}",
    "Starting from this position and playing {sequence}, {question}",
]

# Connectors for the move-by-move answer CoT, sampled per ply.
_SEQ_FIRST = ["First,", "To start,", "Initially,"]
_SEQ_MID   = ["Then,", "Next,", "After that,", "Subsequently,"]
_SEQ_LAST  = ["Finally,", "Lastly,", "And finally,"]


# Distinctness balancer: sentinels appended to answer_class so the driver's
# frequency counter balances forward records whose final-board answer differs
# from the initial-board answer against those it leaves unchanged. They are
# balancing-only metadata (eval/graders read the parse tag, not answer_class);
# the double angle brackets keep them from colliding with real <...> tokens.
DISTINCT_CHANGED = "<<changed>>"
DISTINCT_SAME    = "<<unchanged>>"


def distinctness_marker(changed: bool) -> str:
    return DISTINCT_CHANGED if changed else DISTINCT_SAME


def format_move_sequence_tag(move_dicts: list[dict]) -> str:
    """Prompt move list: each move via `format_move_tag`, oxford-comma joined, in
    CHRONOLOGICAL (play) order — not the `_move_order_key` sort used for answers."""
    return join_and([format_move_tag(m) for m in move_dicts])


def format_move_sequence_cot(move_dicts: list[dict], rng) -> str:
    """'First, <m1>. Then, <m2>. Finally, <mN>.' — each move via `format_move_cot`,
    chronological, with sampled connectors. A single move gets one 'First,' clause."""
    n = len(move_dicts)
    parts = []
    for i, m in enumerate(move_dicts):
        lead = (rng.choice(_SEQ_FIRST) if i == 0
                else rng.choice(_SEQ_LAST) if i == n - 1
                else rng.choice(_SEQ_MID))
        parts.append(f"{lead} {format_move_cot(m, rng)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stage-5 motif annotation prose
# ---------------------------------------------------------------------------

# These registries contain prose only. Detection, thresholds, payload
# construction, and side ordering belong to the motif task / tree code. Every
# complete template has a distinct, controlled lead-in; the task maps all three
# lead-ins back to the same canonical motif id when grading generations.
MATERIAL_MOTIF_TEMPLATES = {
    "material_imbalance": (
        "Material imbalance: after common material is cancelled, {white_clause}, while {black_clause}; {material_relation}.",
        "Unequal material: removing like-for-like pieces leaves this balance: {white_clause}, and {black_clause}; {material_relation}.",
        "Material difference: once shared material is removed, {white_clause}, whereas {black_clause}; {material_relation}.",
    ),
}


POSITION_MOTIF_TEMPLATES = {
    "pawn_structure": (
        "Pawn structure: {side}'s pawn formation is {structure_effect} by {features}.",
        "Structural pawns: {features} make {side}'s pawn structure {structure_result}.",
        "Pawn formation: {side} {structure_verb} from {features}.",
    ),
    "knight_outpost": (
        "Knight outpost: {side}'s {knights} {be} firmly established on {squares}.",
        "Secure knights: {side}'s {knights} {occupy} stable outposts on {squares}.",
        "Outposted knights: the squares {squares} provide strong posts for {side}'s {knights}.",
    ),
    "active_bishop_pair": (
        "Active bishop pair: {side}'s bishops on {squares} form an active pair.",
        "Bishops working together: the bishops on {squares} give {side} active play.",
        "Two active bishops: {side} benefits from the coordinated bishops on {squares}.",
    ),
    "active_bishop": (
        "Active bishops: {side}'s {bishops} on {squares} {be} actively placed.",
        "Bishop activity: {side}'s {bishops} on {squares} {enjoy} useful scope.",
        "Well-placed bishops: the {bishops} on {squares} {be} {active_piece_noun} for {side}.",
    ),
    "bad_bishop": (
        "Bad bishops: {side}'s {bishops} on {squares} {be} hemmed in by pawns of {possessive} own color.",
        "Restricted bishops: pawns on the same color limit {side}'s {bishops} on {squares}.",
        "Hemmed-in bishops: {side}'s {bishops} on {squares} {have} little scope behind {possessive} own pawns.",
    ),
    "bad_knight": (
        "Bad knights: {side}'s {knights} on {squares} {have} very few useful squares.",
        "Restricted knights: {side}'s {knights} on {squares} {be} short of useful moves.",
        "Poorly placed knights: limited mobility makes {side}'s {knights} on {squares} ineffective.",
    ),
    "bad_rook": (
        "Bad rooks: {side}'s {rooks} on {squares} {be} shut in and {have} very few useful moves.",
        "Restricted rooks: {side}'s {rooks} on {squares} {lack} useful mobility.",
        "Inactive rooks: limited mobility leaves {side}'s {rooks} on {squares} poorly placed.",
    ),
    "active_rook": (
        "Active rooks: {side}'s {rooks} on {squares} {have} useful mobility.",
        "Rook activity: {side} has {rooks} actively placed on {squares}.",
        "Well-placed rooks: {side}'s {rooks} on {squares} {give} active play.",
    ),
    "colour_complexion": (
        "Colour-complex weakness: {side} is weak on the {complex} squares.",
        "Weak squares: weaknesses on the {complex} squares trouble {side}.",
        "Vulnerable color complex: {side}'s pawn placement leaves the {complex} squares weak.",
    ),
    "space_advantage": (
        "Space advantage: {side} has more space {region}.",
        "More territory: {side} controls more room {region}.",
        "Spatial edge: {side} enjoys a space advantage {region}.",
    ),
    "passed_pawns": (
        "Passed pawns: {side} has {passed_pawn_noun} on {squares}.",
        "Advanced passers: {side} owns {passed_pawn_noun} on {squares}.",
        "Passed-pawn strength: the {passed_pawn_noun} on {squares} {give} {side} a structural asset.",
    ),
}


TACTICAL_MOTIF_TEMPLATES = {
    "attraction": (
        "Attraction after {moves}: {move} draws {piece} onto {square}, setting up {follow_up}.",
        "Luring tactic after {moves}: {piece} is enticed to {square} by {move}, setting up {follow_up}.",
        "Drawn onto the target after {moves}: {move} pulls {piece} to {square}, after which {follow_up} follows.",
    ),
    "deflection": (
        "Deflection after {moves}: {piece} is drawn away from {target}, making {winning_move} possible.",
        "Defender displaced after {moves}: {move} pulls {piece} away from the defense of {target}, after which {winning_move} wins it.",
        "Defense diverted after {moves}: by forcing {piece} away from {target}, {move} makes {winning_move} possible.",
    ),
    "hanging_piece": (
        "Hanging piece after {moves}: {piece} is left undefended and is captured by {move}.",
        "Loose piece after {moves}: {move} captures the unprotected {piece}.",
        "Undefended piece after {moves}: the exposed {piece} is lost to {move}.",
    ),
    "trapped_piece": (
        "Trapped piece after {moves}: {piece} has no adequate escape and is won by {move}.",
        "No escape after {moves}: the trapped {piece} cannot avoid {move}.",
        "Piece boxed in after {moves}: {piece} is unable to escape, allowing {move} to win it.",
    ),
    "skewer": (
        "Skewer after {moves}: {attacker} drives away {front_target} and wins {rear_target} with {winning_move}.",
        "Line-piece skewer after {moves}: the attack by {attacker} forces {front_target} aside, exposing {rear_target} to {winning_move}.",
        "Target behind the target after {moves}: {front_target} must leave {attacker}'s line, exposing {rear_target} to {winning_move}.",
    ),
    "interference": (
        "Interference after {moves}: {blocking_move} cuts off {defender}'s defense of {target}, making {winning_move} possible.",
        "Defensive line blocked after {moves}: {blocking_move} comes between {defender} and {target}, after which {winning_move} wins it.",
        "Connection broken after {moves}: {blocking_move} interferes with {defender}, leaving {target} vulnerable to {winning_move}.",
    ),
    "intermezzo": (
        "Intermezzo after {moves}: instead of immediately recapturing on {square}, {intermediate_move} is played first, followed by {recapture}.",
        "In-between move after {moves}: {intermediate_move} is inserted before {recapture} on {square}.",
        "Recapture delayed after {moves}: before recapturing on {square} with {recapture}, {side} first plays {intermediate_move}.",
    ),
    "pin": (
        "Pin after {moves}: {pinner} pins {pinned_piece} to {king}, making {move} possible.",
        "Pinned defender after {moves}: because {pinned_piece} is tied to {king} by {pinner}, it cannot prevent {move}.",
        "Piece unable to move after {moves}: {pinner} immobilizes {pinned_piece} against {king}, making {move} possible.",
    ),
    "x_ray_attack": (
        "X-ray attack after {moves}: {attacker} attacks through {screening_piece}; after {screening_move}, {move} wins that piece.",
        "Attack through a piece after {moves}: {attacker}'s line passes through {screening_piece}, which is lost to {move} after {screening_move}.",
        "Hidden line attack after {moves}: once {screening_piece} enters the line with {screening_move}, {attacker} wins it with {move}.",
    ),
    "collinear_move": (
        "Collinear move after {moves}: with {move}, {piece} stays aligned with {enemy_piece} along the {line}.",
        "Pieces kept on one line after {moves}: {move} leaves {piece} and {enemy_piece} aligned along the {line}.",
        "Alignment maintained after {moves}: with {move}, {piece} moves along the {line} while keeping {enemy_piece} in its path.",
    ),
    "fork": (
        "Fork after {moves}: {attacker} simultaneously attacks {targets}.",
        "Double attack after {moves}: {move} lets {attacker} attack {targets} at once.",
        "Multiple targets after {moves}: from {square}, {attacker} forks {targets} with {move}.",
    ),
    "discovered_attack": (
        "Discovered attack after {moves}: {uncovering_move} clears the line for {revealed_attacker} to attack {target}, which is won by {winning_move}.",
        "Attack uncovered after {moves}: by moving aside with {uncovering_move}, {piece} reveals {revealed_attacker}'s attack on {target}, making {winning_move} possible.",
        "Hidden attacker revealed after {moves}: {uncovering_move} opens the line for {revealed_attacker}, exposing {target} to {winning_move}.",
    ),
    "discovered_check": (
        "Discovered check after {moves}: {uncovering_move} clears the line for {revealed_checker} to check {king}.",
        "Check uncovered after {moves}: by moving aside with {uncovering_move}, {piece} reveals a check from {revealed_checker} against {king}.",
        "Hidden check revealed after {moves}: {uncovering_move} opens {revealed_checker}'s line to {king}.",
    ),
    "back_rank_mate": (
        "Back-rank mate after {moves}: {mating_move} checkmates {king} on the back rank.",
        "King trapped on the back rank after {moves}: {king} has no escape and is mated by {mating_move}.",
        "Back-rank checkmate after {moves}: with no flight square available, {mating_move} mates {king}.",
    ),
}

for _mate_n in range(1, 6):
    TACTICAL_MOTIF_TEMPLATES[f"mate_in_{_mate_n}"] = (
        f"Mate in {_mate_n} after {{moves}}: {{side}} forces checkmate with {{mating_line}}.",
        f"Forced mate in {_mate_n} after {{moves}}: the line {{mating_line}} checkmates for {{side}}.",
        f"Checkmate in {_mate_n} after {{moves}}: {{side}} delivers mate with {{mating_line}}.",
    )


MOTIF_TEMPLATES = {
    **MATERIAL_MOTIF_TEMPLATES,
    **POSITION_MOTIF_TEMPLATES,
    **TACTICAL_MOTIF_TEMPLATES,
}


def render_motif_prose(motif: str, values: dict, rng, variant: int | None = None) -> str:
    """Render one normalized motif payload with one of its three prose forms."""
    try:
        templates = MOTIF_TEMPLATES[motif]
    except KeyError as exc:
        raise ValueError(f"unknown motif: {motif!r}") from exc
    if variant is None:
        template = rng.choice(templates)
    else:
        if not 0 <= variant < len(templates):
            raise ValueError(f"motif variant must be 0..{len(templates) - 1}")
        template = templates[variant]
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"{motif} payload is missing {exc.args[0]!r}") from exc


def compose_forward(parts: dict, move_dicts: list[dict], rng) -> dict:
    """Stitch a task's base `parts` (computed on the FINAL board) with the move
    sequence into one forward record. `parts` keys: question, body, parse_tag,
    answer_class, question_type. The prompt gets the 'After {sequence}' frame; the
    answer leads with the move-by-move CoT, then the base body, then the parse tag."""
    seq_tag = format_move_sequence_tag(move_dicts)
    q = parts["question"]
    q = q[:1].lower() + q[1:]   # command form; no-op when the question starts with a token ("<...")
    prompt = rng.choice(AFTER_SEQUENCE_TEMPLATES).format(sequence=seq_tag, question=q)
    answer = f"{format_move_sequence_cot(move_dicts, rng)} {parts['body']}\n\n{parts['parse_tag']}"
    return {
        "question":      prompt,
        "answer":        answer,
        "question_type": parts["question_type"],
        "answer_class":  parts["answer_class"],
    }


def parts_from_render(render: dict) -> dict:
    """Split a task's base `_render` output (computed on the FINAL board) into the
    `parts` dict `compose_forward` expects. A base render combines its prose body
    and parse tag as ``f"{body}\\n\\n{parse_tag}"``; that blank line is the only
    ``"\\n\\n"`` (bodies are single paragraphs, parse tags use single newlines),
    so an rsplit cleanly recovers the two halves without the task having to expose
    them separately — forward tasks layer onto the base `_render` as-is."""
    body, parse_tag = render["answer"].rsplit("\n\n", 1)
    return {
        "question":      render["question"],
        "body":          body,
        "parse_tag":     parse_tag,
        "answer_class":  render["answer_class"],
        "question_type": render["question_type"],
    }


def encode_piece_count(piece_tok: str, n: int) -> str:
    """Compact `<PIECE>?N` form for parse_tag + answer_class.

    n=1 -> bare token (e.g. '<PIECE_WK>'); n>1 -> token followed by the
    count (e.g. '<PIECE_WP>2'). Reused across piece_on_{file,rank,diagonal}
    and piece_count. Grader pairs with the regex
    `(<PIECE_[A-Z]+>|<SQUARE_[A-Z0-9]+>|<EMPTY>)(\\d+)?` to read counts back.
    """
    return f"{piece_tok}{n}" if n > 1 else piece_tok


# ---------------------------------------------------------------------------
# Group-task prose (file / rank / diagonal)
# ---------------------------------------------------------------------------

def format_piece_counts(counts_in_order: list[tuple[str, int]]) -> str:
    """[('<PIECE_WB>', 2), ('<PIECE_BR>', 1)] -> '2 <PIECE_WB>s and 1 <PIECE_BR>'.

    The piece token is pluralized via `plural()` for n != 1 (cosmetic prose
    only; parse_tag/answer_class use the compact count encoding, not this)."""
    return join_and([f"{c} {plural(tok, c)}" for tok, c in counts_in_order])


def format_square_breakdown(items: list[tuple[str, Optional[str]]]) -> str:
    """Per-square CoT walk.

    Each item is (sq_tok, piece_tok_or_None). None = empty square.
    e.g. [('<SQUARE_A1>', '<PIECE_WK>'), ('<SQUARE_A2>', None)]
         -> '<SQUARE_A1> has <PIECE_WK> and <SQUARE_A2> is empty'
    """
    parts = [
        f"{sq_tok} is empty" if piece_tok is None else f"{sq_tok} has {piece_tok}"
        for sq_tok, piece_tok in items
    ]
    return join_and(parts)


# ---------------------------------------------------------------------------
# Line-task QA structure (file / rank / diagonal)
# ---------------------------------------------------------------------------

def line_facts(line_sqs: tuple, board: BoardRepr) -> dict:
    """Shared start/end + parse_tag + answer_class bundle for line tasks.

    Returns:
        start_tok    : str  — sq_tok of the line's first square
        end_tok      : str  — sq_tok of the line's last square
        ordered      : list[(piece_tok, n)]  — empty for an open line
        parse_tag    : str  — compact "<PIECE>?N" concat, or "<EMPTY>"
        answer_class : list[str]  — [start_tok, end_tok, *pieces_flat] or
                                    [start_tok, end_tok, EMPTY_TOKEN]
    """
    start_tok = board.sq_tok(line_sqs[0])
    end_tok   = board.sq_tok(line_sqs[-1])
    ordered   = line_piece_counts(line_sqs, board)
    if not ordered:
        return {
            "start_tok":    start_tok,
            "end_tok":      end_tok,
            "ordered":      [],
            "parse_tag":    EMPTY_TOKEN,
            "answer_class": [start_tok, end_tok, EMPTY_TOKEN],
        }
    pieces_flat  = [encode_piece_count(p, c) for p, c in ordered]
    parse_tag    = "".join(pieces_flat)
    answer_class = [start_tok, end_tok] + pieces_flat
    return {
        "start_tok":    start_tok,
        "end_tok":      end_tok,
        "ordered":      ordered,
        "parse_tag":    parse_tag,
        "answer_class": answer_class,
    }


# ---------------------------------------------------------------------------
# Checkmate explanation (king escape squares)
# ---------------------------------------------------------------------------

def format_mate_adjacency(board: BoardRepr) -> str:
    """Prose for why a checkmated king has no escape: each square adjacent to the
    side-to-move king, in increasing square-number order, is either occupied by an
    own piece or attacked by the opponent. Attackers are computed with the king
    lifted off its square, so a checker that covers a retreat square along its line
    is reported correctly."""
    cb = board.chess_board
    stm = cb.turn
    king_sq = cb.king(stm)
    lifted = cb.copy(stack=False)
    lifted.remove_piece_at(king_sq)
    adj = [s for s in chess.SQUARES if chess.square_distance(king_sq, s) == 1]
    clauses = []
    for s in sorted(adj, key=lambda sq: _sq_num(board.sq_tok(sq))):
        sq_tok = board.sq_tok(s)
        pc = cb.piece_at(s)
        if pc is not None and pc.color == stm:
            clauses.append(f"{sq_tok} is occupied by our own {board.piece_at(s)}")
        else:
            atts = sorted(lifted.attackers(not stm, s))
            att_join = join_and([f"the {board.piece_at(a)} on {board.sq_tok(a)}" for a in atts])
            clauses.append(f"{sq_tok} is attacked by {att_join}")
    return join_and(clauses)
