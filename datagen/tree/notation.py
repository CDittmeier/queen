"""Mode-aware rendering of pieces / squares / moves for the search narrative.

Two renderings of the SAME entity, both POV-anchored to the ROOT mover (the side
to move at the root position, matching the stage forward-task fixed `pov_anchor`
so <PIECE_M*/O*> and the square mirror stay constant across the whole line):

  * mode="token"  -> the special tokens a chess-LM understands
                     (<SQUARE_1..64>, <PIECE_M*/O*>, compact move tags / CoT prose).
  * mode="human"  -> English piece words + algebraic squares + SAN, for us to read.

Anything the model has never seen (piece names, algebraic squares, SAN) is replaced
in token mode; English connectives ("hits", "on", the move numbers) stay in both.
Reuses the exact stage renderers (utils.board_representation / datagen.prose) so
token output is identical to the stage-3/4 data format.
"""
import hashlib
import random
import re

import chess

from utils.board_representation import BoardRepr
from datagen.prose import format_move_cot, format_move_tag

PIECE_WORD = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

# piece-letter (SAN) / piece-word -> python-chess piece type, for tok_free
LETTER_TO_PT = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
                 "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
WORD_TO_PT = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
               "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}
PIECE_WORD_RE = "pawn|knight|bishop|rook|queen|king"


class Notation:
    """Render pieces/squares/moves in `token` or `human` mode, anchored to root."""

    def __init__(self, root_fen: str, mode: str):
        assert mode in ("token", "human")
        self.mode = mode
        self.root_color = chess.Board(root_fen).turn
        # POV pinned to the root side to move, fixed across the whole tree.
        self.br = BoardRepr(root_fen, pov=True, pov_anchor=self.root_color)

    # ---- squares (board_sq is python-chess 0..63, a1=0) ----
    def sq(self, board_sq: int) -> str:
        return self.br.sq_tok(board_sq) if self.mode == "token" else chess.square_name(board_sq)

    # ---- pieces (color = absolute chess color; token mode -> M/O vs root) ----
    def piece(self, piece_type: int, color: bool) -> str:
        if self.mode == "token":
            return self.br.piece_tok_for(color, piece_type)
        return PIECE_WORD[piece_type]

    def piece_on(self, board: chess.Board, board_sq: int) -> str:
        pc = board.piece_at(board_sq)
        return self.piece(pc.piece_type, pc.color) if pc else self.sq(board_sq)

    # ---- moves ----
    def move_dict(self, board: chess.Board, move: chess.Move) -> dict:
        us = board.turn
        frm, to = move.from_square, move.to_square
        d = {"piece": self.br.piece_tok_for(us, board.piece_at(frm).piece_type),
             "from_sq": self.br.sq_tok(frm), "to_sq": self.br.sq_tok(to),
             "captured_piece": None, "en_passant_square": None,
             "castle_type": None, "promotion_to": None, "check_status": None}
        if board.is_en_passant(move):
            cap_sq = to + (-8 if us == chess.WHITE else 8)
            d["captured_piece"] = self.br.piece_tok_for(not us, chess.PAWN)
            d["en_passant_square"] = self.br.sq_tok(cap_sq)
        elif board.is_capture(move):
            d["captured_piece"] = self.br.piece_tok_for(not us, board.piece_at(to).piece_type)
        if board.is_kingside_castling(move):
            d["castle_type"] = "kingside"
        elif board.is_queenside_castling(move):
            d["castle_type"] = "queenside"
        if move.promotion:
            d["promotion_to"] = self.br.piece_tok_for(us, move.promotion)
        board.push(move)
        d["check_status"] = ("checkmate" if board.is_checkmate()
                             else "check" if board.is_check() else None)
        board.pop()
        return d

    def move_tag(self, board: chess.Board, move: chess.Move) -> str:
        """Compact form (`<piece><from>[<cap>]<to>`) for move sequences / line notation."""
        if self.mode == "token":
            return format_move_tag(self.move_dict(board, move))
        return board.san(move)

    def move_prose(self, board: chess.Board, move: chess.Move) -> str:
        """Prose form for standalone call-outs (`<piece> on <from> takes <piece> on <sq>`)."""
        if self.mode == "token":
            seed = int(hashlib.md5((board.fen() + move.uci()).encode()).hexdigest(), 16) & 0xFFFFFFFF
            return format_move_cot(self.move_dict(board, move), random.Random(seed))
        return board.san(move)

    def line(self, root_fen: str, moves) -> str:
        """Render a move sequence as `1. <tag> <tag> 2. <tag> ...`, move numbers from
        1 (Black-to-move root shows the first ply as `1... <tag>`)."""
        b = chess.Board(root_fen)
        parts, num = [], 1
        for i, m in enumerate(moves):
            tag = self.move_tag(b, m)
            if b.turn == chess.WHITE:
                parts.append(f"{num}. {tag}")
            else:
                parts.append(f"{num}... {tag}" if i == 0 else tag)
                num += 1
            b.push(m)
        return " ".join(parts)

    # ---- free-text tokeniser (token mode only) --------------------------------
    # The HCE rationale clauses are generated as English and can still carry raw
    # chess entities (algebraic squares, SAN, "White"/"Black", piece words). In
    # token mode NOTHING may be human-speak: every square -> <SQUARE_n>, every
    # named piece -> <PIECE_M*/O*>, colours -> we/the opponent. Two facts make this
    # exact: (a) the POV square mapping is fixed tree-wide (any algebraic square
    # tokenises unambiguously), and (b) a piece is referenced as "<piece> on
    # <square>", so a board lookup at that square gives the right M*/O* token.
    def sqtok(self, name: str) -> str:
        return self.br.sq_tok(chess.parse_square(name))

    def sqtok_to_board(self, sqtok: str) -> int:
        n = int(sqtok[len("<SQUARE_"):-1]) - 1
        return (n ^ 56) if self.br._is_black_pov else n

    def color_pt_at(self, board_sq, boards):
        for b in boards:
            if b is not None:
                pc = b.piece_at(board_sq)
                if pc is not None:
                    return pc.color, pc.piece_type
        return None, None

    def tok_free(self, text: str, cf_fen: str = None, pf_fen: str = None,
                 default_color: "chess.Color | None" = None) -> str:
        """Replace every residual human-speak chess entity in `text` with tokens.
        No-op outside token mode. `cf_fen`/`pf_fen` (child/parent positions) are
        used only to look up a referenced piece's colour; squares never need them.
        `default_color` is the fallback side for a bare piece word (a concept term
        with no square/possessive to disambiguate) — pass the clause's mover."""
        if self.mode != "token" or not text:
            return text
        boards = [chess.Board(cf_fen) if cf_fen else None,
                  chess.Board(pf_fen) if pf_fen else None]

        def ptok(color, pt):
            return self.br.piece_tok_for(color, pt)

        def col_side(color):
            return "we" if color == self.root_color else "the opponent"

        def col_poss(color):
            return "our" if color == self.root_color else "the opponent's"

        # --- file letters ("d-file") -> the datagen "{start_tok}{end_tok} file"
        # form, naming the file by its two endpoint SQUARE tokens (see
        # datagen/tasks/piece_on_file.py + prose.line_facts). Endpoints ordered
        # bottom-up in render order, matching BoardRepr.files().
        def file(m):
            fi = ord(m.group(1)) - ord("a")
            toks = [self.br.sq_tok(chess.square(fi, 0)), self.br.sq_tok(chess.square(fi, 7))]
            toks.sort(key=lambda t: int(t[len("<SQUARE_"):-1]))
            return f"{toks[0]}{toks[1]} file"
        text = re.sub(r"\b([a-h])-file\b", file, text)
        # Passed-pawn clauses use file-labelled shorthand ("our d-passer").
        # Reuse the same endpoint-token representation as named files.
        text = re.sub(r"\b([a-h])-passer\b",
                      lambda m: file(m).removesuffix(" file") + " passer", text)
        # long diagonal "a1-h8" -> the two endpoint SQUARE tokens (same line convention as files)
        def diagonal(m):
            a, b = m.group(0).split("-")
            toks = [self.br.sq_tok(chess.parse_square(a)), self.br.sq_tok(chess.parse_square(b))]
            toks.sort(key=lambda t: int(t[len("<SQUARE_"):-1]))
            return f"{toks[0]}{toks[1]}"
        text = re.sub(r"\b(?:a1-h8|h8-a1|h1-a8|a8-h1)\b", diagonal, text)
        # bishop-pair idiom: "deprives White of the bishop pair" / "gives Black the bishop pair"
        def bishop_pair(m):
            verb, col = m.group(1), (m.group(2) == "White")
            who = "us" if col == self.root_color else "the opponent"
            link = "of the" if verb == "deprives" else "the"
            return f"{verb} {who} {link} {ptok(col, chess.BISHOP)} pair"
        text = re.sub(r"\b(deprives|gives) (White|Black) (?:of the|the) bishop pair\b", bishop_pair, text)
        # a colour in object position ("... for White ...") -> "us"/"the opponent" (not subject "we")
        text = re.sub(r"\bfor (White|Black)\b",
                      lambda m: "for " + ("us" if (m.group(1) == "White") == self.root_color else "the opponent"), text)
        # castling "king-side"/"queen-side" carry a piece word; datagen also uses
        # "short"/"long" (format_move_cot), so use those (stage-consistent, piece-free).
        text = re.sub(r"\bking-?side\b", "short", text)
        text = re.sub(r"\bqueen-?side\b", "long", text)

        # --- Layer A: algebraic entities ---
        # possessive colour + king  (e.g. "White's king" -> <PIECE_?K>)
        text = re.sub(r"\b(White|Black)'s king\b",
                      lambda m: ptok(m.group(1) == "White", chess.KING), text)
        # material idiom: "White loses a knight", "Black wins the bishop pair"
        CONJ = {"loses": "lose", "wins": "win", "drops": "drop", "gains": "gain", "regains": "regain"}
        def material(m):
            col = (m.group(1) == "White")
            verb = m.group(2)
            if col == self.root_color:                      # "we lose" (not "we loses")
                subj, verb = "we", CONJ.get(verb, verb)
            else:
                subj = "the opponent"
            pair = m.group(5) or ""
            return f"{subj} {verb} {m.group(3)} {ptok(col, WORD_TO_PT[m.group(4)])}{pair}"
        text = re.sub(rf"\b(White|Black)\s+(loses|wins|drops|gains|regains)\s+"
                      rf"(a|an|the)\s+({PIECE_WORD_RE})( pair)?\b", material, text)
        # piece arrow  Xc1→g5   (colour from the from-square)
        def parrow(m):
            col, _ = self.color_pt_at(chess.parse_square(m.group(2)), boards)
            col = self.root_color if col is None else col
            return f"{ptok(col, LETTER_TO_PT[m.group(1)])}{self.sqtok(m.group(2))}→{self.sqtok(m.group(3))}"
        text = re.sub(r"\b([KQRBN])([a-h][1-8])→([a-h][1-8])\b", parrow, text)
        # pawn arrow  e2→e3
        text = re.sub(r"\b([a-h][1-8])→([a-h][1-8])\b",
                      lambda m: f"{self.sqtok(m.group(1))}→{self.sqtok(m.group(2))}", text)
        # SAN piece+square (opt capture): Bc1 / Bxf6 / Nd5
        def san(m):
            col, _ = self.color_pt_at(chess.parse_square(m.group(2)), boards)
            col = self.root_color if col is None else col
            return f"{ptok(col, LETTER_TO_PT[m.group(1)])}{self.sqtok(m.group(2))}"
        text = re.sub(r"\b([KQRBN])x?([a-h][1-8])\b", san, text)
        # pawn capture exd5 -> destination square
        text = re.sub(r"\b[a-h]x([a-h][1-8])\b", lambda m: self.sqtok(m.group(1)), text)
        # any remaining bare square
        text = re.sub(r"\b([a-h][1-8])\b", lambda m: self.sqtok(m.group(1)), text)

        # --- Layer B: piece word before an (already-tokenised) square ---
        def word_on_sq(m):
            art, word, conn, sqtok = m.group(1) or "", m.group(2), m.group(4), m.group(5)
            col, _ = self.color_pt_at(self.sqtok_to_board(sqtok), boards)
            col = self.root_color if col is None else col
            return f"{art}{ptok(col, WORD_TO_PT[word])}{conn}{sqtok}"
        text = re.sub(rf"\b(the |a |an |our |his )?({PIECE_WORD_RE})(s)?"
                      rf"(\s+(?:now\s+)?(?:on\s+|to\s+)?)(<SQUARE_\d+>)", word_on_sq, text)

        # --- leftover colours ---
        text = re.sub(r"\bWhite's\b", lambda m: col_poss(True), text)
        text = re.sub(r"\bBlack's\b", lambda m: col_poss(False), text)
        text = re.sub(r"\bWhite\b", lambda m: col_side(True), text)
        text = re.sub(r"\bBlack\b", lambda m: col_side(False), text)

        # --- Layer C: ANY remaining bare piece word -> appropriate-side token.
        # These are chess concept terms with no square to anchor them ("passed
        # pawn", "king's shelter", "the attacking queen"). Side comes from an
        # adjacent possessive if present, else default_color (the clause mover),
        # else the root mover. Guarantees zero human-speak piece words.
        dc = self.root_color if default_color is None else default_color

        def bare(m):
            poss = m.group(1) or ""
            p = poss.lower()
            if "opponent" in p:
                color = not self.root_color
            elif p.strip() in ("our", "its", "his", "my"):
                color = self.root_color
            else:
                color = dc
            return f"{poss}{ptok(color, WORD_TO_PT[m.group(2)])}"
        text = re.sub(rf"\b(the opponent's |our |its |his |my |the |a |an )?"
                      rf"({PIECE_WORD_RE})(?:s)?\b", bare, text)
        # minor/slider name a piece *category* (no single token) -> generic word
        text = re.sub(r"\b(minor|slider)\b", "piece", text)
        return text

    # residual human-speak still present in a token string (for QA / sample mds):
    # algebraic squares, file letters, SAN, castling piece-words, colours, piece words.
    RESIDUAL = re.compile(rf"\b([a-h][1-8]|[a-h]-(?:file|passer)|[KQRBN][a-h][1-8]|king-?side|queen-?side|"
                           rf"White|Black|{PIECE_WORD_RE})\b")

    @staticmethod
    def residual_humanspeak(text: str):
        return Notation.RESIDUAL.findall(text)
