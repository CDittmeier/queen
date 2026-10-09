"""Annotate a game: Stockfish picks the critical moments, QUEEN explains them.

uv run --frozen --group mac python -m mac_inference.annotate game.pgn

For each critical moment QUEEN analyzes the position from the side to move.
When QUEEN would have played something else, it also analyzes the position
after the move actually played, from the opponent's side. Stockfish checks
every move QUEEN recommends and every line it gives, so the commentary marks
what was verified. Writes an annotated PGN, a Markdown summary, and JSON with
the full model output to the output directory.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import chess
import chess.engine
import chess.pgn

PIECES = {"pawn": "", "knight": "N", "bishop": "B", "rook": "R", "queen": "Q"}
PIECES["king"] = "K"
PIECE_WORD = "|".join(PIECES)
# Decoded QUEEN prose: "white knight g1-f3", "white pawn d4 takes black pawn e5".
MOVE_PHRASE = re.compile(
    rf"(?:\d+\s*(?:\.\.\.|…|\.)\s*|\.\.\.|…)?(?P<color>white|black) "
    rf"(?P<piece>{PIECE_WORD}) (?P<src>[a-h][1-8])"
    rf"(?:-|(?P<capture> takes (?:white|black) (?:{PIECE_WORD}) ))(?P<dst>[a-h][1-8])"
)
FIELDS = (
    r"\n(?:BEST_MOVE|CRITICAL_LINE|PRINCIPAL_VARIATION|PROMISING_MOVES|EVALUATION):"
)


# Stockfish ---------------------------------------------------------------


def expected(score, color):
    """Expected score (0-1) for ``color`` from a Stockfish score."""
    return score.pov(color).wdl(model="sf").expectation()


def survey(game, engine, limit):
    """Stockfish's view of every position: best move, alternatives, played move."""
    plies = []
    board = game.board()
    for move in game.mainline_moves():
        infos = engine.analyse(board, limit, multipv=2)
        best = infos[0]
        color = board.turn
        after = board.copy()
        after.push(move)
        # Material is counted after the opponent's best reply, so a sacrifice
        # such as Nxf7 (which takes a pawn first) still shows as one.
        replied = after.copy()
        if after.is_game_over():
            played_score = expected_terminal(after, color)
        else:
            reply = engine.analyse(after, limit)
            played_score = expected(reply["score"], color)
            replied.push(reply["pv"][0])
        plies.append(
            {
                "ply": len(plies),
                "fen": board.fen(),
                "move": move,
                "san": board.san(move),
                "color": color,
                "best": best["pv"][0],
                "best_san": board.san(best["pv"][0]),
                "best_score": expected(best["score"], color),
                "second_score": (
                    expected(infos[1]["score"], color) if len(infos) > 1 else None
                ),
                "played_score": played_score,
                "cp": best["score"].white().score(mate_score=10000),
                "material": material(replied, color) - material(board, color),
            }
        )
        board.push(move)
    return plies


def expected_terminal(board, color):
    outcome = board.outcome()
    if outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == color else 0.0


def material(board, color):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5}
    values[chess.QUEEN] = 9
    return sum(
        len(board.pieces(piece, color)) * value
        - len(board.pieces(piece, not color)) * value
        for piece, value in values.items()
    )


def classify(ply):
    """Why a move is critical, with its PGN annotation symbol."""
    # Once the game is decided, forced material losses aren't news.
    if not 0.05 < ply["best_score"] < 0.95:
        return None, ""
    loss = ply["best_score"] - ply["played_score"]
    second = ply["second_score"]
    # Material given up after the opponent's best reply, position intact.
    if ply["material"] <= -2 and loss < 0.25:
        return "sacrifice", "!" if loss < 0.05 else "!?"
    if loss >= 0.20:
        return "blunder", "??"
    if loss >= 0.10:
        return "mistake", "?"
    if ply["move"] == ply["best"] and second is not None:
        if ply["best_score"] - second >= 0.15:
            return "only move", "!"
    return None, ""


def critical_moments(plies, count):
    """The most important plies, by how much rode on the move."""
    scored = []
    for ply in plies:
        reason, symbol = classify(ply)
        if reason is None:
            continue
        loss = ply["best_score"] - ply["played_score"]
        stakes = max(loss, (ply["best_score"] - (ply["second_score"] or 0)))
        scored.append((stakes + (0.3 if reason == "sacrifice" else 0), ply))
        ply["reason"], ply["symbol"] = reason, symbol
    chosen = sorted(scored, key=lambda item: item[0], reverse=True)[:count]
    return sorted((ply for _, ply in chosen), key=lambda ply: ply["ply"])


# QUEEN output --------------------------------------------------------------


def moves_in(text, board):
    """Replays QUEEN's move phrases from ``board``; stops at the first that isn't
    legal. Returns the legal moves and whether every phrase was legal."""
    board = board.copy()
    moves = []
    for match in MOVE_PHRASE.finditer(text):
        src, dst = chess.parse_square(match["src"]), chess.parse_square(match["dst"])
        promotion = (
            chess.QUEEN
            if match["piece"] == "pawn" and chess.square_rank(dst) in (0, 7)
            else None
        )
        move = chess.Move(src, dst, promotion)
        if match["color"] != ("white" if board.turn else "black"):
            return moves, False
        if move not in board.legal_moves:
            return moves, False
        moves.append(move)
        board.push(move)
    return moves, True


def field(text, name):
    found = re.search(rf"(?m)^{name}:\s*(.*)$", text)
    return found[1].strip() if found else ""


def prose(text):
    return re.split(FIELDS, re.sub(r"^ANALYSIS:\s*", "", text))[0].strip()


def to_san(text):
    """Shortens decoded move phrases to algebraic notation for readability."""

    def short(match):
        piece, src, dst = match["piece"], match["src"], match["dst"]
        if piece == "king" and src[0] == "e" and abs(ord(dst[0]) - ord("e")) == 2:
            return "O-O" if dst[0] == "g" else "O-O-O"
        if match["capture"]:
            return (src[0] if piece == "pawn" else PIECES[piece]) + "x" + dst
        return PIECES[piece] + dst

    return MOVE_PHRASE.sub(short, text)


def summary(text, sentences=2):
    """The first sentences of the paragraph that names the best move."""
    paragraphs = [p for p in prose(text).split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    chosen = next(
        (p for p in paragraphs if "best move" in p.lower()),
        paragraphs[0],
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", to_san(chosen.strip()))
    return " ".join(parts[:sentences])


def verify(analysis, board, engine, limit):
    """Checks QUEEN's recommendation and line against Stockfish."""
    color = board.turn
    best = engine.analyse(board, limit)
    report = {"best_move_legal": analysis["best_move_uci"] is not None}
    if analysis["best_move_uci"]:
        move = chess.Move.from_uci(analysis["best_move_uci"])
        after = board.copy()
        after.push(move)
        score = (
            expected_terminal(after, color)
            if after.is_game_over()
            else expected(engine.analyse(after, limit)["score"], color)
        )
        # Separate short searches can disagree slightly; a loss is never negative.
        loss = expected(best["score"], color) - score
        report["best_move_loss"] = round(max(0.0, loss), 3)
    line = field(analysis["text"], "CRITICAL_LINE") or field(
        analysis["text"], "PRINCIPAL_VARIATION"
    )
    moves, legal = moves_in(line, board)
    report["line_legal"] = legal and bool(moves)
    report["line"] = board.variation_san(moves) if moves else ""
    # The first move in the line that Stockfish calls a mistake.
    replay = board.copy()
    report["line_sound_plies"] = 0
    for move in moves:
        mover = replay.turn
        top = expected(engine.analyse(replay, limit)["score"], mover)
        replay.push(move)
        if replay.is_game_over():
            got = expected_terminal(replay, mover)
        else:
            got = expected(engine.analyse(replay, limit)["score"], mover)
        if top - got >= 0.10:
            break
        report["line_sound_plies"] += 1
    return report


# Output --------------------------------------------------------------------


def side(color):
    return "White" if color == chess.WHITE else "Black"


def move_label(ply):
    number = ply["ply"] // 2 + 1
    return f"{number}{'.' if ply['color'] == chess.WHITE else '...'}{ply['san']}"


def comment(moment):
    ply = moment["ply"]
    parts = [
        f"{ply['reason'].capitalize()}. Stockfish prefers {ply['best_san']}"
        if ply["move"] != ply["best"]
        else f"{ply['reason'].capitalize()}. Stockfish agrees."
    ]
    before = moment.get("before")
    if before:
        queen = before["analysis"]["best_move_san"] or "no legal move"
        agree = (
            "agrees"
            if before["analysis"]["best_move_uci"] == ply["move"].uci()
            else f"would play {queen}"
        )
        parts.append(f"QUEEN ({side(ply['color'])}) {agree}: {before['summary']}")
        if not before["check"]["line_legal"]:
            parts.append("(QUEEN's line contains an illegal move.)")
    after = moment.get("after")
    if after:
        parts.append(
            f"After {ply['san']}, QUEEN ({side(not ply['color'])}): {after['summary']}"
        )
    return " ".join(parts)


def write_pgn(game, moments, path):
    by_ply = {moment["ply"]["ply"]: moment for moment in moments}
    node = game
    for index, move in enumerate(list(game.mainline_moves())):
        node = node.variation(move)
        if index in by_ply:
            moment = by_ply[index]
            symbol = moment["ply"]["symbol"]
            if symbol:
                node.nags.add(
                    {
                        "!": chess.pgn.NAG_GOOD_MOVE,
                        "!?": chess.pgn.NAG_SPECULATIVE_MOVE,
                        "?": chess.pgn.NAG_MISTAKE,
                        "??": chess.pgn.NAG_BLUNDER,
                    }[symbol]
                )
            node.comment = comment(moment)
    path.write_text(str(game) + "\n")


def write_markdown(game, moments, path):
    headers = game.headers
    lines = [
        f"# {headers.get('White')} vs {headers.get('Black')}",
        "",
        f"{headers.get('Event')}, {headers.get('Site')}, {headers.get('Date')}, "
        f"round {headers.get('Round')} · {headers.get('Result')}",
        "",
    ]
    for moment in moments:
        ply = moment["ply"]
        lines += [
            f"## {move_label(ply)}{ply['symbol']} — {ply['reason']}",
            "",
            f"Stockfish: best {ply['best_san']}, evaluation {ply['cp'] / 100:+.2f}"
            " (White's view, before the move).",
            "",
        ]
        for key, title in (("before", "before the move"), ("after", "after it")):
            view = moment.get(key)
            if not view:
                continue
            analysis, check = view["analysis"], view["check"]
            verdict = (
                f"recommends {analysis['best_move_san']}"
                f" (Stockfish loss {check.get('best_move_loss', 0):.0%})"
                if analysis["best_move_san"]
                else "gave no legal recommendation"
            )
            lines += [
                f"**QUEEN, {view['side']} to move, {title}** — {verdict}",
                "",
                f"> {view['summary']}",
                "",
                f"Line: {check['line'] or '—'} · legal: {check['line_legal']}"
                f" · sound for {check['line_sound_plies']} plies",
                "",
            ]
    path.write_text("\n".join(lines))


def record(moment):
    ply = moment["ply"]
    return {
        **ply,
        "move": ply["move"].uci(),
        "best": ply["best"].uci(),
        "color": side(ply["color"]),
        **{key: moment[key] for key in ("before", "after") if key in moment},
    }


# Main ----------------------------------------------------------------------


def analyze_view(queen, board, engine, limit, cache, label):
    # Consecutive moments often share a position (after one move, before the
    # next); analyze each position once.
    if board.fen() in cache:
        return cache[board.fen()]
    print(f"  QUEEN: {label}", flush=True)
    analysis = queen.analyze(board, 0, 0, lambda *_: None)
    cache[board.fen()] = view = {
        "side": side(board.turn),
        "fen": board.fen(),
        "analysis": analysis,
        "summary": summary(analysis["text"]),
        "check": verify(analysis, board, engine, limit),
    }
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults beside the PGN.")
    parser.add_argument("--moments", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=1.0, help="Stockfish time.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".models" / "queen_pawn-8",
    )
    parser.add_argument(
        "--stockfish-only", action="store_true", help="List moments; skip QUEEN."
    )
    args = parser.parse_args()
    game = chess.pgn.read_game(args.pgn.open())
    if game is None:
        parser.error(f"No game found in {args.pgn}")
    stockfish = shutil.which("stockfish")
    if stockfish is None:
        parser.error("Install Stockfish first: brew install stockfish")
    output = args.output or args.pgn.with_suffix("")
    output.mkdir(parents=True, exist_ok=True)
    limit = chess.engine.Limit(time=args.seconds)

    with chess.engine.SimpleEngine.popen_uci(stockfish) as engine:
        engine.configure({"Threads": 2})
        print("Stockfish: surveying the game…", flush=True)
        plies = survey(game, engine, limit)
        moments = [{"ply": ply} for ply in critical_moments(plies, args.moments)]
        for moment in moments:
            ply = moment["ply"]
            print(f"  {move_label(ply)}{ply['symbol']}  {ply['reason']}", flush=True)
        if not args.stockfish_only:
            from .game import QueenEngine

            print("Loading QUEEN…", flush=True)
            queen = QueenEngine(args.model_dir)
            moves = list(game.mainline_moves())
            cache = {}
            for moment in moments:
                ply = moment["ply"]
                board = game.board()
                for move in moves[: ply["ply"]]:
                    board.push(move)
                moment["before"] = analyze_view(
                    queen, board, engine, limit, cache, f"{move_label(ply)} (before)"
                )
                chose = moment["before"]["analysis"]["best_move_uci"]
                board.push(ply["move"])
                if chose != ply["move"].uci() and not board.is_game_over():
                    moment["after"] = analyze_view(
                        queen, board, engine, limit, cache, f"{move_label(ply)} (after)"
                    )
    stem = args.pgn.stem
    write_pgn(game, moments, output / f"{stem}.annotated.pgn")
    write_markdown(game, moments, output / f"{stem}.md")
    (output / f"{stem}.json").write_text(
        json.dumps(list(map(record, moments)), indent=2)
    )
    print(f"Wrote {output}/", flush=True)


if __name__ == "__main__":
    main()
