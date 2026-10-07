"""Serve a local live chess game: uv run --group mac python -m mac_inference.web."""

import argparse
import errno
import json
import logging
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import chess
import chess.svg

from .game import GameError, LiveGame, QueenEngine

ASSETS = Path(__file__).resolve().parent / "ui"


@lru_cache(maxsize=12)
def piece_svg(code):
    symbol = code[1] if code[0] == "w" else code[1].lower()
    return chess.svg.piece(chess.Piece.from_symbol(symbol)).encode()


def make_server(game, port=8765):
    """Bind the server; ``game`` may be attached later as ``server.game``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            if filename:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
            self.end_headers()
            self.wfile.write(body)

        def json(self, data, status=200):
            self.send(
                json.dumps(data).encode(), "application/json; charset=utf-8", status
            )

        def valid_host(self):
            port = self.server.server_port
            host = self.headers.get("Host")
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.json({"error": "This demo is available on localhost only."}, 403)
                return False
            return True

        def do_GET(self):
            if not self.valid_host():
                return
            game = self.server.game
            path = urlsplit(self.path).path
            if path == "/api/state":
                self.json(game.snapshot())
            elif path == "/api/pgn":
                self.send(
                    game.pgn().encode(),
                    "application/x-chess-pgn",
                    filename="queen-game.pgn",
                )
            elif re.fullmatch(r"/pieces/[wb][PNBRQK]\.svg", path):
                self.send(piece_svg(path[-6:-4]), "image/svg+xml")
            elif path in ("/", "/app.js", "/styles.css"):
                name = "index.html" if path == "/" else path[1:]
                mime = {
                    "index.html": "text/html",
                    "app.js": "text/javascript",
                    "styles.css": "text/css",
                }[name]
                self.send((ASSETS / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.json({"error": "Not found"}, 404)

        def do_POST(self):
            if not self.valid_host():
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                self.json({"error": "Cross-origin requests are not allowed."}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.json({"error": "Expected JSON."}, 415)
                return
            game = self.server.game
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise GameError("Invalid request size.")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise GameError("Expected a JSON object.")
                path = urlsplit(self.path).path
                game_id, version = body.get("game_id"), body.get("version")
                if path == "/api/new":
                    answer = game.new_game(body.get("color"))
                elif path == "/api/move":
                    answer = game.move(body.get("uci"), game_id, version)
                elif path == "/api/retry":
                    answer = game.retry(game_id, version)
                elif path == "/api/undo":
                    answer = game.take_back(game_id, version)
                elif path == "/api/resign":
                    answer = game.resign(game_id, version)
                elif path == "/api/draw":
                    answer = game.claim_draw(game_id, version)
                else:
                    self.json({"error": "Not found"}, 404)
                    return
                self.json(answer)
            except (ValueError, TypeError) as error:
                self.json({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.game = game
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".models" / "queen_pawn-8",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Choose a port between 1 and 65535.")
    if not (args.model_dir / "xattn.pt").is_file():
        parser.error("Download the model first: python -m mac_inference --download")
    logging.basicConfig(level=logging.INFO)
    # Bind before loading the model so a busy port fails fast.
    try:
        server = make_server(None, args.port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        parser.exit(
            1,
            f"Port {args.port} is already in use. If QUEEN is already running, "
            f"open http://127.0.0.1:{args.port}; otherwise choose another port "
            "with --port.\n",
        )
    server.game = game = LiveGame(lambda: QueenEngine(args.model_dir))
    print(f"Play QUEEN at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        game.close()


if __name__ == "__main__":
    main()
