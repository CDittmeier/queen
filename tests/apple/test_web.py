import http.client
import json
import threading

import pytest

from mac_inference.game import LiveGame
from mac_inference.web import make_server


@pytest.fixture
def server():
    game = LiveGame(lambda: object())
    server = make_server(game, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    game.close()


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    data = json.dumps(body) if body is not None else None
    connection.request(method, path, data, headers or {})
    response = connection.getresponse()
    status, content, response_headers = (
        response.status,
        response.read(),
        dict(response.getheaders()),
    )
    connection.close()
    return status, content, response_headers


def test_serves_the_board_assets_and_initial_game_state(server):
    status, body, headers = request(server, "GET", "/")
    assert status == 200 and b"QUEEN\xe2\x80\x99s explanation" in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert request(server, "GET", "/app.js")[0] == 200
    assert b"white-knight" in request(server, "GET", "/pieces/wN.svg")[1]
    state = json.loads(request(server, "GET", "/api/state")[1])
    assert state["human"] == "white" and len(state["legal_moves"]) == 20


def test_rejects_cross_origin_and_rebinding_hosts(server):
    headers = {"Content-Type": "application/json", "Origin": "https://other.example"}
    assert request(server, "POST", "/api/new", {"color": "white"}, headers)[0] == 403
    assert (
        request(server, "GET", "/api/state", headers={"Host": "other.example"})[0]
        == 403
    )


def test_rejects_non_json_and_non_object_bodies(server):
    assert request(server, "POST", "/api/new", {"color": "white"})[0] == 415
    assert (
        request(server, "POST", "/api/new", [], {"Content-Type": "application/json"})[0]
        == 400
    )


def test_illegal_move_is_rejected_at_http_boundary_without_changing_board(server):
    state = json.loads(request(server, "GET", "/api/state")[1])
    body = {"game_id": state["game_id"], "version": state["version"], "uci": "e2e5"}
    assert (
        request(
            server, "POST", "/api/move", body, {"Content-Type": "application/json"}
        )[0]
        == 400
    )
    current = json.loads(request(server, "GET", "/api/state")[1])
    assert current["fen"] == state["fen"] and current["moves"] == []


def test_static_requests_cannot_read_repository_files(server):
    assert request(server, "GET", "/../pyproject.toml")[0] == 404
    assert request(server, "GET", "/.models/queen_pawn-8/config.json")[0] == 404
