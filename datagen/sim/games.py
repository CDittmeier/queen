"""Parallel engine-pool game simulation from a position.

Both players of every game are drawn independently at random (repeats allowed)
from the pool: maia-2200 (1 node); maia3 conditioned at 2300/2400/2500/2600
(1 node); every lc0 net whose networks.json Elo is above 2200 (at its
configured node count); Stockfish at 100 / 1000 / 10000 nodes. Games are
capped at MAX_PLIES (counted as a draw); threefold / fifty-move draws are
claimed.

play_games(fen, n_games, workers) runs one position's games across a process
pool and returns compact dicts {"white", "black", "result", "moves": [uci]} —
the shape the simulate sampler stores as position metadata and
datagen.sim.plans.plans_from_games consumes. Engines live in a small
per-process LRU cache (SELFPLAY_CACHE, default 2): an unbounded cache across
workers exhausts GPU memory, a slightly larger one avoids costly maia3
respawns. Env knobs: LC0_BIN, MAIA3_SERVER (socket path of a shared batching
server), MAIA3_DEVICE (default cuda), SELFPLAY_POOL=legacy / weak / strong /
weak+strong (or lc0-1node for the old debug pool).
"""
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess

REPO = Path(__file__).resolve().parents[2]
LC0 = Path(os.environ.get("LC0_BIN") or REPO / "data/lc0/lc0-src/lc0")
NETS = REPO / "data/engines/lc0_networks"
MAIA3_UCI = REPO / "data/engines/maia3/.venv/bin/maia3-uci"
CUDALIBS = REPO / "data/engines/cudalibs"
MAX_PLIES = 300
N_GAMES = 100

_POOLS = {}
_CACHE = {}


def _network_config():
    with open(NETS / "networks.json") as f:
        return json.load(f)


def legacy_pool():
    """The original stage-5.sim random engine pool."""
    pool = {"maia-2200": ("lc0", NETS / "maia-2200.pb.gz", 1)}
    for elo in (2300, 2400, 2500, 2600):
        pool[f"maia3-{elo}"] = ("maia3", elo, 1)
    cfg = _network_config()
    for net, c in cfg.items():
        if c["elo"] > 2200 and not net.startswith("maia"):
            pool[net.replace(".pb.gz", "")] = ("lc0", NETS / net,
                                               c["config"]["nodes"])
    for n in (100, 1000, 10000):
        pool[f"stockfish-{n}"] = ("sf", None, n)
    return pool


def weak_pool():
    """All installed classical Maia nets and Maia3 at 600..2600 Elo.

    Maia3 is rating-conditioned: these names share one model checkpoint and
    differ only in the Elo passed to the UCI process.
    """
    pool = {}
    for net, c in _network_config().items():
        if net.startswith("maia-"):
            pool[net.removesuffix(".pb.gz")] = (
                "lc0", NETS / net, c["config"]["nodes"])
    for elo in range(600, 2601, 100):
        pool[f"maia3-{elo}"] = ("maia3", elo, 1)
    return pool


def strong_pool():
    """Every installed non-Maia LC0 net plus Stockfish at >=100 nodes."""
    pool = {}
    for net, c in _network_config().items():
        if not net.startswith("maia-"):
            pool[net.removesuffix(".pb.gz")] = (
                "lc0", NETS / net, c["config"]["nodes"])
    for n in (100, 1000, 10000):
        pool[f"stockfish-{n}"] = ("sf", None, n)
    return pool


def engine_pool(mode=None):
    """{name: (kind, arg, nodes)} for the selected named pool.

    Pools are cached independently so callers can construct pairings from
    several pools in one process. Workers normally select their pool through
    SELFPLAY_POOL.
    """
    mode = mode or os.environ.get("SELFPLAY_POOL", "legacy")
    if mode in _POOLS:
        return _POOLS[mode]
    if mode == "legacy":
        pool = legacy_pool()
    elif mode == "weak":
        pool = weak_pool()
    elif mode == "strong":
        pool = strong_pool()
    elif mode == "weak+strong":
        pool = weak_pool() | strong_pool()
    elif mode == "lc0-1node":
        pool = {n: (k, a, 1) for n, (k, a, _) in legacy_pool().items()
                if k == "lc0"}
    else:
        raise ValueError(f"unknown SELFPLAY_POOL={mode!r}")
    _POOLS[mode] = pool
    return pool


class Uci:
    def __init__(self, cmd, env=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True,
                                  bufsize=1, env=env)
        self.send("uci")
        for ln in self.p.stdout:
            if ln.startswith("uciok"):
                break

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def move(self, fen, moves, nodes):
        pos = f"position fen {fen}"
        if moves:
            pos += " moves " + " ".join(moves)
        self.send(pos)
        self.send(f"go nodes {nodes}")
        for ln in self.p.stdout:
            if ln.startswith("bestmove"):
                tok = ln.split()
                return tok[1] if len(tok) > 1 and tok[1] != "(none)" else None
        return None


class SockEngine:
    """maia3 via a shared batching server instead of a per-worker UCI
    subprocess. Same .move() interface; .p.kill() closes."""

    def __init__(self, path, elo):
        import socket
        self.elo = elo
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.f = self.s.makefile("r")
        self.n = 0
        self.p = self

    def kill(self):
        try:
            self.s.close()
        except OSError:
            pass

    def move(self, fen, moves, nodes):
        self.n += 1
        self.s.sendall((json.dumps({"id": self.n, "fen": fen, "moves": moves,
                                    "elo": self.elo}) + "\n").encode())
        ln = self.f.readline()
        return json.loads(ln)["move"] if ln else None


def get_engine(name):
    if name in _CACHE:
        e = _CACHE.pop(name)      # re-insert: LRU order
        _CACHE[name] = e
        return e
    kind, arg, _ = engine_pool()[name]
    if kind == "lc0":
        # lc0's RPATH pins /usr/local/cuda (broken cublas), so preload the
        # known-good cudalibs; the cuda-auto backend then picks the GPU
        env = dict(os.environ,
                   LD_PRELOAD=f"{CUDALIBS}/libcublasLt.so.13 "
                              f"{CUDALIBS}/libcublas.so.13 "
                              f"{CUDALIBS}/libcudart.so.13")
        e = Uci([str(LC0), f"--weights={arg}"], env=env)
    elif kind == "sf":
        from datagen.tree.search import SF_BIN
        e = Uci([str(SF_BIN)])
    elif os.environ.get("MAIA3_SERVER"):
        e = SockEngine(os.environ["MAIA3_SERVER"], arg)
    else:
        env = dict(os.environ, HF_HOME=str(REPO / "data/engines/maia3/hf_cache"))
        e = Uci([str(MAIA3_UCI), "--model", "maia3-5m",
                 "--device", os.environ.get("MAIA3_DEVICE", "cuda"),
                 "--no-use-amp", "--use-uci-history", "--elo", str(arg)],
                env=env)
    _CACHE[name] = e
    return e


def play_one(args):
    """(fen, white, black) -> game dict. Runs in a worker process."""
    fen, white, black = args
    pool = engine_pool()
    get_engine(white)
    get_engine(black)
    cap = max(2, int(os.environ.get("SELFPLAY_CACHE", "2")))
    while len(_CACHE) > cap:
        old = next(n for n in _CACHE if n not in (white, black))
        try:
            _CACHE.pop(old).p.kill()
        except Exception:
            pass
    board = chess.Board(fen)
    moves = []
    while not board.is_game_over(claim_draw=True) and len(moves) < MAX_PLIES:
        name = white if board.turn else black
        mv = get_engine(name).move(fen, moves, pool[name][2])
        if mv is None:
            break
        board.push(chess.Move.from_uci(mv))
        moves.append(mv)
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"
    return {"white": white, "black": black, "result": result, "moves": moves}


def pairings(n_games=N_GAMES, seed=0):
    """The n random engine pairings for one position (seeded, so a position's
    games are reproducible given the pool)."""
    rng = random.Random(seed)
    names = sorted(engine_pool())
    return [(rng.choice(names), rng.choice(names)) for _ in range(n_games)]


def play_games(fen, n_games=N_GAMES, workers=8, seed=0, executor=None):
    """All of one position's games, across a process pool. Pass `executor` to
    reuse one pool (and its engine caches) across many positions."""
    args = [(fen, w, b) for w, b in pairings(n_games, seed)]
    if executor is not None:
        return list(executor.map(play_one, args))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(play_one, args))
