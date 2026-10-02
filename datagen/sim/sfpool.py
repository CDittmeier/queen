"""A bounded pool of Stockfish engines, shared by all workers.

Also shares the result cache across workers. The old per-instance cache meant
two positions that reached the same FEN each paid for it; the tree revisits
positions constantly (deepening re-searches, transpositions, the oracle
walks), so this is a large saving on its own.

    from datagen.sim.sfpool import install
    install(judges=4)                    # before any engine is constructed
"""
import queue
import threading
from contextlib import contextmanager


class Pool:
    """Hand out engines; block when they are all busy."""

    def __init__(self, factory, size):
        self.q = queue.Queue()
        self.made = 0
        self.size = size
        self.factory = factory
        self.lock = threading.Lock()

    @contextmanager
    def borrow(self):
        try:
            e = self.q.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.made < self.size:
                    self.made += 1
                    e = self.factory()          # spawn lazily, up to `size`
                else:
                    e = None
            if e is None:
                e = self.q.get()                # all busy: wait for one
        try:
            yield e
        finally:
            self.q.put(e)


def install(judges=8):
    """Route every SfJudge through one bounded pool and a shared cache.

    An engine can run any requested node budget because ``go nodes N`` is sent
    for each position.  It also serves both primary-only (MultiPV=1) checks and
    full MultiPV=2 judgements, switching that UCI option when borrowed. A
    cached result at a higher node budget satisfies a lower-budget request.
    MultiPV=2 results also satisfy primary-only requests, but not vice versa.
    """
    from datagen.tree import search as S

    orig_init = S.SfJudge.__init__
    orig_search2 = S.SfJudge.search2
    cache = {}
    cache_lock = threading.Lock()

    def raw():
        obj = S.SfJudge.__new__(S.SfJudge)
        orig_init(obj, nodes=1000, multipv=2)
        return obj

    pool = Pool(raw, judges)

    def judge_init(self, nodes=100_000, multipv=2, **kw):
        self.nodes = int(nodes)
        self.multipv = int(multipv)
        if self.multipv not in (1, 2):
            raise ValueError("SfJudge multipv must be 1 or 2")

    def judge_search2(self, fen):
        # Each capability key retains only its highest-node result. Thus a
        # 10k insert replaces a 1k primary result, while a primary-only insert
        # cannot erase the second-line information needed by MultiPV=2.
        key = (self.multipv, fen)
        with cache_lock:
            row = cache.get(key)
            if row is not None and row[0] >= self.nodes:
                return row[1]
        with pool.borrow() as e:
            e.nodes = self.nodes
            if e.multipv != self.multipv:
                e.send(f"setoption name MultiPV value {self.multipv}")
                e.multipv = self.multipv
            got = orig_search2(e, fen)
        capabilities = (1, 2) if self.multipv == 2 else (1,)
        with cache_lock:
            for capability in capabilities:
                ckey = (capability, fen)
                old = cache.get(ckey)
                if old is None or old[0] <= self.nodes:
                    cache[ckey] = (self.nodes, got)
            return cache[key][1]

    S.SfJudge.__init__ = judge_init
    S.SfJudge.search2 = judge_search2
    return {"judge_cache": cache, "judge_pool": pool}
