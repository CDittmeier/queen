"""Batched vLLM generation for the stage-5.sim tree search.

K position threads each run an unmodified Builder; every model call parks on
one Batcher whose single server thread drains same-kind batches into the GPU —
the only way to fill it when the search within one position is a serial chain
of calls. Two flamingo engines (plan + verdict) live in one process:

  * they share models/vllm/flamingo.py's module-global live model, and the
    runner hook routes every batch through whichever was built last — so the
    plan model would be served with the verdict model's cross-attention. The
    batcher runs one batch at a time, so re-pointing the global before each
    batch is enough.
  * CUDA graphs stay off (enforce_eager): they skip the 16 cross-attention
    sublayers and the model emits generic prose — silent and wrong.
  * prefix caching stays off (flamingo_generate hard-sets it): every request
    sends identical prompt token ids, the board rides cross-attention that
    vLLM's block hash cannot see.

BatchedModels subclasses datagen.sim.search.Models but swaps only where a raw
generation comes from; the Builder and flatten are identical between the HF
and batched paths.
"""
from collections import deque
import threading
import time
import traceback
from pathlib import Path

from datagen.sim.search import (PLAN_PROMPT, VERDICT_PROMPT, Models)

REPO = Path(__file__).resolve().parents[2]
MERGED = REPO / "runs/stage5_sim"           # merged_plan/ + merged_verdict/
ENCODER = REPO / "data/engines/lc0_hf_bt5"


class Batcher:
    """Serves generation requests in same-kind batches from one GPU thread."""

    def __init__(self, backend, max_batch=32, wait_s=0.05):
        self.backend, self.max_batch, self.wait_s = backend, max_batch, wait_s
        self.q = {"plan": deque(), "verdict": deque()}
        self.cache, self.inflight, self.errors = {}, {}, {}
        self.demanded, self.ever_demanded = set(), set()
        self.cv = threading.Condition()
        self.stop = False
        self.batches = self.gens = 0
        self.requests = self.cache_hits = self.inflight_hits = 0
        self.prefetch_enqueued = 0
        self.busy_s = 0.0
        self.t = threading.Thread(target=self._serve, daemon=True)
        self.t.start()

    @staticmethod
    def _key(kind, fen, history=None):
        return kind, fen, tuple(history or ())

    def generate(self, kind, fen, history=None):
        history = tuple(history or ())
        key = self._key(kind, fen, history)
        with self.cv:
            self.requests += 1
            self.ever_demanded.add(key)
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
            ev = self.inflight.get(key)
            if ev is None:
                ev = threading.Event()
                self.inflight[key] = ev
                self.q[kind].append((fen, history, ev))
                self.cv.notify_all()
            else:
                self.inflight_hits += 1
            # An earlier speculative enqueue becomes required as soon as a
            # search asks for it. The server always schedules required work
            # ahead of untouched speculation.
            self.demanded.add(key)
        ev.wait()
        with self.cv:
            if key in self.errors:
                raise RuntimeError(f"batched {kind} generation failed for {fen}") \
                    from self.errors[key]
            return self.cache[key]

    def prefetch(self, kind, fens, histories=None):
        """Enqueue generations without waiting: speculative work that keeps
        the GPU busy while the search is still deciding whether it needs them
        — and a cache hit when it does. Deduped against cache and inflight."""
        with self.cv:
            fresh = False
            if histories is None:
                histories = [()] * len(fens)
            if len(fens) != len(histories):
                raise ValueError("fens and histories must have equal length")
            for fen, history in zip(fens, histories):
                history = tuple(history or ())
                key = self._key(kind, fen, history)
                if key in self.cache or key in self.inflight:
                    continue
                ev = threading.Event()
                self.inflight[key] = ev
                self.q[kind].append((fen, history, ev))
                self.prefetch_enqueued += 1
                fresh = True
            if fresh:
                self.cv.notify_all()

    def _serve(self):
        while True:
            with self.cv:
                while not self.stop and not any(self.q.values()):
                    self.cv.wait(timeout=0.05)
                if self.stop and not any(self.q.values()):
                    return
                demanded = {
                    k: sum(self._key(k, fen, history) in self.demanded
                           for fen, history, _ in q)
                    for k, q in self.q.items()
                }
                if any(demanded.values()):
                    # A short verdict batch breaks search chains quickly on a
                    # tie; otherwise favor the kind unblocking most workers.
                    kind = max(self.q, key=lambda k: (
                        demanded[k], k == "verdict", len(self.q[k])))
                else:
                    kind = max(self.q, key=lambda k: len(self.q[k]))
                if not self.q[kind]:
                    continue
                if len(self.q[kind]) < self.max_batch:
                    self.cv.wait(timeout=self.wait_s)   # linger for a fuller batch
                    demanded = {
                        k: sum(self._key(k, fen, history) in self.demanded
                               for fen, history, _ in q)
                        for k, q in self.q.items()
                    }
                    if any(demanded.values()):
                        kind = max(self.q, key=lambda k: (
                            demanded[k], k == "verdict", len(self.q[k])))
                    else:
                        kind = max(self.q, key=lambda k: len(self.q[k]))
                # Stable partition: required requests first, then speculative
                # requests of the same kind to fill unused batch capacity.
                queued = list(self.q[kind])
                self.q[kind].clear()
                queued.sort(key=lambda item: self._key(
                    kind, item[0], item[1]) not in self.demanded)
                take = min(self.max_batch, len(queued))
                items = queued[:take]
                self.q[kind].extend(queued[take:])
            t0 = time.perf_counter()
            try:
                outs = self.backend(
                    kind, [f for f, _, _ in items], [h for _, h, _ in items])
                if len(outs) != len(items):
                    raise RuntimeError(
                        f"backend returned {len(outs)} outputs for {len(items)} requests")
            except Exception as exc:
                traceback.print_exc()
                with self.cv:
                    for fen, history, ev in items:
                        key = self._key(kind, fen, history)
                        self.errors[key] = exc
                        self.demanded.discard(key)
                        ev.set()
                continue
            self.busy_s += time.perf_counter() - t0
            self.batches += 1
            self.gens += len(items)
            with self.cv:
                for (fen, history, ev), text in zip(items, outs):
                    key = self._key(kind, fen, history)
                    self.cache[key] = text
                    self.inflight.pop(key, None)
                    self.demanded.discard(key)
                    ev.set()

    def shutdown(self):
        with self.cv:
            self.stop = True
            # Every synchronous caller has returned before shutdown. Anything
            # still queued is speculative prefetch that no surviving search
            # needs; draining it can waste minutes at the end of a shard.
            for kind, q in self.q.items():
                while q:
                    fen, history, ev = q.popleft()
                    key = self._key(kind, fen, history)
                    self.inflight.pop(key, None)
                    self.demanded.discard(key)
                    ev.set()
            self.cv.notify_all()
        self.t.join(timeout=30)

    def stats(self, wall):
        return {"batches": self.batches, "generations": self.gens,
                "mean_batch": round(self.gens / max(1, self.batches), 2),
                "gpu_busy_frac": round(self.busy_s / max(1e-9, wall), 3),
                "requests": self.requests, "cache_hits": self.cache_hits,
                "inflight_hits": self.inflight_hits,
                "prefetch_enqueued": self.prefetch_enqueued,
                "speculative_generated": sum(
                    key not in self.ever_demanded for key in self.cache)}


class VLLMBackend:
    """Both models on vLLM, one engine each, in this process."""

    def __init__(self, merged_dir=None, encoder=None, gpu_mem=0.42,
                 max_num_seqs=64, use_v1_vllm=False):
        from models.vllm.flamingo_generate import ChessFlamingoGenerator
        merged = Path(merged_dir or MERGED)
        enc = str(encoder or ENCODER)
        self.g = {}
        for kind, d in (("plan", "merged_plan"), ("verdict", "merged_verdict")):
            print(f"[load] vllm {kind} ...", flush=True)
            g = ChessFlamingoGenerator(str(merged / d), enc,
                                       gpu_memory_utilization=gpu_mem,
                                       max_num_seqs=max_num_seqs,
                                       use_v1_vllm=use_v1_vllm,
                                       enforce_eager=True)
            self._patch_prompt_ids(g)
            self.g[kind] = g

    @staticmethod
    def _patch_prompt_ids(g):
        """transformers 5 returns a dict from apply_chat_template(tokenize=True);
        _prompt_ids wants the bare id list and vLLM then chokes on dict keys."""
        orig = g._prompt_ids

        def ids(prompt):
            out = orig(prompt)
            if isinstance(out, dict):
                out = out.get("input_ids", out)
            if out and isinstance(out[0], list):
                out = out[0]
            return list(out)
        g._prompt_ids = ids

    def __call__(self, kind, fens, histories=None):
        import models.vllm.flamingo as fl
        g = self.g[kind]
        fl._MODEL.clear()
        fl._MODEL.append(g.model)
        prompt = PLAN_PROMPT if kind == "plan" else VERDICT_PROMPT
        outs = g.generate(fens, [prompt] * len(fens), histories=histories,
                          temperature=0.0,
                          max_tokens=1500 if kind == "plan" else 128)
        # generate() returns {"token_ids", "text"} per request
        return [(o["text"] if isinstance(o, dict) else o).strip() for o in outs]


class BatchedModels(Models):
    """search.Models with generation routed through the Batcher; the disk
    cache (if given) is still consulted first, read-only."""

    def __init__(self, batcher, cache_path=None, prefetch=True):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path is not None and self.cache_path.exists():
            self._load_cache()
        self.b = batcher
        self.prefetch_enabled = prefetch

    def _gen(self, kind, fen, history=None):
        got = self.cache.get(self.cache_key(kind, fen, history))
        return got if got is not None else self.b.generate(kind, fen, history)

    def prefetch(self, kind, fens, histories=None):
        if self.prefetch_enabled:
            if histories is None:
                histories = [()] * len(fens)
            positions = [
                (fen, history) for fen, history in zip(fens, histories)
                if self.cache_key(kind, fen, history) not in self.cache
            ]
            self.b.prefetch(kind, [fen for fen, _ in positions],
                            [history for _, history in positions])
