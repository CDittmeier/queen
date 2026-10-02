"""Stage-5.sim search: alpha-beta over the plan model's candidates.

The tree framework is datagen/tree's (Tree nodes, HCE reading order); only the
search differs from the HCE pipeline:

  * candidates at EVERY node = first moves of the side-to-move's validated
    plan bullets (datagen.sim.plan_parse — blunder-screened, legality-
    truncated, subset-dropped), with the clearly-best move forced in
    (datagen.sim.force) so the model dropping the critical move can neither
    invent nor hide a refutation;
  * leaves = the verdict model's sentence mapped to centipawns (band
    midpoints), checkmates/draws scored directly;
  * iterative deepening from scratch per depth (generations are fen-cached so
    re-searches are free); `union=True` reports the UNION of every iteration's
    tree — a deeper iteration spends the node budget depth-first, so a
    candidate explored at depth 2 can be cut off before it is reached at
    depth 3; the union keeps it — re-valued by plain negamax so the search
    values agree with what is verbalized.

Models hold the two HF checkpoints plus a position-context-keyed jsonl cache of raw
generations; CachedModels re-renders from the cache alone (no GPU — a miss is
a hard error). The batched vLLM path substitutes only where a raw generation
comes from (datagen.sim.sampler).
"""
import json
import os
import random
import zlib
from pathlib import Path
from types import SimpleNamespace

import chess

from datagen.sim.plan_parse import plan_entries
from datagen.tree.moves import move_priority
from datagen.tree.tree import Tree
from utils.translate_helpers import Translator

PLAN_PROMPT = "What are the plans for each side in this chess position?"
VERDICT_PROMPT = "What is the evaluation of the current position, and why?"
FORCE_NODES = int(os.environ.get("MT_FORCE_NODES", 100_000))
MATE = 300000

# verdict phrase -> pawns for the searcher (midpoints of the SF-convention bands)
_PHRASES = [("much better-to-winning", 3.25), ("better-to-much better", 1.25),
            ("slightly better-to-better", 0.45), ("much better", 2.0),
            ("slightly better", 0.2), ("winning", 5.0), ("better", 0.8)]


def verdict_value(board, verdict_text):
    """Verdict sentence (decoded) -> centipawns from the side-to-move's POV."""
    import re
    low = verdict_text.lower()
    val = 0.0
    for phrase, pawns in _PHRASES:
        if phrase in low:
            val = pawns
            break
    if "winning" in low and "much better-to-winning" not in low:
        val = max(val, 5.0)
    if val:
        m = re.match(r"\s*(white|black)", low)
        if m is None:
            return 0
        sgn = 1 if (m.group(1) == "white") == (board.turn == chess.WHITE) else -1
        return int(sgn * val * 100)
    return 0


def _load_hf(config, ckpt):
    import torch
    import yaml

    from utils.training_utils import init_model_and_tokenizer
    cfg = yaml.safe_load(open(config))
    args = SimpleNamespace(
        arch=cfg["arch"], lora_rank=cfg.get("lora_rank", 0),
        decoder_path=cfg["decoder_path"], encoder_path=cfg["encoder_path"],
        dtype=cfg.get("dtype", "bfloat16"), device="cuda",
        embed_init=cfg.get("embed_init", "semantic"),
        alpha_init=1.0, wo_rand_init=False,
        train_dataset=cfg["train_dataset"], pov=True)
    model, encoder, tokenizer = init_model_and_tokenizer(args)
    sd = torch.load(Path(ckpt) / "trainable.pt", map_location="cpu")
    sd = {k.removeprefix("_orig_mod.").replace("._checkpoint_wrapped_module", ""): v
          for k, v in sd.items()}
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return model, encoder, tokenizer


class Models:
    """Both models + a position-context-keyed jsonl cache of raw generations.

    `plan` / `verdict` are (config_yaml, ckpt_dir) pairs; `cache_path` is the
    jsonl the raw generations are read from and appended to."""

    @staticmethod
    def cache_key(kind, fen, history=None):
        return kind, fen, tuple(history or ())

    def __init__(self, plan, verdict, cache_path):
        self.cache_path = Path(cache_path)
        self._load_cache()
        self._fh = open(self.cache_path, "a")
        print("[load] plan model ...", flush=True)
        self.plan = _load_hf(*plan)
        print("[load] verdict model ...", flush=True)
        self.verdict = _load_hf(*verdict)

    def _load_cache(self):
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path):
                r = json.loads(line)
                self.cache[self.cache_key(
                    r["kind"], r["fen"], r.get("history"))] = r["text"]

    def _gen(self, kind, fen, history=None):
        history = list(history or ())
        key = self.cache_key(kind, fen, history)
        if key in self.cache:
            return self.cache[key]
        import torch
        from datasets import Dataset

        from utils.eval_utils import run_eval
        model, encoder, tokenizer = self.plan if kind == "plan" else self.verdict
        prompt = PLAN_PROMPT if kind == "plan" else VERDICT_PROMPT
        ds = Dataset.from_list([{"fen": fen, "history": history, "prompt": prompt,
                                 "response": "", "extra": {"task": kind}}])
        _, samples = run_eval(model, encoder, ds, tokenizer,
                              torch.device("cuda"), torch.bfloat16,
                              eval_batch_size=1,
                              eval_max_new_tokens=1500 if kind == "plan" else 128,
                              pov=True, temperature=0.0)
        text = samples[0]["generated"].strip()
        record = {"kind": kind, "fen": fen, "text": text}
        if history:
            record["history"] = history
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        self.cache[key] = text
        return text

    def raw(self, kind, fen, history=None):
        """The token-space generation in the narrative's own format."""
        raw = self._gen(kind, fen, history)
        # The deployed 5.a checkpoint predates the final terse convention;
        # current checkpoints already emit "drawish", making this a no-op.
        return raw.replace("dry drawish", "drawish") if kind == "verdict" else raw

    def plans_text(self, fen, history=None):
        """Decoded (absolute White/Black) plan output for the position."""
        return Translator(chess.Board(fen).turn).decode_absolute(
            self.raw("plan", fen, history))

    def verdict_text(self, fen, history=None):
        return Translator(chess.Board(fen).turn).decode_absolute(
            self.raw("verdict", fen, history))


class CachedModels(Models):
    """Re-render a tree whose generations are already cached: no GPU, no model
    load, so it runs on a login node. A miss is a hard error rather than a
    silent fallback -- if it happens the position genuinely needs a GPU run."""

    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self._load_cache()

    def _gen(self, kind, fen, history=None):
        got = self.cache.get(self.cache_key(kind, fen, history))
        assert got is not None, (
            f"no cached {kind} generation for {fen} with the requested history; "
            f"run this position "
            f"through the GPU path first")
        return got


class Builder:
    def __init__(self, models, depth=3, max_nodes=40, cap=4, union=False,
                 force_key=False, seed=7, premise_first=True):
        self.M = models
        self.depth = depth
        self.max_nodes = max_nodes
        self.cap = cap
        self.union = union
        self.force_key = force_key
        self.seed = seed
        self.premise_first = premise_first
        self.entries = {}      # node.id -> the plan entries the search used
        self.histories = {}    # node.id -> prior FENs, oldest to newest
        self.forced = {}       # fen -> the forced line, where forcing fired
        self._ecache = {}      # fen -> entries (deterministic per fen: seeded
                               # forcing rng, cached generations), so union
                               # re-searches skip the re-parse and SF re-screens

    def build(self, root_fen, history=None):
        self.root_history = tuple(history or ())
        return (self._build_union if self.union else self._build_last)(root_fen)

    def history(self, t):
        return self.histories.get(t.id, ())

    def _paths(self, t, path=()):
        yield path
        for c in t.sf_children:
            yield from self._paths(c, path + (c.move_uci,))

    def _build_union(self, root_fen):
        """Same deepening, but the reported tree is the UNION of every
        iteration's tree. A deeper iteration spends the node budget depth-first,
        so a candidate that was explored at depth 2 can be cut off before it is
        reached at depth 3; the union keeps it instead of dropping it."""
        seen, prev = set(), 0
        for d in range(1, self.depth + 1):
            root = self._iterate(root_fen, d)
            seen |= set(self._paths(root))
            self.built_depth = d
            if self.n_nodes >= self.max_nodes or self.n_nodes == prev:
                break
            prev = self.n_nodes
        return self._materialize(root_fen, seen)

    def _materialize(self, root_fen, paths):
        """One tree holding every path any iteration reached, re-evaluated:
        leaves take the verdict model's value, internal nodes negamax over their
        union children (so the search values agree with what is verbalized)."""
        self.n_nodes, self.next_id = 0, 0
        self.plans, self.verdicts, self.entries, self.histories = {}, {}, {}, {}
        nodes = {(): self._node(None, None, root_fen, 0, self.root_history)}
        for path in sorted(paths, key=lambda p: (len(p), p)):
            if not path:
                continue
            parent = nodes[path[:-1]]
            b = chess.Board(parent.fen)
            b.push(chess.Move.from_uci(path[-1]))
            nodes[path] = self._node(
                parent, path[-1], b.fen(), parent.ply + 1,
                self.history(parent) + (parent.fen,))
            parent.sf_children.append(nodes[path])
        self.n_nodes = len(nodes)
        root = nodes[()]
        self._revalue(root)
        self._order(root)
        self._pv(root)
        return root

    def _revalue(self, t):
        board = chess.Board(t.fen)
        if not t.sf_children:
            return self._leaf_eval(t, board)
        plans = self.M.plans_text(t.fen, self.history(t))
        self.plans[t.id] = plans
        self.entries[t.id] = self._entries(t, plans)
        t.value = max(-self._revalue(c) for c in t.sf_children)
        return t.value

    def _iterate(self, root_fen, d):
        """One deepening iteration, from scratch, to depth `d`."""
        self.n_nodes = 1
        self.next_id = 0
        self.plans = {}     # node.id -> decoded plan text (internal nodes)
        self.verdicts = {}  # node.id -> decoded verdict sentence (leaves)
        self.histories = {}
        root = self._node(None, None, root_fen, 0, self.root_history)
        self._expand(root, d, -MATE * 2, MATE * 2)
        return root

    def _build_last(self, root_fen):
        """Iterative deepening: re-search at depth 1, 2, ... so the node
        budget is spent evenly across candidates instead of on one deep
        line. Generations are fen-cached, so re-searches are free. Stops at
        the iteration that hits the node cap (kept, SF-style) or when the
        tree stops growing."""
        root, prev_nodes = None, 0
        for d in range(1, self.depth + 1):
            root = self._iterate(root_fen, d)
            self.built_depth = d
            if self.n_nodes >= self.max_nodes or self.n_nodes == prev_nodes:
                break
            prev_nodes = self.n_nodes
        self._order(root)
        self._pv(root)
        return root

    def _judge(self):
        if getattr(self, "_j", None) is None:
            from datagen.tree.search import SfJudge
            self._j = SfJudge(nodes=FORCE_NODES)
        return self._j

    def _entries(self, t, plans):
        """The node's plan entries, with the key variation forced in.

        Applied at EVERY node, not just the root: the plan model drops the
        critical move at interior nodes too, and when it does the node either
        invents a refutation (every candidate it offered loses, so the move into
        it looks like a mistake) or hides one (the refuting reply is not a
        candidate, so a blunder survives as the critical line)."""
        context = (t.fen, self.history(t))
        if context in self._ecache:
            return self._ecache[context]
        entries = plan_entries(t.fen, plans, premise_first=self.premise_first,
                               raw_text=self.M.raw(
                                   "plan", t.fen, self.history(t)))
        if self.force_key:
            from datagen.sim.force import apply
            # seed from the fen with a stable hash: str hashing is salted per
            # process, and the injected plan's position in the list must not
            # depend on which process rendered the tree
            seed = zlib.crc32(t.fen.encode()) ^ self.seed
            entries, line = apply(t.fen, entries, self._judge(),
                                  random.Random(seed), self.cap)
            if line:
                self.forced[t.fen] = line
        self._ecache[context] = entries
        return entries

    def _node(self, parent, move_uci, fen, ply, history):
        t = Tree(id=self.next_id, parent=parent, move_uci=move_uci, fen=fen,
                 stm=("w" if chess.Board(fen).turn else "b"), value=0, seval=0,
                 flag="EXACT", search_leaf=False, rdepth=0, ply=ply)
        t.root_fen = parent.root_fen if parent else fen
        t.root_stm = parent.root_stm if parent else t.stm
        self.histories[t.id] = tuple(history)
        self.next_id += 1
        return t

    @staticmethod
    def _position_key(fen):
        board = chess.Board(fen)
        return " ".join(board.fen(en_passant="legal").split()[:4])

    def _can_claim_repetition(self, t, board):
        """Exact threefold claim check using the real/search-path FEN history.

        A claim is valid if the current position has occurred three times, or
        if the side to move has a legal move whose result has already occurred
        twice.  python-chess cannot recover this from a standalone FEN.
        """
        from collections import Counter

        counts = Counter(self._position_key(fen) for fen in self.history(t))
        counts[self._position_key(t.fen)] += 1
        if counts[self._position_key(t.fen)] >= 3:
            return True
        for move in board.legal_moves:
            child = board.copy(stack=False)
            child.push(move)
            if counts[self._position_key(child.fen())] >= 2:
                return True
        return False

    def _can_claim_draw(self, t, board):
        return board.can_claim_fifty_moves() or self._can_claim_repetition(t, board)

    def _leaf_eval(self, t, board):
        if board.is_checkmate():
            t.value = -MATE
            self.verdicts[t.id] = "checkmate."
        elif (board.is_stalemate() or board.is_insufficient_material()
              or self._can_claim_draw(t, board)):
            t.value = 0
            self.verdicts[t.id] = "draw."
        else:
            v = self.M.verdict_text(t.fen, self.history(t))
            self.verdicts[t.id] = v
            t.value = verdict_value(board, v)
        t.judge = t.value          # judge eval, node-POV (for _supersession)
        t.seval = t.value
        t.search_leaf = True
        return t.value

    def _expand(self, t, depth, alpha, beta):
        """Negamax with alpha-beta; returns the node value from t.stm's POV."""
        board = chess.Board(t.fen)
        if depth == 0 or board.is_game_over() or self._can_claim_draw(t, board):
            return self._leaf_eval(t, board)
        if self.n_nodes >= self.max_nodes:      # node cap: stop expanding
            return self._leaf_eval(t, board)
        plans = self.M.plans_text(t.fen, self.history(t))
        # candidates = first moves of the FILTERED plans (blunder-screened,
        # subset-dropped), so no node budget goes to truncated lines
        entries = self._entries(t, plans)
        cands, seen = [], set()
        for e in entries:
            mv = e["first"]
            if mv is not None and mv not in seen:
                seen.add(mv)
                cands.append(mv)
        cands = cands[:self.cap]
        if not cands:
            return self._leaf_eval(t, board)
        self.plans[t.id] = plans
        self.entries[t.id] = entries
        if hasattr(self.M, "prefetch"):
            # speculative: queue the children's generations now, so the GPU
            # works while this thread walks the first child's serial chain.
            # A cut-off child's generation is wasted GPU; a kept one is a hit.
            child_fens, child_histories = [], []
            for mv in cands:
                bb = board.copy(stack=False)
                bb.push(mv)
                child_fens.append(bb.fen())
                child_histories.append(self.history(t) + (t.fen,))
            self.M.prefetch("verdict" if depth == 1 else "plan", child_fens,
                            child_histories)
        best = -MATE * 2
        for mv in cands:
            if self.n_nodes >= self.max_nodes and t.sf_children:
                break                            # cap reached mid-node
            b2 = board.copy(stack=False)
            b2.push(mv)
            child = self._node(t, mv.uci(), b2.fen(), t.ply + 1,
                               self.history(t) + (t.fen,))
            self.n_nodes += 1
            t.sf_children.append(child)
            v = -self._expand(child, depth - 1, -beta, -alpha)
            if v > best:
                best = v
            if v > alpha:
                alpha = v
            if alpha >= beta:                   # fail-high: cut remaining plans
                t.flag = "LOWER"
                break
        t.value = best
        return best

    def _order(self, t):
        """our_children = HCE reading order (checks > captures > threats > others)."""
        board = chess.Board(t.fen)
        t.our_children = sorted(
            t.sf_children,
            key=lambda c: move_priority(board, chess.Move.from_uci(c.move_uci)))
        for c in t.sf_children:
            self._order(c)

    def _pv(self, root):
        pv, n = [], root
        while n.our_children:
            n = max(n.our_children, key=lambda c: -c.value)
            pv.append(n)
        root.pv = pv
