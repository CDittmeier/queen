"""vLLM model for the stage-4/5 FLAMINGO checkpoint: SmolLM3 + 16 gated cross-attn
sublayers injected into the decoder stack.

Unlike the llava path, the board is NOT a prompt prefix. models/flamingo.py injects a
DenseXAttn sublayer before decoder layers 0, 2, ..., 30, each attending to its own lc0
encoder layer, on EVERY forward -- prefill and every decode step alike. So there is no
prompt_embeds-shaped hole to put the board into, and the served decoder is not a stock
SmolLM3. Three things make it work here:

  1. The 16 DenseXAttn sublayers are re-created and hooked in front of the same decoder
     layers via forward pre-hooks, so vLLM's paged self-attention is untouched.
  2. Cross-attn K/V depend only on the encoder states, so they are computed ONCE per
     request at prefill and parked in a slot-indexed GPU buffer. Decode steps only read.
  3. Routing: vLLM flattens scheduled tokens into one sequence. The V2 prepare_inputs
     or legacy V1 _prepare_inputs wrapper publishes each token's board-cache slot.
     V2 slots persist per request; V1 may move requests during batch compaction,
     in which case their destination slots are refilled before the next forward.

Encoder states reach this module through a plain module-global registry, which requires
the engine to run IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0; set for you by
models/vllm/flamingo_generate.py). With the default multiprocess EngineCore the model
lives in another process and the registry would be empty.

Only touches models/vllm/*: the training-time models/flamingo.py is untouched and the
llava serving path is untouched.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm import ModelRegistry
from vllm.model_executor.models.transformers import TransformersForCausalLM

N_XATTN = 16          # must match models/flamingo.py FlamingoChessLM.N_XATTN
N_ENC_SQUARES = 64    # lc0 board tokens per encoder layer
_ARCH = "SmolLM3ForCausalLM"
_IMPL = "models.vllm.flamingo:ChessFlamingoSmolLM3ForCausalLM"


def xattn_positions(n_layers: int, n_xattn: int = N_XATTN) -> list[int]:
    """Mirror of models/adapters/base.py::xattn_schedule('every_other_front')."""
    assert 2 * n_xattn <= n_layers, f"cannot fit {n_xattn} x-attns in {n_layers} layers"
    return [2 * i for i in range(n_xattn)]


# --------------------------------------------------------------------------- state
# Filled by the generator before submitting requests; read by the runner hook.

_ENC_STATES: dict[str, torch.Tensor] = {}   # req_id -> (16, 64, encoder_dim), on GPU
_ROUTE: dict = {"active": False, "uniq": None, "segments": None}  # live forward's routing
_MODEL: list = []                           # live-model compatibility handle for callers


def set_encoder_states(req_id: str, enc: torch.Tensor) -> None:
    """Register one request's encoder hidden states, (16, 64, encoder_dim)."""
    assert enc.dim() == 3 and enc.shape[0] == N_XATTN and enc.shape[1] == N_ENC_SQUARES, \
        f"expected (16, 64, D), got {tuple(enc.shape)}"
    _ENC_STATES[req_id] = enc


def drop_encoder_states(req_ids) -> None:
    for r in req_ids:
        _ENC_STATES.pop(r, None)


# --------------------------------------------------------------------------- xattn

class DenseXAttnInfer(nn.Module):
    """Inference-only port of models/flamingo.py::DenseXAttn.

    Same parameters and same math, but K/V come from a per-token gather of the cached
    encoder projections instead of a (B, S_enc, D) tensor, because vLLM hands us a flat
    token stream whose rows belong to different requests.
    """

    def __init__(self, encoder_dim, decoder_dim, n_heads=16, n_kv_heads=16):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = decoder_dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.W_Q = nn.Linear(decoder_dim, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, decoder_dim, bias=False)
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn = nn.Parameter(torch.zeros(1))
        self.norm_y = nn.LayerNorm(decoder_dim)
        self.norm_x = nn.LayerNorm(encoder_dim)
        self.norm_ffn = nn.LayerNorm(decoder_dim)
        # activation "relu2" (the training default): fc1 -> relu^2 -> fc2, d_hidden = 2*d,
        # both bias-free (models/flamingo.py::_SimpleFFN).
        self.ffn = nn.Module()
        self.ffn.fc1 = nn.Linear(decoder_dim, 2 * decoder_dim, bias=False)
        self.ffn.fc2 = nn.Linear(2 * decoder_dim, decoder_dim, bias=False)

    def encode_kv(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """enc (R, 64, encoder_dim) -> K, V each (R, 64, n_kv_heads*head_dim)."""
        x = self.norm_x(enc)
        return self.W_K(x), self.W_V(x)

    def _ffn(self, h):
        return self.ffn.fc2(F.relu(self.ffn.fc1(h)).square())

    def forward(self, y: torch.Tensor, k_tok: torch.Tensor, v_tok: torch.Tensor,
                segments) -> torch.Tensor:
        """y (T, decoder_dim); k_tok/v_tok are either per-token (T, 64, D_kv) gathers or,
        when `segments` is given, per-request (R, 64, D_kv) with (start, length) runs."""
        T = y.shape[0]
        q = self.W_Q(self.norm_y(y)).view(T, self.n_heads, self.head_dim)

        if segments is None:
            # Decode: every row is its own request -> one batched SDPA, q_len 1.
            k = k_tok.view(T, N_ENC_SQUARES, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v_tok.view(T, N_ENC_SQUARES, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            out = F.scaled_dot_product_attention(q.unsqueeze(2), k, v)   # (T, H, 1, hd)
            attn = out.squeeze(2).reshape(T, self.n_heads * self.head_dim)
        else:
            # Prefill/mixed: rows are contiguous per request, so attend per run and avoid
            # materializing a (T, 64, D_kv) gather (2 GB at an 8k chunked prefill).
            attn = y.new_empty((T, self.n_heads * self.head_dim))
            for r, (start, length) in enumerate(segments):
                qs = q[start:start + length].transpose(0, 1).unsqueeze(0)     # (1,H,L,hd)
                k = k_tok[r].view(N_ENC_SQUARES, self.n_kv_heads, self.head_dim) \
                            .transpose(0, 1).unsqueeze(0)
                v = v_tok[r].view(N_ENC_SQUARES, self.n_kv_heads, self.head_dim) \
                            .transpose(0, 1).unsqueeze(0)
                if self.n_rep > 1:
                    k = k.repeat_interleave(self.n_rep, dim=1)
                    v = v.repeat_interleave(self.n_rep, dim=1)
                o = F.scaled_dot_product_attention(qs, k, v)                  # (1,H,L,hd)
                attn[start:start + length] = o.squeeze(0).transpose(0, 1) \
                                              .reshape(length, self.n_heads * self.head_dim)

        y = torch.tanh(self.alpha_attn).to(y.dtype) * self.W_O(attn) + y
        return torch.tanh(self.alpha_ffn).to(y.dtype) * self._ffn(self.norm_ffn(y)) + y


# --------------------------------------------------------------------------- model

class ChessFlamingoSmolLM3ForCausalLM(TransformersForCausalLM):
    """SmolLM3 with the 16 trained cross-attention sublayers hooked back in."""

    def __init__(self, *, vllm_config, prefix: str = "", **kw):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kw)
        cfg = vllm_config.model_config.hf_config
        self.decoder_dim = cfg.hidden_size
        self.max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.dtype = vllm_config.model_config.dtype
        self._device = vllm_config.device_config.device
        self._xattn_ready = False
        self.x_attn_layers = None
        _MODEL.clear(); _MODEL.append(self)
        _install_runner_hook(vllm_config.use_v2_model_runner)
        # Attach before vLLM startup profiling so its memory budget includes these
        # weights and caches. The generator requires eager execution.
        xattn_path = Path(vllm_config.model_config.model) / "xattn.pt"
        if xattn_path.exists():
            self.attach_xattn(torch.load(xattn_path, map_location="cpu"))

    def load_weights(self, weights):
        """Declare the x-attn parameters as loaded.

        They come from xattn.pt in __init__, not from the merged safetensors, but the
        default loader strict-checks state_dict() against whatever load_weights reports
        and would otherwise abort with "weights were not initialized from checkpoint"."""
        loaded = super().load_weights(weights)
        return loaded | {n for n, _ in self.named_parameters()
                         if n.startswith("x_attn_layers.")}

    # -- layer discovery -----------------------------------------------------
    def _decoder_layers(self):
        m = self.model
        if hasattr(m, "layers"):
            return m.layers
        return m.language_model.layers      # some Transformers-backend layouts nest it

    # -- setup ---------------------------------------------------------------
    def attach_xattn(self, state_dict, encoder_dim=1024):
        """Build the 16 sublayers from a flamingo checkpoint's xattn.pt and hook them
        before decoder layers 0, 2, ..., 30 (same schedule as training)."""
        layers = self._decoder_layers()
        positions = xattn_positions(len(layers))
        self.x_attn_layers = nn.ModuleList([
            DenseXAttnInfer(encoder_dim, self.decoder_dim) for _ in range(N_XATTN)
        ]).to(device=self._device, dtype=self.dtype)
        missing, unexpected = self.x_attn_layers.load_state_dict(state_dict, strict=False)
        assert not unexpected, f"unexpected x-attn keys: {list(unexpected)[:6]}"
        assert not missing, f"missing x-attn keys: {list(missing)[:6]}"

        self.encoder_dim = encoder_dim
        kv_dim = self.x_attn_layers[0].n_kv_heads * self.x_attn_layers[0].head_dim
        dev = self._device
        # Slot-indexed cross-attn K/V, written once per request at prefill.
        self.k_cache = torch.zeros((self.max_reqs, N_XATTN, N_ENC_SQUARES, kv_dim),
                                   device=dev, dtype=self.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.slot_owner: list = [None] * self.max_reqs
        # Persistent routing buffer. A replayed CUDA graph does not re-run the hook, so
        # the captured kernels must read a fixed address that _prepare_inputs refreshes
        # in place. Zero-filled so graph-padding rows index slot 0 instead of garbage.
        self.slot_buf = torch.zeros(self.max_tokens, dtype=torch.long, device=dev)

        for xi, pos in enumerate(positions):
            layers[pos].register_forward_pre_hook(self._make_hook(xi), with_kwargs=True)
        self._xattn_ready = True

    def fill_slot(self, slot: int, req_id: str) -> None:
        """Project this request's encoder states into the K/V cache at `slot`."""
        if req_id not in _ENC_STATES:
            raise KeyError(
                f"request {req_id!r} reached the flamingo model with no board registered. "
                "Every request needs set_encoder_states(req_id, ...) before submission — "
                "see models/vllm/flamingo_generate.py.")
        enc = _ENC_STATES[req_id].to(self.k_cache.device, self.dtype)
        for xi, layer in enumerate(self.x_attn_layers):
            k, v = layer.encode_kv(enc[xi].unsqueeze(0))
            self.k_cache[slot, xi] = k[0]
            self.v_cache[slot, xi] = v[0]
        self.slot_owner[slot] = req_id

    # -- the hook ------------------------------------------------------------
    def _make_hook(self, xattn_idx: int):
        def hook(module, args, kwargs):
            if not self._xattn_ready or not _ROUTE["active"]:
                return None                              # profiling / dummy runs
            if args:
                h, rest = args[0], args[1:]
            else:
                h, rest = kwargs["hidden_states"], ()
            squeeze = h.dim() == 3                       # vLLM feeds (1, T, D)
            y = h[0] if squeeze else h

            segments, uniq = _ROUTE["segments"], _ROUTE["uniq"]
            layer = self.x_attn_layers[xattn_idx]
            if segments is None:
                slot = self.slot_buf[:y.shape[0]]     # fixed address -> graph-safe
                k_tok = self.k_cache[slot, xattn_idx]
                v_tok = self.v_cache[slot, xattn_idx]
            else:
                k_tok = self.k_cache[uniq, xattn_idx]
                v_tok = self.v_cache[uniq, xattn_idx]
            y = layer(y, k_tok, v_tok, segments)

            y = y.unsqueeze(0) if squeeze else y
            if args:
                return (y, *rest), kwargs
            kwargs["hidden_states"] = y
            return args, kwargs
        return hook


# --------------------------------------------------------------------------- routing

_HOOK_INSTALLED = set()
_WARMING_V2 = False


def _route_requests(model, req_ids, slots_np, counts) -> None:
    """Publish board K/V and token routing in the runner's actual request order."""
    for request, slot in zip(req_ids, slots_np):
        if model.slot_owner[slot] != request:
            model.fill_slot(int(slot), request)
    dev = model.k_cache.device
    per_token = torch.from_numpy(np.repeat(slots_np, counts))
    model.slot_buf[:per_token.numel()].copy_(per_token.to(dev), non_blocking=True)
    _ROUTE["active"] = True
    _ROUTE["uniq"] = torch.from_numpy(slots_np).to(dev, torch.long)
    _ROUTE["segments"] = (None if (counts == 1).all()
                          else list(zip(np.cumsum(counts) - counts, counts)))


def _install_runner_hook(use_v2: bool) -> None:
    """Adapt each runner's prepared inputs without changing cross-attention math."""
    if use_v2 in _HOOK_INSTALLED:
        return
    if use_v2:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker import gpu_worker
        original_warmup = gpu_worker.warmup_kernels

        def warmup(*args, **kwargs):
            # V2 sends synthetic startup requests through the real input-preparation
            # path. Only this explicit scope may run without registered boards.
            global _WARMING_V2
            _WARMING_V2 = True
            try:
                return original_warmup(*args, **kwargs)
            finally:
                _WARMING_V2 = False
                _ROUTE["active"] = False

        gpu_worker.warmup_kernels = warmup
        method = "prepare_inputs"
    else:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        method = "_prepare_inputs"
    original = getattr(GPUModelRunner, method)

    def patched(self, scheduler_output, *args, **kwargs):
        out = original(self, scheduler_output, *args, **kwargs)
        model = self.model
        if _WARMING_V2 or not getattr(model, "_xattn_ready", False):
            _ROUTE["active"] = False
            return out
        if use_v2:
            # V2 reorders decode/prefill rows; batch indices are not persistent slots.
            req_ids = out.req_ids
            slots_np = out.idx_mapping_np.astype(np.int64, copy=False)
            counts = out.num_scheduled_tokens.astype(np.int64, copy=False)
            if out.num_tokens_after_padding != out.num_tokens:
                raise RuntimeError("Flamingo V2 routing requires unpadded eager inputs")
        else:
            req_ids = self.input_batch.req_ids
            slots_np = np.array([self.input_batch.req_id_to_index[r] for r in req_ids],
                                dtype=np.int64)
            counts = np.array([scheduler_output.num_scheduled_tokens[r] for r in req_ids],
                              dtype=np.int64)
        if not req_ids:
            _ROUTE["active"] = False
            return out
        _route_requests(model, req_ids, slots_np, counts)
        return out

    setattr(GPUModelRunner, method, patched)
    _HOOK_INSTALLED.add(use_v2)


def register() -> None:
    """Route vLLM's SmolLM3 architecture to the flamingo subclass.

    Registered by import-path string, matching models/vllm/llava.py. NOTE: llava.py and
    this module both claim SmolLM3ForCausalLM, so a process must import only one of them.
    """
    ModelRegistry.register_model(_ARCH, _IMPL)


register()
