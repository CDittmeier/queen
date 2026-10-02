"""DecoderAdapter protocol — single point of contact between bridge models
(LLaVA / Flamingo) and the underlying HF causal-LM decoder.

Each decoder family (SmolLM3, Gemma 3, ...) implements one DecoderAdapter that
hides per-family differences in:

  * new-token embedding scaling — Gemma multiplies by sqrt(hidden_size) inside
    its embed_tokens wrapper; SmolLM3 does not. Our self.new_embed lives outside
    that wrapper, so the bridge has to apply the right scaling explicitly.
  * position embeddings — Gemma 3 has dual RoPE (one theta for global layers,
    another for sliding-window layers); SmolLM3 has a single shared rotary_emb.
  * causal masks — Gemma 3's sliding-window layers need a windowed mask, global
    layers need the full causal mask; SmolLM3 uses one mask for every layer.
  * decoder-layer call signature — Gemma 3's layer kwargs differ from SmolLM3's.
  * chat-template stop tokens — SmolLM3 uses ChatML's ``<|im_end|>``, Gemma uses
    ``<end_of_turn>``.

Design principle: every method on the adapter hides a real per-family
difference. Operations that are uniform across the decoder families we target
(layer iteration via ``base.model.layers``, the model-final ``base.model.norm``,
``base.model.embed_tokens`` for existing-vocab tokens, ``base.lm_head``) stay
as direct calls from the bridge — wrapping zero-difference operations adds
indirection without payoff.

LLaVA consumes only the attributes + ``scale_embed`` (its forward routes
through HF's full ``base.model(...)`` so it never touches the per-layer walk).
Flamingo consumes the full surface because its manual layer walk has to
interleave cross-attn injection between decoder blocks.

Concrete implementations live in models/adapters/{smollm,gemma}.py.
models/adapters/__init__.py exports ``adapter_for(decoder_or_config)`` which
dispatches to the right class by ``config.model_type``.
"""
from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn


class DecoderAdapter(Protocol):
    """Per-family decoder interface used by FlamingoChessLM / LLaVAChessLM.

    ``pos_info`` and ``mask`` are adapter-opaque values returned from
    ``build_position_info`` / ``build_causal_mask`` — only the same adapter's
    ``apply_layer`` consumes them, so their concrete shape is a private
    contract between those methods (e.g. a tuple for SmolLM3, a dict for
    Gemma 3).
    """

    # --- Attributes -----------------------------------------------------------
    hidden_size: int       # decoder hidden dim; sizes bridge / new_embed / file/rank embeds
    vocab_size: int        # frozen-vocab cutoff for split embedding
    n_layers: int          # decoder layer count; drives xattn schedule
    model_type: str        # "smollm3" / "gemma3_text"; for stage-2 compatibility guard
    layer_cls_name: str    # HF decoder-layer class name; for FSDP transformer_cls_names_to_wrap

    # --- New-token embedding scaling (LLaVA + Flamingo) -----------------------
    def scale_embed(self, embed: torch.Tensor) -> torch.Tensor:
        """Apply decoder-family-specific scaling to a new-token embedding tensor.

        SmolLM3: identity. Gemma 3: ``embed * sqrt(hidden_size)``, matching the
        scaling Gemma3TextScaledWordEmbedding applies inside the HF
        embed_tokens wrapper for existing-vocab tokens.
        """
        ...

    # --- Eval utility ---------------------------------------------------------
    def stop_token_ids(self, tokenizer) -> list[int]:
        """Token IDs that terminate assistant generation under this model's
        chat template (SmolLM3 ChatML's ``<|im_end|>``, Gemma's
        ``<end_of_turn>``). The eval generation loop short-circuits when any
        of these is emitted.
        """
        ...

    # --- Flamingo manual-walk primitives --------------------------------------
    def build_position_info(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> Any:
        """Precompute position-dependent info reused across all decoder layers.

        SmolLM3: returns the ``(cos, sin)`` tuple from the single shared
        ``base.model.rotary_emb``.
        Gemma 3: returns ``{"global": (cos,sin), "local": (cos,sin)}`` because
        sliding-window layers use a different RoPE theta than global layers.
        """
        ...

    def build_causal_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Any:
        """Build the causal mask(s) needed by the decoder's attention.

        SmolLM3: returns one full causal mask used by every layer.
        Gemma 3: returns ``{"global": full, "sliding": windowed}`` so
        ``apply_layer`` can pass both to the layer's forward — HF's Gemma 3
        decoder layer selects internally on ``self.self_attn.is_sliding``.

        ``position_ids`` is forwarded to the upstream mask helper to match
        HF's own ``Model.forward`` invocation exactly (used for
        packed-sequence detection and some edge cases).
        """
        ...

    def apply_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        *,
        mask: Any,
        pos_info: Any,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke one decoder layer's forward with the right kwargs.

        Each family's HF decoder-layer module has a different signature; this
        method is where that lives. ``mask`` and ``pos_info`` are the values
        returned by ``build_causal_mask`` / ``build_position_info``, opaque to
        the bridge model.
        """
        ...


# ---------------------------------------------------------------------------
# Shared helpers (not part of the protocol)
# ---------------------------------------------------------------------------

def xattn_schedule(
    n_layers: int,
    n_xattn: int = 16,
    kind: str = "every_other_front",
) -> list[int]:
    """Compute the decoder-layer positions where Flamingo injects cross-attn.

    "every_other_front" — inject before layers ``[0, 2, ..., 2*(n_xattn-1)]``,
    leaving the trailing ``n_layers - 2*n_xattn`` layers without injection.
    Reproduces the original SmolLM3 schedule (``range(0, 32, 2)`` over 36
    layers); on Gemma 3-4B (34 layers) it leaves layers 32-33 without
    injection.
    """
    if kind != "every_other_front":
        raise NotImplementedError(f"Unknown xattn schedule kind: {kind!r}")
    if 2 * n_xattn > n_layers:
        raise ValueError(
            f"Cannot fit {n_xattn} x-attns at every-other in {n_layers} layers "
            f"(need at least {2 * n_xattn})"
        )
    return [2 * i for i in range(n_xattn)]
