"""Qwen3 adapter — implements the DecoderAdapter protocol for HF Qwen3.

Qwen3 (dense, e.g. qwen3-4b) is architecturally the *SmolLM3-shaped* member of
our decoder zoo, not the Gemma-shaped one. Concretely, vs. Gemma3:

  * single shared RoPE (``model.rotary_emb``) → one ``(cos, sin)`` reused by every
    layer. There is no ``rotary_emb_local`` / dual-theta split, so
    ``build_position_info`` returns a bare tuple, not a {"global","local"} dict.
  * no sliding-window attention in the dense 4B (``config.sliding_window`` is
    null and ``config.layer_types`` is all ``"full_attention"``), so
    ``build_causal_mask`` only builds the full causal mask. We keep the same
    ``_has_sliding`` conditional SmolLM3 uses so a future Qwen config that *does*
    declare sliding layers still gets the right mask dict.
  * no embedding scaling — Qwen3 has a plain ``nn.Embedding`` (no
    ``ScaledWordEmbedding`` wrapper), so ``scale_embed`` is the identity. Our
    ``new_embed`` and the frozen ``embed_tokens`` therefore live at the same
    magnitude with no correction (unlike Gemma's ×sqrt(hidden_size)).
  * ``Qwen3DecoderLayer.forward`` returns the new hidden state *directly* (a
    tensor), like SmolLM3 — not a ``(hidden, ...)`` tuple like Gemma3, so
    ``apply_layer`` returns the call result as-is.

The chat-template terminator is ChatML's ``<|im_end|>`` — the *same literal* as
SmolLM3 (Qwen ships a ChatML template in its tokenizer_config), so
``stop_token_ids`` mirrors SmolLM3Adapter exactly.

Net: this class is a near-twin of SmolLM3Adapter. It is kept as its own file (one
adapter per decoder family) so each adapter reads as an independent mirror of its
HF ``Model.forward`` rather than a shared abstraction whose correctness couples
two families together.

Qwen-specific notes that do NOT live here (handled generically elsewhere):
  * Qwen3-4B pads its embedding table to ``config.vocab_size = 151936`` while the
    tokenizer holds only ``len(tokenizer) = 151669`` real tokens. The 267 extra
    rows are sliced off by the ``frozen_vocab`` path in
    ``{Flamingo,LLaVA}ChessLM.from_pretrained`` (driven by
    ``training_utils`` passing ``frozen_vocab=len(tokenizer)``). By the time this
    adapter reads ``config.vocab_size`` it is already the sliced value, so
    ``self.vocab_size`` is the correct split-embedding cutoff.
  * ``tie_word_embeddings=True`` (shared with SmolLM3 and Gemma) — the resize done
    by the frozen_vocab slice resizes the tied lm_head too; nothing for the
    adapter to do.
  * QK-norm (RMSNorm on q/k inside Qwen3 attention) is internal to the decoder
    layer and invisible to the manual walk.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


class Qwen3Adapter:
    """Adapter for HF Qwen3 (dense) decoders.

    Mirrors ``Qwen3Model.forward`` so the Flamingo manual walk reproduces its
    behavior exactly:
      - single shared ``rotary_emb`` returning one ``(cos, sin)`` for all layers
      - per-layer causal-mask lookup keyed by ``layer.attention_type``
        (``config.layer_types[i]``; all ``"full_attention"`` for the 4B dense)
      - the decoder layer returns the new hidden state directly (not a tuple)

    Like SmolLM3, the sliding mask is only built when ``config.layer_types``
    actually declares ``"sliding_attention"`` — false for qwen3-4b, but the
    conditional matches upstream and stays correct if a future Qwen variant
    enables it.
    """

    model_type: str = "qwen3"
    layer_cls_name: str = "Qwen3DecoderLayer"

    def __init__(self, decoder: nn.Module):
        self._decoder = decoder
        config = decoder.config
        self.hidden_size: int = config.hidden_size
        self.vocab_size: int = config.vocab_size  # already frozen_vocab-sliced upstream
        self.n_layers: int = config.num_hidden_layers
        # Qwen3Config populates ``layer_types`` (all "full_attention" for the
        # dense 4B) even though it's absent from config.json. getattr keeps us
        # safe if a future config omits it.
        layer_types = getattr(config, "layer_types", [])
        self._has_sliding: bool = "sliding_attention" in layer_types

    def scale_embed(self, embed: torch.Tensor) -> torch.Tensor:
        return embed   # Qwen3 has no embedding scaling (plain nn.Embedding)

    def stop_token_ids(self, tokenizer) -> list[int]:
        # Qwen ships a ChatML template; each turn terminates with <|im_end|>
        # (same literal as SmolLM3). Filter vocab misses (would map to unk).
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(tid, int) or tid < 0 or tid == tokenizer.unk_token_id:
            return []
        return [tid]

    def build_position_info(
        self, hidden: torch.Tensor, position_ids: torch.Tensor
    ) -> Any:
        # Single shared rotary_emb → (cos, sin) tuple, reused by every layer.
        return self._decoder.model.rotary_emb(hidden, position_ids)

    def build_causal_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, Any]:
        # Mirrors the mask dict Qwen3Model.forward builds, keyed by
        # ``layer.attention_type``. apply_layer looks up the right entry.
        kwargs = dict(
            config=self._decoder.config,
            input_embeds=hidden,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        masks: dict[str, Any] = {"full_attention": create_causal_mask(**kwargs)}
        if self._has_sliding:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
        return masks

    def apply_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        *,
        mask: dict[str, Any],
        pos_info: Any,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        # Qwen3 decoder layer returns the new hidden state directly (no tuple).
        return layer(
            hidden,
            attention_mask=mask[layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=pos_info,
        )
