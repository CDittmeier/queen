"""Gemma 3 adapter — implements the DecoderAdapter protocol for HF Gemma 3."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


class Gemma3Adapter:
    """Adapter for HF Gemma 3 (text-only) decoders.

    Mirrors ``Gemma3TextModel.forward`` so the Flamingo manual walk reproduces
    its behavior exactly:
      - dual RoPE: ``rotary_emb`` (global, θ=1M) and ``rotary_emb_local``
        (sliding, θ=10K). build_position_info bundles both.
      - mask dict always contains both ``full_attention`` and
        ``sliding_attention`` entries; apply_layer looks up per
        ``layer.attention_type``.
      - the Gemma decoder layer takes BOTH ``position_embeddings_global`` and
        ``position_embeddings_local`` and selects internally via
        ``self_attn.is_sliding``.
      - the layer's forward returns a tuple; hidden state is element 0.

    embed_scale matches ``Gemma3TextScaledWordEmbedding``'s internal
    ``embed * sqrt(hidden_size)`` — applied to our own ``new_embed`` output so
    new tokens arrive at the decoder with the same magnitude as existing
    vocab tokens.
    """

    model_type: str = "gemma3_text"
    layer_cls_name: str = "Gemma3DecoderLayer"

    def __init__(self, decoder: nn.Module):
        # Resolve to the inner Gemma3TextModel regardless of wrapper:
        #   - Gemma3ForConditionalGeneration: .language_model is the TextModel directly
        #     (via @property → self.model.language_model)
        #   - Gemma3ForCausalLM (text-only):   .model is the TextModel
        #   - Gemma3TextModel:                  already the TextModel
        if hasattr(decoder, "language_model"):
            self._text_model = decoder.language_model
        elif hasattr(decoder, "model") and decoder.model.__class__.__name__ == "Gemma3TextModel":
            self._text_model = decoder.model
        else:
            self._text_model = decoder
        config = self._text_model.config
        # Hard-fail on the vision-text bidirectional-attention path — needs an
        # or_mask_function overlay we haven't wired up.
        assert not getattr(config, "use_bidirectional_attention", False), (
            "Gemma3Adapter does not support use_bidirectional_attention=True yet."
        )
        self.hidden_size: int = config.hidden_size
        self.vocab_size: int = config.vocab_size
        self.n_layers: int = config.num_hidden_layers
        self._scale: float = math.sqrt(config.hidden_size)

    def scale_embed(self, embed: torch.Tensor) -> torch.Tensor:
        return embed * self._scale

    def stop_token_ids(self, tokenizer) -> list[int]:
        # Gemma chat template terminates each turn with <end_of_turn>.
        tid = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if not isinstance(tid, int) or tid < 0 or tid == tokenizer.unk_token_id:
            return []
        return [tid]

    def build_position_info(
        self, hidden: torch.Tensor, position_ids: torch.Tensor
    ) -> dict[str, Any]:
        # Dual RoPE — both are reused across every layer. The HF Gemma 3 layer
        # takes both kwargs and picks via self_attn.is_sliding.
        return {
            "global": self._text_model.rotary_emb(hidden, position_ids),
            "local":  self._text_model.rotary_emb_local(hidden, position_ids),
        }

    def build_causal_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, Any]:
        # Gemma 3 always builds both masks. apply_layer looks up by
        # layer.attention_type, the same key shape SmolLM3 produces.
        kwargs = dict(
            config=self._text_model.config,
            input_embeds=hidden,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        return {
            "full_attention":    create_causal_mask(**kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**kwargs),
        }

    def apply_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        *,
        mask: dict[str, Any],
        pos_info: dict[str, Any],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        # Gemma 3 decoder layer returns a tuple (hidden_states, ...).
        layer_out = layer(
            hidden,
            position_embeddings_global=pos_info["global"],
            position_embeddings_local=pos_info["local"],
            attention_mask=mask[layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )
        return layer_out[0]
