"""SmolLM3 adapter — implements the DecoderAdapter protocol for HF SmolLM3."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


class SmolLM3Adapter:
    """Adapter for HF SmolLM3 decoders.

    Mirrors ``SmolLM3Model.forward`` so the Flamingo manual walk reproduces its
    behavior exactly:
      - per-layer causal mask lookup via ``layer.attention_type``
      - single shared rotary_emb returning one ``(cos, sin)`` pair for all layers
      - layer forward returns the new hidden state directly (not wrapped in a tuple)

    SmolLM3 *can* have sliding-window layers when ``config.layer_types`` contains
    ``"sliding_attention"``; for SmolLM3-3B that list is all ``"full_attention"``
    so the sliding mask isn't built. Keeping the conditional matches upstream.
    """

    model_type: str = "smollm3"
    layer_cls_name: str = "SmolLM3DecoderLayer"

    def __init__(self, decoder: nn.Module):
        self._decoder = decoder
        config = decoder.config
        self.hidden_size: int = config.hidden_size
        self.vocab_size: int = config.vocab_size
        self.n_layers: int = config.num_hidden_layers
        layer_types = getattr(config, "layer_types", [])
        self._has_sliding: bool = "sliding_attention" in layer_types

    def scale_embed(self, embed: torch.Tensor) -> torch.Tensor:
        return embed   # SmolLM3 does not scale embeddings

    def stop_token_ids(self, tokenizer) -> list[int]:
        # ChatML terminator. Filter out vocab misses (would map to unk).
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
        # Mirrors the mask dict SmolLM3Model.forward builds, keyed by
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
        # SmolLM3 decoder layer returns the new hidden state directly (no tuple).
        return layer(
            hidden,
            attention_mask=mask[layer.attention_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=pos_info,
        )
