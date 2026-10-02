from typing import Iterator, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)


class ChessLM(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def trainable_parameters(self) -> Iterator[nn.Parameter]: ...

    def trainable_state_dict(self) -> dict: ...

    def load_trainable_state_dict(self, state_dict: dict) -> None: ...

    def get_diagnostics(self) -> dict[str, float]: ...

    def param_groups(self, lr: float, decoder_lr: float | None = None,
                     embed_lr: float | None = None) -> list[dict]: ...


# ---------------------------------------------------------------------------
# FSDP-ready base class + loss dispatch
#
# Shared base for the flamingo / llava archs. Subclassing
# PreTrainedModel is the minimal FSDP hook: exposes `_no_split_modules` for the
# auto-wrap policy and gives the model a `config`. HF weight-init is unused (the
# decoder is pretrained, bridges self-init), so `_init_weights` is a no-op.
# `_loss_from_hidden` runs a chunked LM-head / cross-entropy by default; set
# `use_liger_loss=True` to route through Liger's fused Triton kernel instead
# (needed by stage-5 to cut the extra hidden-state activation copy; the chunked
# path is preferred elsewhere because Liger's `.item()` graph-breaks torch.compile).
# ---------------------------------------------------------------------------

class ChessLMConfig(PretrainedConfig):
    model_type = "chess_lm"


class ChessLMPreTrainedModel(PreTrainedModel):
    config_class = ChessLMConfig
    base_model_prefix = "chess_lm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SmolLM3DecoderLayer", "Gemma3DecoderLayer", "Qwen3DecoderLayer", "DenseXAttn"]
    logit_chunk_size = 1024   # supervised tokens per LM-head chunk (chunked path)
    use_liger_loss = False    # opt in per run; overridden from training config

    def _init_weights(self, module):
        return  # decoder is pretrained; bridges init in __init__

    def _loss_from_hidden(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Next-token cross-entropy from final hidden states (B, S, D).

        Two paths, selected by `self.use_liger_loss`:
          * chunked (default): stream the LM head in `logit_chunk_size` slices so
            the full (B, S, V) logits are never materialized; sums fp32 CE.
          * liger: Liger's fused linear+CE Triton kernel. Saves an extra
            hidden-state copy vs. chunked but graph-breaks torch.compile at its
            internal `.item()`.
        """
        base = self._base_decoder
        hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        labels = labels[:, 1:].reshape(-1)

        # Drop non-supervised (-100) rows before either path — the chunked path
        # needs this to size its chunks correctly, and it also spares Liger's
        # fused kernel the expensive hidden @ vocab projection on rows whose
        # loss is discarded anyway.
        keep = labels != -100
        hidden = hidden[keep]
        labels = labels[keep]
        n = labels.numel()
        if n == 0:
            return hidden.sum() * 0.0  # degenerate batch: keep a grad-connected 0

        if self.use_liger_loss:
            from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
            weight = base.lm_head.weight
            if self.n_new_tokens > 0:
                weight = torch.cat([weight, self.new_lm_head.weight], dim=0)
            return LigerFusedLinearCrossEntropyLoss()(weight, hidden, labels, base.lm_head.bias)

        # Chunked path — stream the LM head so the full (N, V) logits are
        # never materialized at once.
        chunk = chunk_size or self.logit_chunk_size
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, n, chunk):
            h_c = hidden[i:i + chunk]
            logits_c = base.lm_head(h_c)
            if self.n_new_tokens > 0:
                logits_c = torch.cat([logits_c, self.new_lm_head(h_c)], dim=-1)
            total = total + F.cross_entropy(logits_c.float(), labels[i:i + chunk], reduction="sum")
        return total / n


def init_new_token_embeddings(
    n_new_tokens: int,
    decoder_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Embedding | None, nn.Linear | None]:
    """
    Returns a (new_embed, new_lm_head) pair with tied weights, or (None, None).
    new_lm_head.weight is tied to new_embed.weight (matches SmolLM3's tie_word_embeddings=True).

    device/dtype: if provided, both modules are allocated directly on that
    device/dtype. Tying is established AFTER allocation so a subsequent .to()
    cannot sever it. Callers should avoid calling .to() on these modules after
    construction — that would re-create the Parameter on one module and break
    the tie.
    """
    if n_new_tokens == 0:
        return None, None
    factory = {"device": device, "dtype": dtype}
    new_embed = nn.Embedding(n_new_tokens, decoder_dim, **factory)
    new_lm_head = nn.Linear(decoder_dim, n_new_tokens, bias=False, **factory)
    new_lm_head.weight = new_embed.weight
    return new_embed, new_lm_head


# ---------------------------------------------------------------------------
# Shared LoRA / decoder helpers
#
# lora_rank semantics used across all three architectures:
#   < 0  →  decoder fully frozen (no grad, not saved in checkpoint)
#   = 0  →  decoder fully trainable
#   > 0  →  decoder backbone frozen + LoRA adapters on Q/K/V/O
# ---------------------------------------------------------------------------

def apply_lora(
    decoder: nn.Module,
    lora_rank: int,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA / freezing to decoder according to lora_rank semantics.

    lora_rank < 0: freeze all decoder parameters; return decoder unchanged.
    lora_rank = 0: leave decoder fully trainable; return unchanged.
    lora_rank > 0: wrap with PEFT LoRA (backbone frozen, adapters trainable).
    """
    if lora_rank < 0:
        for p in decoder.parameters():
            p.requires_grad_(False)
        return decoder
    if lora_rank == 0:
        return decoder
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=lora_rank,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(decoder, cfg)


def load_causal_decoder(decoder_path: str, **hf_kwargs) -> nn.Module:
    """Load a text-only HF causal-LM decoder, dispatched by path. Gemma3ForCausalLM
    pulls just the text stack from a multimodal checkpoint. One line per new family."""
    p = decoder_path.lower()
    if "smollm" in p:
        return AutoModelForCausalLM.from_pretrained(decoder_path, **hf_kwargs)
    if "qwen" in p:
        return AutoModelForCausalLM.from_pretrained(decoder_path, **hf_kwargs)
    if "gemma" in p:
        return Gemma3ForCausalLM.from_pretrained(decoder_path, **hf_kwargs)
    raise ValueError(f"no decoder loader registered for path: {decoder_path}")


def unwrap_decoder(decoder: nn.Module) -> nn.Module:
    """Return the underlying HF CausalLM, unwrapping PEFT if present."""
    if hasattr(decoder, "get_base_model"):
        return decoder.get_base_model()
    return decoder


def decoder_trainable_params(decoder: nn.Module, lora_rank: int) -> list[nn.Parameter]:
    """Return the decoder parameters that should be optimized.

    lora_rank < 0: [] (decoder is frozen)
    lora_rank = 0: all decoder parameters
    lora_rank > 0: only requires_grad=True params (the LoRA adapters)
    """
    if lora_rank < 0:
        return []
    if lora_rank > 0:
        return [p for p in decoder.parameters() if p.requires_grad]
    return list(decoder.parameters())


def save_decoder_state(decoder: nn.Module, lora_rank: int) -> dict:
    """Serialize trainable decoder state. Only call when lora_rank >= 0."""
    if lora_rank > 0:
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(decoder)
    return decoder.state_dict()


def load_decoder_state(decoder: nn.Module, lora_rank: int, state: dict) -> None:
    """Load decoder state produced by save_decoder_state."""
    if lora_rank > 0:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(decoder, state)
    else:
        decoder.load_state_dict(state)
