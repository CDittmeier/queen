import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.base import (
    ChessLMConfig,
    ChessLMPreTrainedModel,
    apply_lora,
    decoder_trainable_params,
    init_new_token_embeddings,
    load_causal_decoder,
    load_decoder_state,
    save_decoder_state,
    unwrap_decoder,
)
from models.adapters import adapter_for
from models.adapters.base import DecoderAdapter, xattn_schedule


# --- FFN helpers ---

class _SimpleFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, activation: str, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)
        self.activation = activation
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "relu2":
            x = F.relu(x).square()
        elif self.activation == "gelu":
            x = F.gelu(x)
        return self.drop(self.fc2(x))


class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up   = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def _make_ffn(d_model: int, activation: str, dropout: float) -> nn.Module:
    d_hidden = 2 * d_model
    if activation == "swiglu":
        return _SwiGLUFFN(d_model, d_hidden, dropout)
    assert activation in ("relu2", "gelu"), f"Unknown activation: {activation!r}"
    return _SimpleFFN(d_model, d_hidden, activation, dropout)


# --- Flamingo-style gated cross-attention sublayer ---

class DenseXAttn(nn.Module):
    """
    Flamingo-style gated cross-attention sublayer.
    Inserts between frozen decoder layers; only this module's parameters are trained.

    Args:
        encoder_dim:  hidden dim of the chess encoder (LC0 BT5: 1024)
        decoder_dim:  hidden dim of the LLM decoder   (SmolLM3 3B: 2048, Gemma 3-4B: 2560)
        n_heads:      number of Q heads               (default: 16, full MHA)
        n_kv_heads:   number of KV heads              (default: 16, equal to n_heads → full MHA,
                      no GQA constraint; set < n_heads to enable GQA)
        activation:   FFN activation — "relu2" (Flamingo default), "gelu", "swiglu"
        dropout:      applied inside attention and FFN
        alpha_init:   initial value of alpha (pre-tanh). DEFAULT: 1.0 (tanh(1)≈0.762,
                      gate ~76% open at init — avoids cold-start trap where K/V get near-zero
                      gradient and the encoder side never learns).
                      0.0 = original Flamingo (gate fully closed at init; only works when
                      wo_rand_init=True so alpha gets gradients via random W_O).
        wo_rand_init: DEFAULT: False. By default W_O is zero-initialized so cross-attn residual
                      contribution is 0 at step 0 regardless of alpha — frozen decoder sees its
                      normal input, gate is functionally open for learning immediately. Set True
                      for the Flamingo-original combo (alpha=0, W_O random).
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 16,
        activation: str = "relu2",
        dropout: float = 0.0,
        alpha_init: float = 1.0,
        wo_rand_init: bool = False,
    ):
        super().__init__()
        assert decoder_dim % n_heads == 0, "decoder_dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = decoder_dim // n_heads
        self.n_rep      = n_heads // n_kv_heads
        self.dropout    = dropout

        self.W_Q = nn.Linear(decoder_dim,                  n_heads    * self.head_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim,                  n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim,                  n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, decoder_dim,                     bias=False)
        if not wo_rand_init:
            nn.init.zeros_(self.W_O.weight)

        self.alpha_attn = nn.Parameter(torch.tensor([alpha_init]))
        self.alpha_ffn  = nn.Parameter(torch.tensor([alpha_init]))

        self.norm_y   = nn.LayerNorm(decoder_dim)
        self.norm_x   = nn.LayerNorm(encoder_dim)
        self.norm_ffn = nn.LayerNorm(decoder_dim)

        self.ffn = _make_ffn(decoder_dim, activation, dropout)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        y : (B, S_dec, decoder_dim)  — decoder residual stream
        x : (B, S_enc, encoder_dim)  — encoder hidden states for this layer (S_enc = 64)
        returns: y updated with gated cross-attention and gated FFN, same shape as input y
        """
        B, S_dec, _ = y.shape
        S_enc = x.shape[1]

        q = self.W_Q(self.norm_y(y))
        x_n = self.norm_x(x)
        k = self.W_K(x_n)
        v = self.W_V(x_n)

        q = q.view(B, S_dec, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, S_enc, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_enc, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, S_dec, self.n_heads * self.head_dim)
        attn_out = self.W_O(attn_out)

        y = torch.tanh(self.alpha_attn).to(y.dtype) * attn_out + y
        y = torch.tanh(self.alpha_ffn).to(y.dtype) * self.ffn(self.norm_ffn(y)) + y

        return y


# --- FlamingoChessLM ---

class FlamingoChessLM(ChessLMPreTrainedModel):
    """
    Frozen LM decoder bridged to a frozen LC0 chess encoder via 16 trainable
    DenseXAttn sublayers (Flamingo-style gated cross-attention).

    Expects pre-computed, canonicalized encoder hidden states so that encoder
    inference can be batched and cached independently of the decoder.

    Decoder-family specifics (hidden_size, layer count, per-layer mask + RoPE +
    call signature) come from a DecoderAdapter passed in at construction.

    Architecture:
      - x-attn sublayer i is injected before decoder layer X_ATTN_POSITIONS[i]
      - x-attn sublayer i attends to encoder layer i (1-to-1 pairing, layers 0–15)
      - Trainable: DenseXAttn layers + new token embeddings always; plus LoRA
        adapters (lora_rank>0) or full decoder (lora_rank=0); decoder frozen (lora_rank<0)

    Inputs:
      input_ids              : (B, S)             — tokenized decoder input
      encoder_hidden_states  : (B, 16, 64, 1024)  — pre-computed, canonicalized
      attention_mask         : (B, S)             — padding mask (optional)
    """

    N_XATTN     = 16
    ENCODER_DIM = 1024
    RESPONSE_LOGPROB_CHUNK_SIZE = 128

    def __init__(
        self,
        decoder: nn.Module,
        adapter: DecoderAdapter,
        n_new_tokens: int = 0,
        lora_rank: int = -1,
        x_attn_kwargs: dict = None,
    ):
        super().__init__(ChessLMConfig())

        self.lora_rank = lora_rank
        self.decoder   = apply_lora(decoder, lora_rank)
        self.adapter   = adapter

        decoder_dim = self.adapter.hidden_size

        self.X_ATTN_POSITIONS = xattn_schedule(
            self.adapter.n_layers, n_xattn=self.N_XATTN, kind="every_other_front",
        )

        x_attn_kwargs = x_attn_kwargs or {}
        self.x_attn_layers = nn.ModuleList([
            DenseXAttn(
                encoder_dim=self.ENCODER_DIM,
                decoder_dim=decoder_dim,
                **x_attn_kwargs,
            )
            for _ in range(self.N_XATTN)
        ])

        self._xattn_at = {pos: i for i, pos in enumerate(self.X_ATTN_POSITIONS)}

        self.n_new_tokens = n_new_tokens
        dec_param = next(self.decoder.parameters())
        self.new_embed, self.new_lm_head = init_new_token_embeddings(
            n_new_tokens, decoder_dim,
            device=dec_param.device, dtype=dec_param.dtype,
        )

    @property
    def _base_decoder(self) -> nn.Module:
        return unwrap_decoder(self.decoder)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        response_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self._hidden_states(
            input_ids,
            encoder_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if response_mask is not None:
            if labels is not None:
                raise ValueError("labels and response_mask are mutually exclusive")
            return self._response_token_logprobs_from_hidden(
                h, input_ids, response_mask, attention_mask
            )
        if labels is not None:
            return self._loss_from_hidden(h, labels)
        base = self._base_decoder
        logits = base.lm_head(h)
        if self.n_new_tokens > 0:
            logits = torch.cat([logits, self.new_lm_head(h)], dim=-1)
        return logits

    def _hidden_states(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run the shared Flamingo decoder path and return final hidden states."""
        B, S = input_ids.shape
        device = input_ids.device
        base = self._base_decoder

        # Embed text tokens (split embedding). HF's base.model.embed_tokens
        # auto-scales existing-vocab tokens for Gemma (Gemma3TextScaledWordEmbedding
        # multiplies by sqrt(hidden_size) inside) and is a plain lookup for
        # SmolLM3; our new_embed lives outside that wrapper.
        frozen_vocab = self.adapter.vocab_size
        if self.n_new_tokens > 0:
            clipped = input_ids.clamp(max=frozen_vocab - 1)
            h = base.model.embed_tokens(clipped)
            new_mask = input_ids >= frozen_vocab
            # Branchless: a batch with no new tokens must still run new_embed,
            # else its FSDP gradient hook fires in a different order on that
            # rank (its grad then only flows via the tied lm_head, first in
            # backward) and the ranks' reduce-scatters desync -> NCCL hang.
            new_ids = (input_ids - frozen_vocab).clamp(min=0)
            new_h = self.adapter.scale_embed(self.new_embed(new_ids)).to(h.dtype)
            h = torch.where(new_mask.unsqueeze(-1), new_h, h)
        else:
            h = base.model.embed_tokens(input_ids)

        cache_position = torch.arange(S, device=device)
        if position_ids is None:
            # .contiguous() severs the expand view's ._base (= the unsqueezed
            # cache_position). Gemma's build_causal_mask feeds position_ids into
            # create_sliding_window_causal_mask, which otherwise exposes
            # position_ids._base.size() to dynamo as an untracked shape symbol
            # and trips a guard AssertionError on recompile.
            position_ids = cache_position.unsqueeze(0).expand(B, -1).contiguous()

        pos_info = self.adapter.build_position_info(h, position_ids)
        try:
            mask = self.adapter.build_causal_mask(
                h, attention_mask, position_ids, cache_position
            )
        except TypeError as error:
            # Transformers 5 renamed input_embeds -> inputs_embeds and removed
            # cache_position from its public mask helpers. The adapters retain
            # the Transformers 4 path used by SFT; support the newer runtime
            # here as well because RL and vLLM share that environment.
            if "input_embeds" not in str(error):
                raise
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            mask_kwargs = dict(
                config=self.decoder.config,
                inputs_embeds=h,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            mask = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(self.adapter, "_has_sliding", False):
                mask["sliding_attention"] = \
                    create_sliding_window_causal_mask(**mask_kwargs)

        for layer_idx, decoder_layer in enumerate(base.model.layers):
            xattn_idx = self._xattn_at.get(layer_idx)

            def layer_forward(
                value,
                decoder_layer=decoder_layer,
                xattn_idx=xattn_idx,
                layer_idx=layer_idx,
            ):
                if xattn_idx is not None:
                    enc = encoder_hidden_states[:, xattn_idx].to(value.dtype)
                    value = self.x_attn_layers[xattn_idx](value, enc)
                if not hasattr(decoder_layer, "attention_type"):
                    # Transformers 5 moved this selector onto config.layer_types.
                    attention_type = self.decoder.config.layer_types[layer_idx]
                    return decoder_layer(
                        value,
                        attention_mask=mask[attention_type],
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=pos_info,
                    )
                return self.adapter.apply_layer(
                    decoder_layer, value,
                    mask=mask, pos_info=pos_info,
                    position_ids=position_ids, cache_position=cache_position,
                )

            if self.training and getattr(self, "rl_activation_checkpointing", False):
                h = checkpoint(
                    layer_forward,
                    h,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                h = layer_forward(h)

        return base.model.norm(h)

    def response_token_logprobs(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        response_mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Return next-token log probabilities selected by ``response_mask``.

        The returned tensor has shape ``(B, S - 1)``: entry ``[:, i]`` is the
        log probability assigned to ``input_ids[:, i + 1]``. Entries outside
        the response mask are zero. LM-head chunks are recomputed in backward,
        bounding the otherwise dominant ``response_tokens x vocabulary``
        activation memory for long GRPO rollouts.
        """
        hidden = self._hidden_states(
            input_ids,
            encoder_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self._response_token_logprobs_from_hidden(
            hidden, input_ids, response_mask, attention_mask, chunk_size
        )

    def _response_token_logprobs_from_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if response_mask.shape != input_ids.shape:
            raise ValueError(
                f"response_mask shape {tuple(response_mask.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        selected = response_mask[:, 1:].bool()
        if attention_mask is not None:
            selected = selected & attention_mask[:, 1:].bool()
        flat_hidden = hidden[:, :-1][selected]
        flat_targets = input_ids[:, 1:][selected]
        base = self._base_decoder

        def project(selected_hidden, targets):
            logits = base.lm_head(selected_hidden)
            if self.n_new_tokens > 0:
                logits = torch.cat([logits, self.new_lm_head(selected_hidden)], dim=-1)
            return -F.cross_entropy(logits.float(), targets, reduction="none")

        pieces = []
        chunk = chunk_size or self.RESPONSE_LOGPROB_CHUNK_SIZE
        for start in range(0, flat_targets.numel(), chunk):
            hidden_chunk = flat_hidden[start:start + chunk]
            target_chunk = flat_targets[start:start + chunk]
            if torch.is_grad_enabled() and hidden_chunk.requires_grad:
                token_logps = checkpoint(
                    project,
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                token_logps = project(hidden_chunk, target_chunk)
            pieces.append(token_logps)

        flat_logps = (
            torch.cat(pieces)
            if pieces
            else hidden.new_empty(0, dtype=torch.float32)
        )
        return hidden.new_zeros(selected.shape, dtype=torch.float32).masked_scatter(
            selected, flat_logps
        )

    @property
    def device(self) -> torch.device:
        return next(self.x_attn_layers.parameters()).device

    def trainable_parameters(self):
        params = list(self.x_attn_layers.parameters())
        params += decoder_trainable_params(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            params += list(self.new_embed.parameters())
        return iter(params)

    def trainable_state_dict(self) -> dict:
        d = {"x_attn_layers": self.x_attn_layers.state_dict()}
        if self.lora_rank >= 0:
            d["decoder"] = save_decoder_state(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            d["new_embed"] = self.new_embed.state_dict()
        return d

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        self.x_attn_layers.load_state_dict(state_dict["x_attn_layers"])
        if self.lora_rank >= 0 and "decoder" in state_dict:
            load_decoder_state(self.decoder, self.lora_rank, state_dict["decoder"])
        if self.n_new_tokens > 0 and "new_embed" in state_dict:
            self.new_embed.load_state_dict(state_dict["new_embed"])

    def param_groups(self, lr: float, decoder_lr: float | None = None,
                     embed_lr: float | None = None) -> list[dict]:
        groups = [{"params": list(self.x_attn_layers.parameters()), "lr": lr}]
        dec_params = decoder_trainable_params(self.decoder, self.lora_rank)
        if dec_params:
            # lora_rank=0 unfreezes the pretrained backbone — use lr*0.1 to avoid
            # destroying it. lora_rank>0 trains fresh adapters from scratch
            # (B=0 init) — full lr is appropriate.
            # decoder_lr overrides this when set explicitly (e.g. to decouple bridge/decoder LRs).
            dec_lr = decoder_lr if decoder_lr is not None else (lr if self.lora_rank > 0 else lr * 0.1)
            groups.append({"params": dec_params, "lr": dec_lr})
        if self.n_new_tokens > 0:
            emb_lr = embed_lr if embed_lr is not None else lr * 0.1
            groups.append({"params": list(self.new_embed.parameters()), "lr": emb_lr})
        return groups

    def get_diagnostics(self) -> dict[str, float]:
        def _alpha(p):  # gather the sharded DTensor under FSDP2; plain tensor otherwise
            if hasattr(p, "full_tensor"):
                p = p.full_tensor()
            return torch.tanh(p.float()).item()
        return {
            f"alpha_attn/layer_{i:02d}": _alpha(layer.alpha_attn)
            for i, layer in enumerate(self.x_attn_layers)
        } | {
            f"alpha_ffn/layer_{i:02d}": _alpha(layer.alpha_ffn)
            for i, layer in enumerate(self.x_attn_layers)
        }

    @classmethod
    def from_pretrained(
        cls,
        decoder_path: str,
        n_new_tokens: int = 0,
        lora_rank: int = -1,
        device: torch.device | str = None,
        x_attn_kwargs: dict = None,
        frozen_vocab: int | None = None,
        **hf_kwargs,
    ):
        if device is not None:
            hf_kwargs.setdefault("device_map", device)
        decoder = load_causal_decoder(decoder_path, **hf_kwargs)
        if frozen_vocab is not None and frozen_vocab < decoder.config.vocab_size:
            # Gemma pre-allocates 64 extra embedding rows beyond the tokenizer
            # vocab. Slice them off so config.vocab_size == len(tokenizer_base)
            # and the split-embed / new_lm_head logic works without special cases.
            decoder.resize_token_embeddings(frozen_vocab)
            decoder.config.vocab_size = frozen_vocab
        adapter = adapter_for(decoder)
        model = cls(
            decoder, adapter,
            n_new_tokens=n_new_tokens, lora_rank=lora_rank, x_attn_kwargs=x_attn_kwargs,
        )
        decoder_dtype  = next(decoder.parameters()).dtype
        decoder_device = next(decoder.parameters()).device
        model.x_attn_layers.to(device=decoder_device, dtype=decoder_dtype)
        # new_embed / new_lm_head are already on decoder_device/dtype (allocated
        # there in __init__ via init_new_token_embeddings) — casting them here
        # would re-create the Parameter on each module and sever weight tying.
        return model

    @classmethod
    def from_merged_pretrained(
        cls,
        merged_path: str,
        frozen_vocab: int,
        lora_rank: int = -1,
        device: torch.device | str = None,
        x_attn_kwargs: dict = None,
        **hf_kwargs,
    ):
        """Reconstruct trainable Flamingo state from a merged vLLM checkpoint.

        A merged checkpoint stores chess-token embeddings as the decoder's
        vocabulary suffix. Training keeps that suffix in ``new_embed`` so it
        remains independently trainable when the decoder is frozen. This
        loader splits those rows back out and restores the adjacent
        ``xattn.pt`` into the normal :class:`FlamingoChessLM` representation.
        """
        from pathlib import Path
        from transformers import AutoModelForCausalLM

        decoder = AutoModelForCausalLM.from_pretrained(merged_path, **hf_kwargs)
        merged_vocab = decoder.config.vocab_size
        if not 0 < frozen_vocab < merged_vocab:
            raise ValueError(
                f"frozen_vocab={frozen_vocab} must be below merged vocab "
                f"size {merged_vocab}"
            )
        n_new_tokens = merged_vocab - frozen_vocab
        merged_new_embed = decoder.get_input_embeddings().weight[
            frozen_vocab:merged_vocab
        ].detach().clone()

        decoder.resize_token_embeddings(frozen_vocab)
        decoder.config.vocab_size = frozen_vocab
        if device is not None:
            # Loading on CPU and moving after the vocabulary split avoids a
            # mixed CPU/CUDA module: Transformers/Accelerate can leave modules
            # created by resize_token_embeddings on CPU under device_map.
            decoder.to(device)
        adapter = adapter_for(decoder)
        model = cls(
            decoder,
            adapter,
            n_new_tokens=n_new_tokens,
            lora_rank=lora_rank,
            x_attn_kwargs=x_attn_kwargs,
        )
        decoder_parameter = next(decoder.parameters())
        model.x_attn_layers.to(
            device=decoder_parameter.device,
            dtype=decoder_parameter.dtype,
        )
        with torch.no_grad():
            model.new_embed.weight.copy_(
                merged_new_embed.to(
                    device=model.new_embed.weight.device,
                    dtype=model.new_embed.weight.dtype,
                )
            )

        xattn_path = Path(merged_path) / "xattn.pt"
        if not xattn_path.is_file():
            raise FileNotFoundError(f"merged Flamingo checkpoint lacks {xattn_path}")
        model.x_attn_layers.load_state_dict(torch.load(xattn_path, map_location="cpu"))
        return model
