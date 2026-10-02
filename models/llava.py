import torch
import torch.nn as nn

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
from models.adapters.base import DecoderAdapter


class LLaVAChessLM(ChessLMPreTrainedModel):
    """
    Frozen LC0 encoder + frozen/finetuned LM decoder bridged via a LLaVA-style MLP connector.

    All 16 encoder hidden states are channel-concatenated per square, projected to
    decoder dim via a 2-layer MLP, and prepended as 64 prefix tokens to the decoder
    input sequence. The decoder runs self-attention over prefix + text jointly.

    Learned 2D spatial embeddings (file + rank) are added to prefix tokens after the
    MLP projection to bake in board structure without modifying the decoder's RoPE.

    Decoder-family specifics (hidden_size, new-token embed scaling) come from a
    DecoderAdapter passed in at construction.

    lora_rank semantics: <0 = frozen decoder, 0 = full fine-tuning (default), >0 = LoRA adapters.

    Inputs:
      input_ids              : (B, S)             — tokenized decoder input
      encoder_hidden_states  : (B, 16, 64, 1024)  — pre-computed, canonicalized
      attention_mask         : (B, S)             — padding mask (optional)
      position_ids           : (B, S)             — 0-based TEXT positions
                                                    (model offsets by N_ENC_SQUARES
                                                    internally so text lives at
                                                    64..63+S and prefix at 0).
                                                    Default: arange(0, S).

    Output logits cover only the S text positions (prefix positions are discarded).
    """

    N_ENC_LAYERS  = 16
    ENCODER_DIM   = 1024
    N_ENC_SQUARES = 64
    N_FILES       = 8
    N_RANKS       = 8

    def __init__(
        self,
        decoder: nn.Module,
        adapter: DecoderAdapter,
        n_new_tokens: int = 0,
        lora_rank: int = 0,
    ):
        super().__init__(ChessLMConfig())

        self.lora_rank = lora_rank
        self.decoder = apply_lora(decoder, lora_rank)
        self.adapter = adapter

        decoder_dim = self.adapter.hidden_size
        mlp_hidden  = 2 * decoder_dim

        in_dim = self.N_ENC_LAYERS * self.ENCODER_DIM   # 16384
        self.connector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mlp_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_hidden, decoder_dim, bias=False),
        )

        self.file_embed = nn.Embedding(self.N_FILES, decoder_dim)
        self.rank_embed = nn.Embedding(self.N_RANKS, decoder_dim)

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
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device
        base = self._base_decoder

        # 1. Project encoder hidden states to 64 prefix token embeddings.
        enc = torch.cat(
            [encoder_hidden_states[:, e] for e in range(self.N_ENC_LAYERS)], dim=-1
        ).to(next(self.connector.parameters()).dtype)              # (B, 64, 16384)
        prefix = self.connector(enc)                               # (B, 64, hidden)

        # Add learned 2D spatial embeddings.
        sq = torch.arange(self.N_ENC_SQUARES, device=device)
        prefix = prefix + self.file_embed(sq % self.N_FILES) + self.rank_embed(sq // self.N_FILES)

        # 2. Embed text tokens (split embedding). HF's base.model.embed_tokens
        #    auto-scales existing-vocab tokens for Gemma (via Gemma3TextScaledWordEmbedding)
        #    and is a plain lookup for SmolLM3; our new_embed lives outside that
        #    wrapper so we apply the matching scale via adapter.scale_embed.
        frozen_vocab = self.adapter.vocab_size
        if self.n_new_tokens > 0:
            clipped = input_ids.clamp(max=frozen_vocab - 1)
            text_embeds = base.model.embed_tokens(clipped)
            new_mask = input_ids >= frozen_vocab
            if new_mask.any():
                new_ids = (input_ids[new_mask] - frozen_vocab).clamp(min=0)
                text_embeds[new_mask] = self.adapter.scale_embed(
                    self.new_embed(new_ids)
                ).to(text_embeds.dtype)
        else:
            text_embeds = base.model.embed_tokens(input_ids)      # (B, S, hidden)

        # 3. Prepend prefix to text embeddings.
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)   # (B, 64+S, hidden)

        # 4. Extend attention mask to cover prefix positions.
        if attention_mask is not None:
            prefix_mask = torch.ones(B, self.N_ENC_SQUARES, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 5. Position IDs: prefix tokens all at position 0 (RoPE identity: cos=1, sin=0),
        #    text tokens at 64–63+S. Assigning position 0 to all prefix tokens disables
        #    relative positional encoding between them via RoPE; spatial structure is
        #    carried entirely by the learned file/rank embeddings.
        # If position_ids is supplied by the caller (e.g. _batched_decode's cumsum trick for
        # left-padding), it covers text tokens only with 0-based indexing; offset by N_ENC_SQUARES.
        prefix_pos = torch.zeros(B, self.N_ENC_SQUARES, dtype=torch.long, device=device)
        if position_ids is None:
            text_pos = torch.arange(self.N_ENC_SQUARES, self.N_ENC_SQUARES + S, device=device).unsqueeze(0).expand(B, -1)
        else:
            text_pos = position_ids + self.N_ENC_SQUARES
        position_ids = torch.cat([prefix_pos, text_pos], dim=1)            # (B, 64+S)

        # 6. Run decoder over prefix + text; keep only text output positions.
        #    LLaVA delegates the full decoder forward to HF — base.model(...)
        #    handles all per-family internals (sliding-window attention, dual RoPE,
        #    dual norm, ...).
        model_out = base.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        h_out = model_out.last_hidden_state[:, self.N_ENC_SQUARES:, :]  # (B, S, hidden)

        if labels is not None:
            return self._loss_from_hidden(h_out, labels)
        logits = base.lm_head(h_out)
        if self.n_new_tokens > 0:
            logits = torch.cat([logits, self.new_lm_head(h_out)], dim=-1)
        return logits

    @property
    def device(self) -> torch.device:
        return next(self.connector.parameters()).device

    def trainable_parameters(self):
        return iter(
            list(self.connector.parameters()) +
            list(self.file_embed.parameters()) +
            list(self.rank_embed.parameters()) +
            decoder_trainable_params(self.decoder, self.lora_rank) +
            (list(self.new_embed.parameters()) if self.n_new_tokens > 0 else [])
        )

    def trainable_state_dict(self) -> dict:
        d = {
            "connector":  self.connector.state_dict(),
            "file_embed": self.file_embed.state_dict(),
            "rank_embed": self.rank_embed.state_dict(),
        }
        if self.lora_rank >= 0:
            d["decoder"] = save_decoder_state(self.decoder, self.lora_rank)
        if self.n_new_tokens > 0:
            d["new_embed"] = self.new_embed.state_dict()
        return d

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        self.connector.load_state_dict(state_dict["connector"])
        self.file_embed.load_state_dict(state_dict["file_embed"])
        self.rank_embed.load_state_dict(state_dict["rank_embed"])
        if self.lora_rank >= 0 and "decoder" in state_dict:
            load_decoder_state(self.decoder, self.lora_rank, state_dict["decoder"])
        if self.n_new_tokens > 0 and "new_embed" in state_dict:
            self.new_embed.load_state_dict(state_dict["new_embed"])

    def param_groups(self, lr: float, decoder_lr: float | None = None,
                     embed_lr: float | None = None) -> list[dict]:
        bridge = (
            list(self.connector.parameters()) +
            list(self.file_embed.parameters()) +
            list(self.rank_embed.parameters())
        )
        groups = [{"params": bridge, "lr": lr}]
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
        return {}

    @classmethod
    def from_pretrained(
        cls,
        decoder_path: str,
        n_new_tokens: int = 0,
        lora_rank: int = 0,
        device: torch.device | str = None,
        frozen_vocab: int | None = None,
        **hf_kwargs,
    ):
        if device is not None:
            hf_kwargs.setdefault("device_map", device)
        decoder = load_causal_decoder(decoder_path, **hf_kwargs)
        if frozen_vocab is not None and frozen_vocab < decoder.config.vocab_size:
            decoder.resize_token_embeddings(frozen_vocab)
            decoder.config.vocab_size = frozen_vocab
        adapter = adapter_for(decoder)
        model = cls(decoder, adapter, n_new_tokens=n_new_tokens, lora_rank=lora_rank)
        decoder_dtype  = next(decoder.parameters()).dtype
        decoder_device = next(decoder.parameters()).device
        model.connector.to(device=decoder_device, dtype=decoder_dtype)
        model.file_embed.to(device=decoder_device, dtype=decoder_dtype)
        model.rank_embed.to(device=decoder_device, dtype=decoder_dtype)
        # new_embed / new_lm_head are already on decoder_device/dtype (allocated
        # there in __init__ via init_new_token_embeddings) — casting them here
        # would re-create the Parameter on each module and sever weight tying.
        return model
