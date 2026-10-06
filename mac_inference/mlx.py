import json
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_lm.generate import stream_generate
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.smollm3 import Model, ModelArgs
from mlx_lm.sample_utils import apply_top_k, apply_top_p


class CrossAttention(nn.Module):
    """MLX equivalent of the upstream Flamingo DenseXAttn, with cached board K/V."""

    def __init__(self, encoder_dim=1024, decoder_dim=2048, n_heads=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = decoder_dim // n_heads
        self.W_Q = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.W_O = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.norm_y = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.norm_x = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.norm_ffn = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.ffn = {
            "fc1": nn.Linear(decoder_dim, 2 * decoder_dim, bias=False),
            "fc2": nn.Linear(2 * decoder_dim, decoder_dim, bias=False),
        }
        self.alpha_attn = mx.zeros((1,))
        self.alpha_ffn = mx.zeros((1,))
        self._kv = None

    def set_board(self, states):
        states = self.norm_x(states)
        shape = (*states.shape[:2], self.n_heads, self.head_dim)
        self._kv = (
            self.W_K(states).reshape(shape).transpose(0, 2, 1, 3),
            self.W_V(states).reshape(shape).transpose(0, 2, 1, 3),
        )
        mx.eval(self._kv)

    def __call__(self, hidden):
        if self._kv is None:
            raise RuntimeError("A board must be encoded before generation")
        batch, length, _ = hidden.shape
        q = (
            self.W_Q(self.norm_y(hidden))
            .reshape(
                batch,
                length,
                self.n_heads,
                self.head_dim,
            )
            .transpose(0, 2, 1, 3)
        )
        attended = (
            mx.fast.scaled_dot_product_attention(
                q,
                *self._kv,
                scale=self.head_dim**-0.5,
            )
            .transpose(0, 2, 1, 3)
            .reshape(batch, length, -1)
        )
        hidden = hidden + mx.tanh(self.alpha_attn) * self.W_O(attended)
        ffn = self.ffn["fc2"](
            mx.square(nn.relu(self.ffn["fc1"](self.norm_ffn(hidden))))
        )
        return hidden + mx.tanh(self.alpha_ffn) * ffn


class QueenModel(Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        if args.num_hidden_layers != 36 or "sliding_attention" in args.layer_types:
            raise ValueError("Expected the released SmolLM3 QUEEN architecture")
        self.x_attn_layers = [
            CrossAttention(decoder_dim=args.hidden_size) for _ in range(16)
        ]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            self.model.embed_tokens(inputs)
            if input_embeddings is None
            else input_embeddings
        )
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Wrong number of decoder cache layers")
        mask = create_attention_mask(h, cache[0])
        for index, (layer, layer_cache) in enumerate(
            zip(self.layers, cache, strict=True)
        ):
            if index < 32 and index % 2 == 0:
                h = self.x_attn_layers[index // 2](h)
            h = layer(h, mask, cache=layer_cache)
        h = self.model.norm(h)
        return (
            self.model.embed_tokens.as_linear(h)
            if self.args.tie_word_embeddings
            else self.lm_head(h)
        )


def load_model(directory: Path) -> QueenModel:
    config = json.loads((directory / "config.json").read_text())
    model = QueenModel(ModelArgs.from_dict(config))
    weights = {}
    for file in sorted(directory.glob("model-*.safetensors")):
        weights.update(mx.load(str(file)))
    state = torch.load(directory / "xattn.pt", map_location="cpu", weights_only=True)
    weights.update(
        {
            f"x_attn_layers.{key}": mx.array(tensor.float().numpy()).astype(mx.bfloat16)
            for key, tensor in state.items()
        }
    )
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def sampler(temperature: float):
    """Apply temperature, top-k, then top-p, as in the published vLLM settings."""

    def sample(logprobs):
        if temperature == 0:
            return mx.argmax(logprobs, axis=-1)
        filtered = apply_top_p(apply_top_k(logprobs / temperature, 20), 0.95)
        return mx.random.categorical(filtered)

    return sample


class MLXRunner:
    def __init__(self, directory: Path):
        self.model = load_model(directory)

    def set_board(self, states):
        states = mx.array(states.float().cpu().numpy()).astype(mx.bfloat16)
        for index, bridge in enumerate(self.model.x_attn_layers):
            bridge.set_board(states[:, index])

    def generate(self, tokenizer, ids, max_tokens, temperature, seed):
        mx.random.seed(seed)
        start = perf_counter()
        tokens = []
        last = None
        for response in stream_generate(
            self.model,
            tokenizer,
            ids,
            max_tokens=max_tokens,
            sampler=sampler(temperature),
        ):
            tokens.append(response.token)
            last = response
        elapsed = perf_counter() - start
        if last and last.finish_reason == "stop":
            tokens.pop()  # The final stream response contains the EOS token.
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        return raw, {
            "generated_tokens": len(tokens),
            "generation_seconds": elapsed,
            "tokens_per_second": len(tokens) / elapsed,
            "decode_tokens_per_second": last.generation_tps if last else 0,
            "finish_reason": last.finish_reason if last else "length",
            "peak_mlx_memory_gb": mx.get_peak_memory() / 1e9,
        }
