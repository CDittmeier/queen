from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from models.flamingo import DenseXAttn


class CachedCrossAttention:
    """Cache board K/V once; use the upstream trained sublayer for everything else."""

    def __init__(self, layer: DenseXAttn, states: torch.Tensor):
        self.layer = layer
        states = layer.norm_x(states)
        shape = (states.shape[0], states.shape[1], layer.n_kv_heads, layer.head_dim)
        self.k = layer.W_K(states).view(shape).transpose(1, 2)
        self.v = layer.W_V(states).view(shape).transpose(1, 2)
        self.k = self.k.repeat_interleave(layer.n_rep, dim=1)
        self.v = self.v.repeat_interleave(layer.n_rep, dim=1)

    def __call__(self, hidden):
        layer = self.layer
        batch, length, _ = hidden.shape
        q = (
            layer.W_Q(layer.norm_y(hidden))
            .view(
                batch,
                length,
                layer.n_heads,
                layer.head_dim,
            )
            .transpose(1, 2)
        )
        value = F.scaled_dot_product_attention(q, self.k, self.v)
        value = value.transpose(1, 2).reshape(batch, length, -1)
        hidden = hidden + torch.tanh(layer.alpha_attn) * layer.W_O(value)
        return hidden + torch.tanh(layer.alpha_ffn) * layer.ffn(layer.norm_ffn(hidden))


class MPSRunner:
    def __init__(self, directory: Path):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                directory,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to("mps")
            .eval()
        )
        self.bridges = (
            torch.nn.ModuleList(
                [DenseXAttn(1024, self.model.config.hidden_size) for _ in range(16)]
            )
            .to(device="mps", dtype=torch.bfloat16)
            .eval()
        )
        self.bridges.load_state_dict(
            torch.load(
                directory / "xattn.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        self._hooks = []

    def clear_board(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.inference_mode()
    def set_board(self, states):
        self.clear_board()
        for index, bridge in enumerate(self.bridges):
            cached = CachedCrossAttention(
                bridge,
                states[:, index].to(next(bridge.parameters()).dtype),
            )

            def hook(module, args, kwargs, cached=cached):
                if args:
                    return (cached(args[0]), *args[1:]), kwargs
                return args, {
                    **kwargs,
                    "hidden_states": cached(kwargs["hidden_states"]),
                }

            self._hooks.append(
                self.model.model.layers[2 * index].register_forward_pre_hook(
                    hook,
                    with_kwargs=True,
                )
            )

    @torch.inference_mode()
    def generate(self, tokenizer, ids, max_tokens, temperature, seed):
        if len(self._hooks) != 16:
            raise RuntimeError("A board must be encoded before generation")
        torch.manual_seed(seed)
        inputs = torch.tensor([ids], device="mps")
        settings = {"do_sample": temperature > 0}
        if temperature > 0:
            settings.update(temperature=temperature, top_k=20, top_p=0.95)
        torch.mps.synchronize()
        start = perf_counter()
        output = self.model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=max_tokens,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
            **settings,
        )[0, len(ids) :]
        torch.mps.synchronize()
        elapsed = perf_counter() - start
        tokens = output.tolist()
        stopped = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
        raw = tokenizer.decode(
            tokens[:-1] if stopped else tokens, skip_special_tokens=False
        )
        return raw, {
            "generated_tokens": len(tokens),
            "generation_seconds": elapsed,
            "tokens_per_second": len(tokens) / elapsed,
            "finish_reason": "stop" if stopped else "length",
        }
