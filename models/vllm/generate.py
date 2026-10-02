"""Fast stage-5 generation on vLLM.

Pipeline per request:
  FEN         -> lc0 encoder + connector (BoardBridge)     -> 64 prefix embeds
  prompt text -> chat template -> merged embed_tokens       -> text embeds
  [prefix; text] embeds -> vLLM (SmolLM3, position-fixed)   -> tokens

The decoder is the merged SmolLM3 (merge_weights.py); the 77 chess tokens live in
its vocab, so vLLM samples them natively. The board prefix is injected as input
embeddings via prompt_embeds, and the custom model (models/vllm/llava.py) pins the
prefix RoPE positions to 0. Together these reproduce the reference model.

    from models.vllm.generate import ChessVLLMGenerator
    g = ChessVLLMGenerator(merged_dir, encoder_path)
    outs = g.generate(fens, prompts, temperature=0.0, max_tokens=1024)
    # outs[i] = {"token_ids": [...], "text": "..."}
"""
import json
import os
from pathlib import Path

# The custom LLaVA model is registered in this process before vLLM starts.
# Keeping EngineCore in-process avoids forking after that registration path has
# initialized CUDA (which PyTorch rejects on recent releases).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from models.vllm.bridge import BoardBridge
from models.vllm.llava import register

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    """Read a single weight tensor from a (possibly sharded) safetensors dir."""
    index = model_dir / "model.safetensors.index.json"
    shard = json.load(open(index))["weight_map"][name] if index.exists() else "model.safetensors"
    with safe_open(model_dir / shard, framework="pt") as f:
        return f.get_tensor(name)


class ChessVLLMGenerator:
    def __init__(self, merged_dir, encoder_path, *, dtype="bfloat16",
                 gpu_memory_utilization=0.55, max_model_len=2048, seed=0,
                 pov=True, enforce_eager=True):
        merged_dir = Path(merged_dir)
        self.device = torch.device("cuda")

        register()   # route SmolLM3ForCausalLM -> our position-fixed subclass
        from vllm import LLM
        self.llm = LLM(
            model=str(merged_dir),
            dtype=dtype,
            enable_prompt_embeds=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            seed=seed,
        )
        self.tok = AutoTokenizer.from_pretrained(str(merged_dir))
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

        # Loaded after vLLM has claimed its pool, so they use the remaining GPU.
        amp = _DTYPES[dtype]
        self.bridge = BoardBridge(merged_dir / "bridge.pt", encoder_path=encoder_path,
                                  device=self.device, dtype=amp, pov=pov)
        self.embed = _load_tensor(merged_dir, "model.embed_tokens.weight").to(self.device, amp)

    def _prompt_ids(self, prompt: str) -> list[int]:
        out = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        # transformers 5 returns a BatchEncoding here rather than the plain
        # list returned by older releases.
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    @torch.no_grad()
    def generate(self, fens, prompts, histories=None, *, temperature=0.0,
                 top_k=20, top_p=0.95, max_tokens=1024, seed=None, repetition_penalty=1.0):
        from vllm import SamplingParams
        assert len(fens) == len(prompts)
        prefix = self.bridge.prefix_from_fens(fens, histories)      # (B, 64, decoder_dim)

        prompt_embeds = []
        for i in range(len(fens)):
            ids = torch.tensor(self._prompt_ids(prompts[i]), device=self.device)
            text_emb = self.embed[ids]                              # (S, decoder_dim)
            seq = torch.cat([prefix[i], text_emb], dim=0)           # (64+S, decoder_dim)
            prompt_embeds.append({"prompt_embeds": seq.to(torch.bfloat16).cpu()})

        sampling = SamplingParams(
            temperature=temperature,
            top_k=(top_k if temperature > 0 else -1),
            top_p=(top_p if temperature > 0 else 1.0),
            max_tokens=max_tokens,
            stop_token_ids=[self.im_end],
            seed=seed,
            repetition_penalty=repetition_penalty,
        )
        results = []
        for out in self.llm.generate(prompt_embeds, sampling):
            ids = list(out.outputs[0].token_ids)
            if ids and ids[-1] == self.im_end:
                ids = ids[:-1]
            results.append({"token_ids": ids,
                            "text": self.tok.decode(ids, skip_special_tokens=False)})
        return results
