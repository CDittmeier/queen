"""Fast flamingo generation on vLLM.

Pipeline per request:
  FEN         -> lc0 encoder                       -> (16, 64, 1024) encoder states
  states      -> registered against a request id   -> cross-attn K/V, cached per slot
  prompt text -> chat template -> token ids        -> vLLM (SmolLM3 + 16 x-attn layers)

Unlike the llava generator there are no prompt embeds: the board enters through the
cross-attention sublayers, not the token sequence, so requests are plain token ids and
the board rides a side channel keyed by request id (models/vllm/flamingo.py).

That side channel is a module-global, so the engine MUST run in-process -- this module
sets VLLM_ENABLE_V1_MULTIPROCESSING=0 at import, before vLLM is imported. Importing it
after vLLM has already started an out-of-process engine will not work.

    from models.vllm.flamingo_generate import ChessFlamingoGenerator
    g = ChessFlamingoGenerator(merged_dir, encoder_path)
    outs = g.generate(fens, prompts, temperature=0.0, max_tokens=512)
"""
import os

# Must precede any vllm import: the board side channel is a module-global, which only
# reaches the model when the v1 EngineCore runs in this process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# The board registry is keyed by request id. Newer vLLM re-randomizes the id after
# add_request (external_req_id -> f"{id}-{uuid}"), which would leave the model
# looking up a key nobody registered; our ids are already unique per engine.
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


import torch
from transformers import AutoTokenizer

from models.encoder import Lc0Bt4HFModel
from utils.lc0_planes import encode_fen_batch
from utils.utils import encode_planes, turn_tensor

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ChessFlamingoGenerator:
    def __init__(self, merged_dir, encoder_path, *, dtype="bfloat16",
                 gpu_memory_utilization=0.55, max_model_len=2048, max_num_seqs=64,
                 seed=0, pov=True, enforce_eager=True, async_engine=False,
                 enable_sleep_mode=False, distributed_executor_backend=None,
                 data_parallel_size=1, use_v1_vllm=False):
        if not enforce_eager:
            raise ValueError("Flamingo requires eager execution for board-conditioned cross-attention")
        # vLLM caches this process-wide setting after startup. Select explicitly,
        # rather than inheriting an old shell's V1 override; never switch live engines.
        import vllm.envs as vllm_envs
        use_v2 = not use_v1_vllm
        if (vllm_envs._is_envs_cache_enabled()
                and vllm_envs.VLLM_USE_V2_MODEL_RUNNER != use_v2):
            raise RuntimeError("Select the Flamingo vLLM runner in a fresh process")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if use_v2 else "0"
        from pathlib import Path
        merged_dir = Path(merged_dir)
        self.device = torch.device("cuda")
        self.amp = _DTYPES[dtype]
        self.pov = pov

        from models.vllm import flamingo as fl
        self.fl = fl
        fl.register()

        # `async_engine` swaps the offline LLM for AsyncLLM, which admits a
        # request the moment it arrives instead of only at a batch boundary. The
        # sublayers and the runner hook attach the same way either side, so the
        # synchronous path is unaffected by this being here.
        # Prefix caching MUST stay off. Every request here sends the same
        # prompt token ids -- the position enters only through cross-attention,
        # which vLLM's block hash cannot see. So vLLM judges two requests to
        # share a prefix when their boards differ, and serves the second from
        # KV computed while attending the first one's board. The output stays
        # fluent (each decode step re-injects the right board) but describes
        # the wrong position: measured on 8 positions, usable plan bullets fell
        # from 23/32 to 16/32, two positions losing all four.
        kw = dict(model=str(merged_dir), dtype=dtype,
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  enforce_eager=enforce_eager, seed=seed,
                  enable_prefix_caching=False,
                  enable_sleep_mode=enable_sleep_mode)
        if distributed_executor_backend is not None:
            kw["distributed_executor_backend"] = distributed_executor_backend
        if data_parallel_size != 1:
            kw["data_parallel_size"] = data_parallel_size
        self.is_async = async_engine
        if async_engine:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            self.llm = None
            self.async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
        else:
            from vllm import LLM
            self.async_llm = None
            self.llm = LLM(**kw)
        # The sublayers attach inside the model's __init__ (it finds xattn.pt in the
        # model dir), so they are present for profiling and CUDA-graph capture.
        self.model = self._live_model()
        assert self.model._xattn_ready, f"xattn.pt not found in {merged_dir}"

        self.tok = AutoTokenizer.from_pretrained(str(merged_dir))
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.encoder = Lc0Bt4HFModel.from_pretrained(encoder_path, local_files_only=True)
        self.encoder.to(device=self.device, dtype=self.amp).eval()

    def _live_model(self):
        """The nn.Module inside the running engine. Only reachable because the engine is
        in-process; the executor attribute name has moved between vLLM versions, so try
        the known spellings rather than pinning one."""
        ex = (self.async_llm.engine_core.engine_core.model_executor
              if self.llm is None else self.llm.llm_engine.model_executor)
        for path in (("driver_worker", "model_runner", "model"),
                     ("driver_worker", "worker", "model_runner", "model")):
            obj = ex
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise RuntimeError(f"cannot locate the live model on executor {type(ex).__name__}")

    def _prompt_ids(self, prompt: str) -> list[int]:
        out = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        # transformers 5 returns a BatchEncoding here, and iterating that yields
        # its KEYS -- vLLM would receive a sequence of strings where it wants
        # token ids, and fail comparing them against the vocab size
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    @torch.no_grad()
    def _encode(self, fens, histories):
        planes = encode_fen_batch(fens, histories).to(self.device)
        return encode_planes(self.encoder, planes, self.amp,
                             pov=self.pov, turn=turn_tensor(fens))   # (B, 16, 64, 1024)

    @torch.no_grad()
    def generate(self, fens, prompts, histories=None, *, temperature=0.0,
                 top_k=20, top_p=0.95, max_tokens=1024, seed=None,
                 repetition_penalty=1.0, return_logprobs=False, seeds=None):
        from vllm import SamplingParams, TokensPrompt
        assert len(fens) == len(prompts)
        if histories is None:
            histories = [None] * len(fens)
        enc = self._encode(fens, histories)

        requests = [TokensPrompt(prompt_token_ids=self._prompt_ids(p))
                    for p in prompts]

        def sampling_params(request_seed):
            return SamplingParams(
                temperature=temperature,
                top_k=(top_k if temperature > 0 else -1),
                top_p=(top_p if temperature > 0 else 1.0),
                max_tokens=max_tokens,
                stop_token_ids=[self.im_end],
                seed=request_seed,
                repetition_penalty=repetition_penalty,
                logprobs=(1 if return_logprobs else None),
                flat_logprobs=return_logprobs,
            )

        if seeds is not None:
            if len(seeds) != len(requests):
                raise ValueError(f"received {len(seeds)} seeds for {len(requests)} requests")
            sampling = [sampling_params(request_seed) for request_seed in seeds]
        else:
            sampling = sampling_params(seed)
        # The board is keyed by vLLM's request id, which the engine mints itself and
        # whose format has changed across versions (str(counter), then
        # f"{counter}-{uuid}"). Rather than predict it, claim each id as the request
        # is submitted: add_request runs in prompt order and always before the
        # request can be scheduled, so the board is registered in time.
        engine = self.llm.llm_engine
        original_add = engine.add_request
        req_ids, nth = [], iter(range(len(requests)))

        def add_request(request_id, *args, **kwargs):
            self.fl.set_encoder_states(request_id, enc[next(nth)])
            req_ids.append(request_id)
            return original_add(request_id, *args, **kwargs)

        engine.add_request = add_request
        try:
            outs = self.llm.generate(requests, sampling)
        finally:
            engine.add_request = original_add
            self.fl.drop_encoder_states(req_ids)

        results = []
        for out in outs:
            completion = out.outputs[0]
            ids = list(completion.token_ids)
            token_logprobs = None
            if return_logprobs:
                if completion.logprobs is None:
                    raise RuntimeError("vLLM did not return requested sampled-token log-probabilities")
                token_logprobs = [
                    float(position[token_id].logprob)
                    for token_id, position in zip(ids, completion.logprobs)
                ]
            if ids and ids[-1] == self.im_end:
                ids = ids[:-1]
                if token_logprobs is not None:
                    token_logprobs = token_logprobs[:-1]
            result = {
                "token_ids": ids,
                "text": self.tok.decode(ids, skip_special_tokens=False),
                "finish_reason": completion.finish_reason,
            }
            if token_logprobs is not None:
                result["token_logprobs"] = token_logprobs
            results.append(result)
        return results
