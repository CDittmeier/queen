"""vLLM model for stage-5 serving: SmolLM3 with the board-prefix position scheme.

The stage-5 decoder (models/llava.py) consumes 64 board-prefix embeddings followed
by the text tokens, with every prefix token pinned to RoPE position 0 and text
starting at position 64. We feed vLLM the [prefix; text] sequence pre-embedded via
prompt_embeds; vLLM then assigns contiguous RoPE positions, so text lands at 64..
(already correct) but the prefix lands at 0..63 (wrong — it must be all-0). Serving
with contiguous prefix positions shifts early-answer logits by up to ~18 and
corrupts generation, so we re-pin the prefix positions to 0.

Because the prompt is always exactly 64 prefix tokens then text, and text positions
are always >= 64, the test `position < 64` identifies prefix tokens exactly — an
elementwise transform that needs no sequence-boundary bookkeeping and is safe under
batching / chunked prefill.

Everything else — weight loading, paged attention, SmolLM3's NoPE layers, the tied
lm_head over the merged 128333-token vocab — is inherited unchanged from vLLM's
TransformersForCausalLM.
"""
import torch
from vllm import ModelRegistry
from vllm.model_executor.models.transformers import TransformersForCausalLM

N_PREFIX = 64   # board-prefix tokens; must match models/llava.py's N_ENC_SQUARES
_ARCH = "SmolLM3ForCausalLM"
_IMPL = "models.vllm.llava:ChessSmolLM3ForCausalLM"


class ChessSmolLM3ForCausalLM(TransformersForCausalLM):
    """SmolLM3 with the 64 board-prefix RoPE positions pinned to 0."""

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        # With enable_prompt_embeds the runner embeds token-id positions itself; the
        # board prefix arrives pre-embedded via prompt_embeds (nothing to merge here).
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        positions = torch.where(positions < N_PREFIX,
                                torch.zeros_like(positions), positions)
        return super().forward(input_ids, positions, intermediate_tensors,
                               inputs_embeds, **kwargs)


def register() -> None:
    """Route vLLM's SmolLM3 architecture to the position-fixed subclass.

    Registered by import-path string so vLLM worker subprocesses (which import the
    model lazily) resolve it too; this needs the repo root on PYTHONPATH, which
    holds when serving is launched from the repo root.
    """
    ModelRegistry.register_model(_ARCH, _IMPL)


register()   # register on import, for the string-path lookup in worker subprocesses
