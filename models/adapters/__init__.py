"""Per-decoder-family DecoderAdapter registry.

``adapter_for(decoder)`` returns the concrete adapter for a loaded HF causal-LM
decoder, dispatched by ``config.model_type``. Concrete implementations live in
sibling modules (smollm.py, gemma.py) and are lazy-imported so an
in-progress / missing adapter doesn't break this package's import.

Adding a new decoder family means: (1) write the adapter class in a new sibling
module, (2) add a case in ``adapter_for`` for its ``config.model_type`` string.
"""
from __future__ import annotations

from models.adapters.base import DecoderAdapter


def adapter_for(decoder) -> DecoderAdapter:
    """Construct the per-family DecoderAdapter for an HF causal-LM decoder.

    Dispatches by ``decoder.config.model_type``. The returned adapter binds to
    the given decoder instance — call this once per loaded model.

    Raises NotImplementedError for unsupported model types.
    """
    model_type = decoder.config.model_type

    if model_type == "smollm3":
        from models.adapters.smollm import SmolLM3Adapter
        return SmolLM3Adapter(decoder)

    if model_type in ("gemma3_text", "gemma3"):
        from models.adapters.gemma import Gemma3Adapter
        return Gemma3Adapter(decoder)

    if model_type == "qwen3":
        from models.adapters.qwen import Qwen3Adapter
        return Qwen3Adapter(decoder)

    raise NotImplementedError(
        f"No DecoderAdapter implemented for model_type={model_type!r}. "
        f"Add a case in models/adapters/__init__.py::adapter_for."
    )


def layer_cls_for_config(config) -> str:
    """Return the HF decoder-layer class name for a decoder *config* — no model
    instance needed.

    Used by train.py to resolve the FSDP ``transformer_cls_names_to_wrap`` list
    *before* the model is built (the FSDP plugin is constructed first). Dispatch
    mirrors ``adapter_for`` but reads ``layer_cls_name`` off the adapter class
    rather than instantiating it.
    """
    model_type = config.model_type

    if model_type == "smollm3":
        from models.adapters.smollm import SmolLM3Adapter
        return SmolLM3Adapter.layer_cls_name

    if model_type in ("gemma3_text", "gemma3"):
        from models.adapters.gemma import Gemma3Adapter
        return Gemma3Adapter.layer_cls_name

    if model_type == "qwen3":
        from models.adapters.qwen import Qwen3Adapter
        return Qwen3Adapter.layer_cls_name

    raise NotImplementedError(
        f"No DecoderAdapter implemented for model_type={model_type!r}. "
        f"Add a case in models/adapters/__init__.py::layer_cls_for_config."
    )


__all__ = ["adapter_for", "layer_cls_for_config", "DecoderAdapter"]
