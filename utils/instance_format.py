"""Standardized chess training-instance format.

A training instance is a dict with five keys (see ``lab/format.md``):

    fen      : str        — the **current/final** position (the board lc0 reads
                            and that the question is about).
    history  : list[str]  — the prior context lc0 needs to fill its 8 history
                            planes. Two accepted forms (auto-detected by
                            ``lc0_planes._build_boards``):
                              * prior board **FENs**, oldest->newest, excluding
                                ``fen`` (the natural form when ``fen`` is final);
                              * legacy UCI **plies** with ``fen`` as the root.
                            Empty list for a static position (then the history
                            planes are synthesized from ``fen``).
    prompt   : str        — the question / user turn (text).
    response : str        — the gold answer (text; no trailing period — the
                            EOS token is appended at tokenization).
    extra    : dict       — arbitrary evaluation metadata (game_id, ply,
                            task, score_expr, ...). NOT fed to the model.

Consumption:
    fen + history     -> lc0 input planes (board features for the bridge)
    prompt + response -> tokenized, with the prompt span masked to -100 so
                         only the response (+ EOS) contributes to the loss
    extra             -> carried alongside for eval tooling; the collator
                         does not forward it to the model.

``fen`` is the final board; ``history`` holds the preceding board states. You
can't replay backwards from a bare FEN, so when ``fen`` is final the prior
boards are stored as FENs. This maps to the collaborator's
``(start_fen, move_list, end_fen)`` triples: our ``fen`` == their ``end_fen``;
``history`` == the boards their ``(start_fen, move_list)`` produces (minus the
final one).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


# Canonical keys.
KEY_FEN = "fen"
KEY_HISTORY = "history"
KEY_PROMPT = "prompt"
KEY_RESPONSE = "response"
KEY_EXTRA = "extra"

# Column-name fallbacks so the loader tolerates the ``lab/format.md`` MDS
# naming (in_text/out_text) and the legacy val-JSONL naming (q_text/a_text).
_FEN_ALIASES = (KEY_FEN, "chess_fen", "start_fen")
_HISTORY_ALIASES = (KEY_HISTORY, "moves", "uci_moves", "move_list")
_PROMPT_ALIASES = (KEY_PROMPT, "in_text", "q_text", "question")
_RESPONSE_ALIASES = (KEY_RESPONSE, "out_text", "a_text", "answer")

# Fallback chat template (ChatML) for the rare tokenizer that ships without
# one. The prompt becomes the user turn and the response the assistant turn.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def has_prompt_response(item: Any) -> bool:
    """True if ``item`` looks like a standardized text instance."""
    return any(k in item for k in _PROMPT_ALIASES) and any(k in item for k in _RESPONSE_ALIASES)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return as_text(value.item())
    return str(value)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [as_text(v) for v in value]
    # a single string of space-separated moves, or one move
    text = as_text(value).strip()
    return text.split() if text else []


def _first(item: Any, aliases: Tuple[str, ...], default: Any = None) -> Any:
    for k in aliases:
        if k in item and item[k] is not None:
            return item[k]
    return default


def to_standard_instance(item: Any) -> dict:
    """Normalize a raw dataset row to the five standardized keys."""
    return {
        KEY_FEN: as_text(_first(item, _FEN_ALIASES, "")),
        KEY_HISTORY: _as_str_list(_first(item, _HISTORY_ALIASES, [])),
        KEY_PROMPT: as_text(_first(item, _PROMPT_ALIASES, "")),
        KEY_RESPONSE: as_text(_first(item, _RESPONSE_ALIASES, "")),
        KEY_EXTRA: _first(item, (KEY_EXTRA,), {}) or {},
    }


def ensure_chat_template(tokenizer) -> None:
    """Give the tokenizer a default ChatML template if it lacks one. Most
    instruct tokenizers already ship one; this is the worst-case fallback."""
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


def tokenize_instance(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Tokenize one instance into (input_ids, labels) with the prompt masked.

    The prompt is the user turn and the response the assistant turn, rendered
    through the tokenizer's chat template. The user turn (and the assistant
    generation prefix) is masked to -100; the assistant content and its closing
    token(s) are supervised, so the model also learns to stop.
    """
    ensure_chat_template(tokenizer)
    full = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    prefix = [{"role": "user", "content": prompt}]
    input_ids = list(tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False, enable_thinking=False))
    prompt_ids = list(tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if max_length is not None and max_length > 0:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels
