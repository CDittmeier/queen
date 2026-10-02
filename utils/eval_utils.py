"""Generative evaluation: per-task grading on the final answer of each row.

Each gold ``response`` is structured as ``{prose-with-tokens}\\n\\n{answer-tokens}``.
We split on the last ``\\n\\n`` to isolate the trailing answer segment, then
dispatch to a per-task grader. The default is multiset (Counter) equality on
the special tokens parsed from the answer segment — order-free, but repetitions
are significant (so ``<PIECE_WB><PIECE_WB>`` ⇒ two white bishops grades
correctly for piece-counting tasks). Tasks whose parse_tag pins the queried
token at position 0 (``piece_on_square``, ``square_of_piece``) use an
anchored-multiset grader that requires exact match of the first token. Tasks
whose gold answer is a prose count expression (``material_count``,
``piece_count``) use an exact-string grader instead. New tasks can register
their own grader in ``_GRADERS`` without touching the dispatcher.
"""
import os
import re
from collections import Counter, defaultdict
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils.instance_format import (
    KEY_EXTRA,
    KEY_FEN,
    KEY_HISTORY,
    KEY_PROMPT,
    KEY_RESPONSE,
    ensure_chat_template,
    to_standard_instance,
)
from utils.lc0_planes import encode_fen_batch
from utils.utils import encode_planes, turn_tensor


def _final_answer(text: str) -> str:
    """Everything after the last blank line, stripped (the gold/predicted answer)."""
    return text.rsplit("\n\n", 1)[-1].strip()


# ---------------------------------------------------------------------------
# Grader registry. Default = multiset (Counter) equality on the special tokens
# extracted from each answer segment; tasks listed in _GRADERS override.
# ---------------------------------------------------------------------------

# Token regex: a SQUARE/PIECE/EMPTY token optionally followed by a count
# (e.g. '<PIECE_WP>2' = two white pawns). Missing count = 1.
_TOKEN_RE = re.compile(r"(<(?:SQUARE|PIECE)_[A-Z0-9]+>|<EMPTY>)(\d+)?")


def _parse_token_multiset(s: str) -> Counter:
    c: Counter = Counter()
    for tok, n in _TOKEN_RE.findall(s):
        c[tok] += int(n) if n else 1
    return c


def _multiset_grade(pred: str, gold: str) -> bool:
    return _parse_token_multiset(pred) == _parse_token_multiset(gold)


def _anchored_multiset_grade(pred: str, gold: str) -> bool:
    """Multiset grader with a required leading anchor token.

    The first `<SQUARE|PIECE|EMPTY>` token is the query anchor and must match
    exactly between pred and gold; the remainder is compared as an order-free
    token multiset (same shape as `_multiset_grade`). Used for tasks whose
    parse_tag contract pins the queried token to position 0 — without this,
    e.g. gold `<SQUARE_28><PIECE_K>` and pred `<PIECE_K><SQUARE_28>` would
    grade equal even though the model emitted the format inverted.
    """
    def split(s: str) -> tuple[str, str]:
        m = _TOKEN_RE.search(s)
        if m is None:
            return "", s
        return m.group(1), s[m.end():]
    pa, pr = split(pred.strip())
    ga, gr = split(gold.strip())
    return pa == ga and _parse_token_multiset(pr) == _parse_token_multiset(gr)


def _exact_grade(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def _piece_count_grade(pred: str, gold: str) -> bool:
    """Sectioned multiset grader for piece_count.

    parse_tag layout is exactly 4 non-blank lines:

        {side1_label}                       e.g. "white" / "player"
        {compact piece list for side1}      e.g. "<PIECE_WP>5<PIECE_WN>2..."
        {side2_label}                       e.g. "black" / "opponent"
        {compact piece list for side2}

    The side labels must match exactly; piece lines are compared as
    counted multisets via `_parse_token_multiset` (so the order within
    a side is free, but pieces must not leak across sides).
    """
    pred_lines = [ln.strip() for ln in pred.strip().split("\n") if ln.strip()]
    gold_lines = [ln.strip() for ln in gold.strip().split("\n") if ln.strip()]
    if len(pred_lines) != 4 or len(gold_lines) != 4:
        return False
    if pred_lines[0] != gold_lines[0] or pred_lines[2] != gold_lines[2]:
        return False
    return (
        _parse_token_multiset(pred_lines[1]) == _parse_token_multiset(gold_lines[1])
        and _parse_token_multiset(pred_lines[3]) == _parse_token_multiset(gold_lines[3])
    )


def _move_list_grade(pred: str, gold: str) -> bool:
    """Grader for a single-line move-list parse_tag: `<component> m1,m2,...`.

    The space-separated leading component (e.g. the queried `<piece><square>`)
    anchors the query and must match exactly; the comma-separated move list is
    compared as an order-free multiset. A component with no trailing list (the
    piece has no legal move) yields an empty move multiset.
    """
    def parse(s: str) -> tuple[str, Counter]:
        comp, _, rest = s.strip().partition(" ")
        return comp, Counter(m for m in rest.split(",") if m)
    pc, pm = parse(pred)
    gc, gm = parse(gold)
    return pc == gc and pm == gm


_PIECE_SQ_RE = re.compile(r"(<PIECE_[A-Z]+>)(<SQUARE_[0-9]+>)")


def _piece_square_pairs(s: str) -> Counter:
    """Multiset of (piece, square) pairs in a spaceless `<piece><square>...` list,
    keeping each piece bound to its square (so squares can't be swapped between
    pieces undetected)."""
    return Counter(_PIECE_SQ_RE.findall(s))


def _attacker_defender_grade(pred: str, gold: str) -> bool:
    """square_attackers grader. parse_tag = `<prefix> <attackers> <defenders>` (the
    defender list is absent for empty squares). The prefix (`<square>` or
    `<piece><square>`) must match exactly; the attacker and defender lists are each
    compared as an order-free (piece, square)-pair multiset, kept apart by the
    single space that separates the two lists."""
    def parse(s: str) -> tuple[str, Counter, Counter]:
        prefix, _, rest = s.strip().partition(" ")
        att, _, dfd = rest.partition(" ")
        return prefix, _piece_square_pairs(att), _piece_square_pairs(dfd)
    pp, pa, pd = parse(pred)
    gp, ga, gd = parse(gold)
    return pp == gp and pa == ga and pd == gd


def _motif_annotation_grade(pred: str, gold: str) -> bool:
    """Canonical motif-and-side equality across the controlled prose variants."""
    from datagen.tasks.hce_motif_annotation import grade_motif_response

    return grade_motif_response(pred, gold)


_DEFAULT_GRADER: Callable[[str, str], bool] = _multiset_grade
_GRADERS: dict[str, Callable[[str, str], bool]] = {
    "material_count":   _exact_grade,
    "is_checkmate":     _exact_grade,
    "piece_count":      _piece_count_grade,
    "piece_moves":      _move_list_grade,
    "square_attackers": _attacker_defender_grade,
    "checking_moves":   _move_list_grade,
    "capture_moves":    _move_list_grade,
    "check_parries":    _move_list_grade,
    "hce_motif_annotation": _motif_annotation_grade,
    # parse_tag pins the queried token first (`<sq><piece>` and
    # `<piece><sq>*` / `<piece><EMPTY>` respectively); plain multiset
    # grading would reward emitting the format inverted.
    "piece_on_square":  _anchored_multiset_grade,
    "square_of_piece":  _anchored_multiset_grade,
}


def _grade(task: str, pred: str, gold: str) -> bool:
    return _GRADERS.get(task, _DEFAULT_GRADER)(pred, gold)


# ---------------------------------------------------------------------------
# Batched greedy generation
# ---------------------------------------------------------------------------

def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Apply temperature / top-k / top-p and return next token ids (B,).

    temperature=0 → greedy argmax (no sampling).
    Mirrors the order transformers uses: temperature → top-k → top-p → multinomial.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    scores = logits / temperature

    if top_k > 0:
        k = min(top_k, scores.size(-1))
        cutoff = torch.topk(scores, k)[0][..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=False)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumprobs <= (1 - top_p)
        remove[..., -1:] = False  # always keep at least one token
        scores = scores.masked_fill(
            remove.scatter(1, sorted_idx, remove), float("-inf")
        )

    return torch.multinomial(scores.softmax(dim=-1), num_samples=1).squeeze(-1)


def _batched_decode(
    model,
    enc_hidden: torch.Tensor,
    prompt_ids_list: list[list[int]],
    pad_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.95,
) -> list[list[int]]:
    """Decode a batch of prompts, stopping each example at EOS.

    Returns a list of generated token ID lists (prompt tokens excluded).
    temperature=0 uses greedy argmax; >0 uses top-k/top-p sampling.

    Left-padding: padding is prepended so all real tokens end at position S-1.
    position_ids = attn_mask.cumsum(-1) - 1 gives each token its correct RoPE
    position (pads get 0, which is masked anyway). Generated tokens append to
    the right and always get the next correct position index.
    """
    B = len(prompt_ids_list)
    prompt_lens = [len(ids) for ids in prompt_ids_list]
    max_prompt  = max(prompt_lens)

    input_ids = torch.full((B, max_prompt), pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt,                dtype=torch.long, device=device)
    for i, (ids, l) in enumerate(zip(prompt_ids_list, prompt_lens)):
        input_ids[i, max_prompt - l:] = torch.tensor(ids, dtype=torch.long, device=device)
        attn_mask[i, max_prompt - l:] = 1

    done      = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]

    for _ in range(max_new_tokens):
        if done.all():
            break

        position_ids = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(input_ids, enc_hidden, attn_mask, position_ids=position_ids)

        next_logits = logits[:, -1, :]                                            # (B, V)
        next_tokens = _sample_next_token(next_logits, temperature, top_k, top_p)  # (B,)

        new_col   = torch.where(done, torch.full_like(next_tokens, pad_token_id), next_tokens)
        input_ids = torch.cat([input_ids, new_col.unsqueeze(1)], dim=1)
        attn_mask = torch.cat([attn_mask, (~done).long().unsqueeze(1)], dim=1)

        for i in range(B):
            if not done[i]:
                tok = next_tokens[i].item()
                generated[i].append(tok)
                if tok == eos_token_id:
                    done[i] = True

    return generated


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _eval_collate(batch: list[dict]) -> list[dict]:
    return [to_standard_instance(ex) for ex in batch]


def resolve_stop_id(model, tokenizer, eos_id: int) -> int:
    """Return the assistant-turn terminator id for this decoder family."""
    adapter = getattr(model, "adapter", None)
    stop_ids = adapter.stop_token_ids(tokenizer) if adapter is not None else []
    stop_id = stop_ids[0] if stop_ids else tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(stop_id, int) or stop_id < 0 or stop_id == tokenizer.unk_token_id:
        stop_id = eos_id
    return stop_id


@torch.no_grad()
def run_eval(
    model,
    encoder,
    eval_dataset,
    tokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    eval_batch_size: int,
    eval_max_new_tokens: int,
    *,
    pov: bool,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.95,
    max_examples: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Run generative eval over the eval dataset (or a capped subset).

    Grades the final answer (text after the last blank line) per ``extra.task``
    using the per-task grader registry (multiset match on parsed special
    tokens by default; exact string match for prose-count tasks).
    Returns (metrics, samples) where samples contains every evaluated example.
    """
    model.eval()
    ensure_chat_template(tokenizer)

    if max_examples is not None:
        eval_dataset = eval_dataset.select(range(min(max_examples, len(eval_dataset))))

    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer.eos_token_id is None — generation cannot detect stop"
    if tokenizer.pad_token_id is None:
        print(f"[run_eval] WARNING: tokenizer.pad_token_id is None; using eos_token_id ({eos_id}) as pad.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    # Use the family-specific assistant-turn terminator, not eos_token_id — the
    # model never emits eos, so generation would run to the token cap otherwise.
    stop_id = resolve_stop_id(model, tokenizer, eos_id)

    loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
    )

    stats   = defaultdict(lambda: {"total": 0, "correct": 0})
    samples = []

    if os.environ.get("TEST_MODE") == "1":
        from tqdm import tqdm
        loader = tqdm(loader, desc="eval batches", dynamic_ncols=True)

    for batch in loader:
        prompts   = [s[KEY_PROMPT] for s in batch]
        fens      = [s[KEY_FEN] for s in batch]
        histories = [s[KEY_HISTORY] or None for s in batch]
        golds     = [_final_answer(s[KEY_RESPONSE]) for s in batch]
        tasks     = [s[KEY_EXTRA].get("task", "all") for s in batch]

        # Prompt = the user turn rendered through the chat template (same prefix
        # training masks), with the assistant generation prompt appended.
        prompt_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]

        enc_hidden = encode_planes(
            encoder, encode_fen_batch(fens, histories).to(device), amp_dtype,
            pov=pov, turn=turn_tensor(fens),
        )

        gen_ids_list = _batched_decode(
            model, enc_hidden, prompt_ids_list,
            pad_id, stop_id, eval_max_new_tokens, device, amp_dtype,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

        for i, gen_ids in enumerate(gen_ids_list):
            if gen_ids and gen_ids[-1] == stop_id:
                gen_ids = gen_ids[:-1]                       # drop the stop token
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            pred    = _final_answer(generated_text)
            correct = _grade(tasks[i], pred, golds[i])

            stats[tasks[i]]["total"]   += 1
            stats[tasks[i]]["correct"] += int(correct)
            samples.append({
                "task":      tasks[i],
                "fen":       fens[i],
                "prompt":    prompts[i],
                "generated": generated_text,
                "gold":      golds[i],
                "pred":      pred,
                "correct":   correct,
            })

    metrics: dict[str, float] = {}
    total_all = correct_all = 0
    for task, s in stats.items():
        n = s["total"]
        metrics[f"{task}/correct"] = s["correct"] / n if n else 0.0
        total_all   += n
        correct_all += s["correct"]
    metrics["overall/correct"] = correct_all / total_all if total_all else 0.0

    return metrics, samples
