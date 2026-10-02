"""Merge a stage-5 LLaVAChessLM checkpoint into a standard HF SmolLM3 checkpoint
that vLLM can load, plus a small bridge.pt side file.

The stage-5 model keeps the 77 chess answer tokens out of the decoder vocab: their
input embeddings live in a separate `new_embed` and their logits in a tied
`new_lm_head` (models/base.py). vLLM's sampler only knows the decoder's own tied
lm_head over the standard vocab, so it cannot emit these tokens as-is. This folds
them in: resize SmolLM3 embed/lm_head 128256 -> 128333 (tied, so one matrix), copy
the fine-tuned base rows, and append `new_embed` as the 77 new rows (ids
128256..128332 == the frozen_vocab offset the model uses). The result is a vanilla
SmolLM3ForCausalLM that natively generates every token.

The board prefix (connector + file/rank spatial embeds) is NOT part of the decoder;
it is saved to bridge.pt and applied at generation time (models/vllm/bridge.py).

    python -m models.vllm.merge_weights \
        --ckpt runs/stage5_llava_040_fp32/step_0003200 \
        --train-dataset data/stage5/0-0-40/train_all.arrow \
        --out /tmp/merged/step_0003200
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.special_tokens import maybe_add_special_tokens
from utils.training_utils import _load_dataset_pov

BASE_VOCAB = 128256   # SmolLM3 config vocab_size == the model's frozen_vocab offset
_BRIDGE_KEYS = ("connector.0.weight", "connector.0.bias",
                "connector.1.weight", "connector.3.weight",
                "file_embed.weight", "rank_embed.weight")


def _clean_key(k: str) -> str:
    """Strip the torch.compile prefix and activation-checkpoint wrapper infix."""
    return k.removeprefix("_orig_mod.").replace("._checkpoint_wrapped_module", "")


def merge(ckpt, out, train_dataset, base="local/SmolLM3-3B", save_dtype=torch.bfloat16):
    """Write the merged SmolLM3 + bridge.pt for `ckpt` into `out`."""
    ckpt, out = Path(ckpt), Path(out)
    sd = {_clean_key(k): v for k, v in torch.load(ckpt / "trainable.pt", map_location="cpu").items()}
    dec = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
    new_embed = sd["new_embed.weight"]                       # (77, 2048)
    n_new = new_embed.shape[0]
    print(f"[merge] decoder tensors={len(dec)}  new_embed={tuple(new_embed.shape)}")

    # --- tokenizer with the 77 chess tokens (ids BASE_VOCAB..BASE_VOCAB+76) ---
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    pov = _load_dataset_pov(train_dataset)
    added = maybe_add_special_tokens(tok, SimpleNamespace(pov=pov))
    assert added == n_new, f"tokenizer added {added} tokens != checkpoint's {n_new}"
    assert len(tok) == BASE_VOCAB + n_new, f"len(tok)={len(tok)} != {BASE_VOCAB + n_new}"
    first_new = tok.convert_tokens_to_ids("<SQUARE_1>")
    assert first_new == BASE_VOCAB, f"first new id {first_new} != {BASE_VOCAB}"
    print(f"[merge] pov={pov} n_new={n_new} first_new_id={first_new} len(tok)={len(tok)}")

    # --- base model, overlay fine-tuned decoder, resize, splice new rows ---
    print("[merge] loading base SmolLM3 (fp32, cpu)...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32,
                                                 local_files_only=True)
    missing, unexpected = model.load_state_dict(dec, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied -> mirrors embed_tokens
    assert not missing, f"unexpected MISSING decoder keys: {missing[:8]}"
    assert not unexpected, f"unexpected EXTRA decoder keys: {unexpected[:8]}"
    assert model.config.tie_word_embeddings, "expected tied embeddings"

    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    emb.data[BASE_VOCAB:BASE_VOCAB + n_new] = new_embed.to(emb.dtype)
    assert model.get_output_embeddings().weight.data_ptr() == emb.data_ptr(), "tie broke"
    assert model.config.vocab_size == BASE_VOCAB + n_new

    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged SmolLM3 -> {out}")
    model.to(save_dtype).save_pretrained(out)
    tok.save_pretrained(out)

    bridge = {k: sd[k] for k in _BRIDGE_KEYS}
    torch.save(bridge, out / "bridge.pt")
    print(f"[merge] saved bridge.pt ({len(bridge)} tensors)")
    print("[merge] DONE")


def merge_flamingo(ckpt, out, train_dataset, base="local/SmolLM3-3B",
                   save_dtype=torch.bfloat16):
    """Same idea as merge(), for a FLAMINGO checkpoint.

    The decoder may be frozen (no ``decoder.*`` tensors) or fully fine-tuned;
    in the latter case its weights are overlaid exactly as in :func:`merge`.
    The board side is 16 cross-attention sublayers, saved to ``xattn.pt`` for
    ``models/vllm/flamingo.py`` to hook back in.
    """
    ckpt, out = Path(ckpt), Path(out)
    sd = {_clean_key(k): v for k, v in torch.load(ckpt / "trainable.pt", map_location="cpu").items()}
    xattn = {k[len("x_attn_layers."):]: v for k, v in sd.items() if k.startswith("x_attn_layers.")}
    dec = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
    assert xattn, "no x_attn_layers.* tensors — is this really a flamingo checkpoint?"
    new_embed = sd["new_embed.weight"]
    n_new = new_embed.shape[0]
    print(f"[merge] flamingo x_attn tensors={len(xattn)} decoder tensors={len(dec)} "
          f"new_embed={tuple(new_embed.shape)}")

    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    pov = _load_dataset_pov(train_dataset)
    added = maybe_add_special_tokens(tok, SimpleNamespace(pov=pov))
    assert added in (0, n_new), (
        f"tokenizer added {added} tokens, expected either an already-merged "
        f"tokenizer or all {n_new} checkpoint tokens"
    )
    assert len(tok) == BASE_VOCAB + n_new, (
        f"tokenizer size {len(tok)} != {BASE_VOCAB}+{n_new}"
    )
    assert tok.convert_tokens_to_ids("<SQUARE_1>") == BASE_VOCAB
    print(f"[merge] pov={pov} n_new={n_new} added={added} len(tok)={len(tok)}")

    print("[merge] loading base SmolLM3...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32,
                                                 local_files_only=True)
    # ``base`` may itself be a prior merged RL checkpoint. Split its chess
    # vocabulary suffix back off before applying the training representation's
    # decoder state; the current ``new_embed`` rows are appended again below.
    if model.config.vocab_size != BASE_VOCAB:
        assert model.config.vocab_size == BASE_VOCAB + n_new, (
            f"base vocab {model.config.vocab_size} is neither frozen "
            f"{BASE_VOCAB} nor merged {BASE_VOCAB + n_new}"
        )
        model.resize_token_embeddings(BASE_VOCAB)
        model.config.vocab_size = BASE_VOCAB
    if dec:
        missing, unexpected = model.load_state_dict(dec, strict=False)
        missing = [m for m in missing if m != "lm_head.weight"]
        assert not missing, f"unexpected MISSING decoder keys: {missing[:8]}"
        assert not unexpected, f"unexpected EXTRA decoder keys: {unexpected[:8]}"
        print(f"[merge] overlaid {len(dec)} fine-tuned decoder tensors")
    else:
        print("[merge] decoder frozen; base weights used as-is")
    assert model.config.tie_word_embeddings, "expected tied embeddings"
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    emb.data[BASE_VOCAB:BASE_VOCAB + n_new] = new_embed.to(emb.dtype)
    assert model.get_output_embeddings().weight.data_ptr() == emb.data_ptr(), "tie broke"

    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged SmolLM3 -> {out}")
    model.to(save_dtype).save_pretrained(out)
    tok.save_pretrained(out)
    torch.save(xattn, out / "xattn.pt")
    print(f"[merge] saved xattn.pt ({len(xattn)} tensors)")
    print("[merge] DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="dir with trainable.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-dataset", required=True, help="only used to pick the POV token set")
    ap.add_argument("--base", default="local/SmolLM3-3B")
    ap.add_argument("--arch", choices=["llava", "flamingo"], default="llava")
    ap.add_argument("--save-dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="Output checkpoint dtype.")
    a = ap.parse_args()
    fn = merge if a.arch == "llava" else merge_flamingo
    save_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[a.save_dtype]
    fn(a.ckpt, a.out, a.train_dataset, base=a.base, save_dtype=save_dtype)


if __name__ == "__main__":
    main()
