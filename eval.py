"""Standalone test-set eval for a stage-1/stage-2 checkpoint.

Rebuilds the model exactly as training does (base decoder/encoder + runtime POV
tokens + side embeddings), overlays the trained params from the checkpoint's
portable `trainable.pt` (a flat named-parameter dict -> load_state_dict
strict=False), and runs the generative grader over a test arrow.

    python eval.py \
        --config configs/train/stage1/llava.yaml \
        --ckpt   runs/stage1_llava_frozen_cos_60k/step_0040000 \
        --test-dataset data/stage2/test_dataset.arrow \
        --out    runs/eval/stage2_step40k.json [--max-examples N]
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from datasets import load_from_disk

from utils.eval_utils import run_eval
from utils.training_utils import _load_dataset_pov, init_model_and_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--regret", action="store_true",
                    help="also compute stage-5 regret metrics via the Stockfish oracle")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override cfg eval_max_new_tokens (stage-5 narratives need ~4096)")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    args = SimpleNamespace(
        arch=cfg["arch"], lora_rank=cfg.get("lora_rank", 0),
        decoder_path=cfg["decoder_path"], encoder_path=cfg["encoder_path"],
        dtype=cfg.get("dtype", "bfloat16"), device="cuda",
        embed_init=cfg.get("embed_init", "semantic"),
        alpha_init=1.0, wo_rand_init=False,
        train_dataset=cfg["train_dataset"],
    )
    args.pov = _load_dataset_pov(args.train_dataset)
    model, encoder, tokenizer = init_model_and_tokenizer(args)

    sd = torch.load(Path(a.ckpt) / "trainable.pt", map_location="cpu")
    # torch.compile adds an `_orig_mod.` prefix (OptimizedModule wrapper) and
    # activation checkpointing inserts a `._checkpoint_wrapped_module` infix per
    # wrapped submodule; the eval model has neither, so strip both to recover the
    # plain parameter names.
    def _clean(k: str) -> str:
        return k.removeprefix("_orig_mod.").replace("._checkpoint_wrapped_module", "")
    sd = {_clean(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    overlaid = [k for k in sd if k not in unexpected]
    print(f"[load] trainable.pt tensors={len(sd)} overlaid={len(overlaid)} "
          f"unexpected={len(unexpected)}")
    assert not unexpected, f"unexpected keys in trainable.pt: {unexpected[:10]}"
    assert overlaid, "no checkpoint tensor matched a model parameter!"

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    test_ds = load_from_disk(a.test_dataset)
    print(f"[data] test set: {len(test_ds)} examples; pov={args.pov}")

    metrics, samples = run_eval(
        model, encoder, test_ds, tokenizer, torch.device("cuda"), amp_dtype,
        eval_batch_size=cfg.get("eval_batch_size", 8),
        eval_max_new_tokens=a.max_new_tokens or cfg.get("eval_max_new_tokens", 384),
        pov=args.pov, temperature=a.temperature, top_k=20, top_p=0.95,
        max_examples=a.max_examples,
    )

    if a.regret:
        from utils.stage5_eval import compute_metrics
        metrics.update(compute_metrics(samples))

    print(json.dumps(metrics, indent=2))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": a.ckpt, "n": len(samples), "metrics": metrics,
               "samples": samples}, open(out, "w"), indent=2)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
