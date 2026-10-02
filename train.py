import argparse
import functools
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, FullyShardedDataParallelPlugin, skip_first_batches
from accelerate.utils import DataLoaderConfiguration, GradientAccumulationPlugin
from transformers import AutoConfig

from models.adapters import layer_cls_for_config

from utils.training_utils import initialize_training_objects, post_eval, collate_fn
from utils.eval_utils import run_eval
from utils.utils import encode_planes

try:
    import wandb
except ImportError:  # optional
    wandb = None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_trainable(accelerator, model, out_path: Path) -> None:
    # Small, portable artifact: just the trainable parameters (the bridge + new
    # embeddings, plus the decoder when it is unfrozen), gathered to full tensors
    # on the main process. The sharded save_state is for resuming training; this
    # is what you load for inference / eval elsewhere. full_tensor() is an
    # all-gather, so every rank walks the same params; only rank 0 keeps them.
    raw = accelerator.unwrap_model(model)
    sd = {}
    for name, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        t = p.data
        if hasattr(t, "full_tensor"):  # FSDP2 shards each param as a DTensor
            t = t.full_tensor()
        if accelerator.is_main_process:
            sd[name] = t.detach().to("cpu")
    if accelerator.is_main_process:
        torch.save(sd, out_path)
    accelerator.wait_for_everyone()


def save_checkpoint(accelerator, model, tokenizer, run_dir: Path, step: int) -> Path:
    # save_state: FSDP-sharded model + optimizer + scheduler + RNG + dataloader
    # position, for exact resume. save_trainable: the small portable trainable-
    # only state. The step lives in a sidecar.
    ckpt_dir = run_dir / f"step_{step:07d}"
    # Return the caching allocator's reserved pool to the driver before the
    # collective save. dist_cp's plan gather (gather_object) allocates its NCCL
    # buffer via raw cudaMalloc, outside PyTorch's allocator. After a memory-heavy
    # eval the reserved high-water — set by the full-seq (B,S,V) logits, with
    # V=262K on Gemma — leaves ~0 driver-free memory, so that cudaMalloc fails as
    # "NCCL Error 1: unhandled cuda error". Freeing here restores the headroom.
    # Runs on all ranks (save_state is collective), so every rank frees first.
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.save_state(str(ckpt_dir))
    # Defensive barrier + reclaim between the two collective save phases. NOTE: the
    # exp4 watchdog deadlocks were ultimately traced to EVAL-time cross-rank
    # divergence, not the save — so this is insurance, not the fix for that bug.
    # Still cheap and worth keeping: it serializes save_state (DCP) and
    # save_trainable (per-param full_tensor() all-gathers) so they can't interleave
    # into a mismatched-collective deadlock, and frees headroom for the ~1.34GB
    # (Gemma) embedding full_tensor that save_trainable materializes per rank.
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    save_trainable(accelerator, model, ckpt_dir / "trainable.pt")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(str(ckpt_dir))
        (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step}))
        print(f"[step {step}] checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(accelerator, ckpt_dir: str) -> int:
    accelerator.load_state(str(ckpt_dir))
    return json.loads((Path(ckpt_dir) / "train_state.json").read_text())["step"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_metrics(step: int, metrics: dict, jsonl_file, txt_file) -> None:
    row = {"step": step, **metrics}
    jsonl_file.write(json.dumps(row) + "\n")
    jsonl_file.flush()

    txt_file.write(f"step={step}\n")
    for k, v in metrics.items():
        txt_file.write(f"  {k:<40s} {v:.4f}\n")
    txt_file.write("\n")
    txt_file.flush()

    print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def save_generations(ckpt_dir: Path, samples: list[dict]) -> None:
    with open(ckpt_dir / "generations.json", "w") as f:
        json.dump(samples, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a chess-LM (config-driven)")
    parser.add_argument("--config", default=None,
                        help="YAML config. Precedence: CLI > YAML > argparse defaults.")

    # --- reproducibility ---
    g = parser.add_argument_group("reproducibility")
    g.add_argument("--seed", type=int, default=42)

    # --- training ---
    g = parser.add_argument_group("training")
    g.add_argument("--batch-size",       type=int,   default=256)
    g.add_argument("--max-seq-len",      type=int,   default=256)
    g.add_argument("--n-steps",          type=int,   default=10_000)
    g.add_argument("--lr",               type=float, default=1e-4)
    g.add_argument("--decoder-lr",       type=float, default=None,
                   help="LR for decoder LoRA group; defaults to --lr if not set")
    g.add_argument("--embed-lr",         type=float, default=None,
                   help="LR for the new_embed (special-token) param group; defaults to lr*0.1 if not set")
    g.add_argument("--grad-accum-steps", type=int,   default=1)
    g.add_argument("--max-grad-norm",    type=float, default=1.0)

    # --- optimizer / scheduler ---
    g = parser.add_argument_group("optimizer")
    g.add_argument("--weight-decay",  type=float, default=0.01)
    g.add_argument("--scheduler",     choices=["constant", "cosine", "linear"], default="constant")
    g.add_argument("--warmup-ratio",  type=float, default=0.05,
                   help="Fraction of n_steps used for linear warmup (cosine/linear schedulers only)")
    g.add_argument("--min-lr-rate",   type=float, default=0.0,
                   help="Cosine LR floor as a fraction of peak LR (e.g. 0.1 -> decay stops at 10%% of peak). Cosine only.")

    # --- eval ---
    g = parser.add_argument_group("eval")
    g.add_argument("--eval-freq",           type=int,   default=500)
    g.add_argument("--save-freq",           type=int,   default=None,
                   help="Checkpoint cadence in steps; falls back to eval_freq when unset.")
    g.add_argument("--eval-loss-only",      action="store_true", default=False,
                   help="Eval = teacher-forced val loss only (skip the generation eval). "
                        "For runs whose task has no grader, e.g. stage-5 hce_rationale.")
    g.add_argument("--eval-batch-size",     type=int,   default=64)
    g.add_argument("--eval-max-new-tokens", type=int,   default=128)
    g.add_argument("--eval-max-examples",   type=int,   default=1900,
                   help="Cap eval dataset size (default 1900 = 25 positions × 76 examples)")
    g.add_argument("--log-samples",         type=int,   default=4)
    g.add_argument("--eval-at-start",       action="store_true", default=False)
    g.add_argument("--temperature",         type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy")
    g.add_argument("--top-k",               type=int,   default=20)
    g.add_argument("--top-p",               type=float, default=0.95)

    # --- model ---
    g = parser.add_argument_group("model")
    g.add_argument("--arch",      choices=["flamingo", "llava"], default="flamingo")
    g.add_argument("--lora-rank", type=int, default=-1,
                   help="LoRA rank: <0 = frozen decoder, 0 = full fine-tuning, >0 = LoRA adapters")
    g.add_argument("--decoder-path", default=None)
    g.add_argument("--encoder-path", default=None)
    g.add_argument("--alpha-init",    type=float, default=1.0,
                   help="Initial alpha value (pre-tanh). 1.0 (default) → tanh≈0.76 gate ~76% open. "
                        "0.0 = original Flamingo (gate closed).")
    g.add_argument("--wo-rand-init", action="store_true", default=False,
                   help="Use random init for W_O (default: zero-init).")
    g.add_argument("--dtype",        choices=["bfloat16", "float16", "float32"],
                   default="bfloat16")
    g.add_argument("--device",       default="cuda")
    g.add_argument("--compile",      action="store_true", default=False,
                   help="torch.compile the model (faster steady-state, ~2min cold-start)")

    # --- special tokens (added automatically when the tokenizer lacks them) ---
    g = parser.add_argument_group("special tokens")
    g.add_argument("--embed-init", choices=["semantic", "random"], default="semantic",
                   help="Init for added token embeddings (used only when tokens are added)")

    # --- data ---
    g = parser.add_argument_group("data")
    g.add_argument("--train-dataset", default=None, help="Path to HF Arrow train dataset dir")
    g.add_argument("--eval-dataset",  default=None, help="Path to HF Arrow eval dataset dir")
    g.add_argument("--num-workers",   type=int, default=4)

    # --- output ---
    g = parser.add_argument_group("output")
    g.add_argument("--exp-name",    default=None, help="Experiment name; outputs go to runs/{exp_name}/")
    g.add_argument("--output-dir",  default="chesslm/runs/")
    g.add_argument("--resume-from", default=None, help="Path to a checkpoint dir to resume from")
    g.add_argument("--stop-file", default=None,
                   help="Checkpoint and exit at an optimizer-step boundary when this file exists.")
    g.add_argument("--init-from", default=None,
                   help="Warm-start: load this checkpoint's trainable.pt (weights only, "
                        "strict=False) into a fresh run. Unlike --resume-from it does NOT "
                        "restore the optimizer/scheduler/step. Ignored when --resume-from is set.")

    args = parser.parse_args()
    if args.config:
        import yaml
        # Comparing parsed values to defaults cannot detect an explicit CLI
        # value that happens to equal its argparse default (e.g.
        # --grad-accum-steps 1). Inspect option presence instead.
        argv = sys.argv[1:]
        explicit = {
            action.dest
            for action in parser._actions
            if any(arg == opt or arg.startswith(opt + "=")
                   for opt in action.option_strings for arg in argv)
        }
        for k, v in (yaml.safe_load(open(args.config)) or {}).items():
            if k not in explicit:
                setattr(args, k, v)
    missing = [k for k in ("decoder_path", "encoder_path", "train_dataset", "eval_dataset", "exp_name")
               if getattr(args, k, None) is None]
    if missing:
        parser.error("missing required key(s) (set via --config or CLI): " + ", ".join(missing))
    return args


def get_diagnostics(model) -> dict[str, float]:
    """Unwraps torch.compile's OptimizedModule wrapper before delegating."""
    raw = getattr(model, "_orig_mod", model)
    return raw.get_diagnostics()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    verbose = os.environ.get("TEST_MODE") == "1"

    # --- throughput-benchmark hooks (env-driven; all no-ops in normal runs) ---
    # BENCH_STEPS>0 times that many optimizer steps (after BENCH_WARMUP) then
    # exits before any eval/checkpoint. BENCH_RESHARD/BENCH_ATTN/BENCH_COMPILE
    # override the FSDP reshard, decoder attention impl, and torch.compile knobs.
    bench_steps  = int(os.environ.get("BENCH_STEPS", "0"))
    bench_warmup = int(os.environ.get("BENCH_WARMUP", "3"))
    stop_after_steps = int(os.environ.get("STOP_AFTER_STEPS", "0"))
    if "BENCH_ATTN" in os.environ:
        args.attn_implementation = os.environ["BENCH_ATTN"]
    if os.environ.get("BENCH_COMPILE") == "1":
        args.compile = True
    reshard_after_forward = getattr(args, "reshard_after_forward", True)
    if "BENCH_RESHARD" in os.environ:
        reshard_after_forward = os.environ["BENCH_RESHARD"] == "1"
    activation_checkpointing = getattr(args, "activation_checkpointing", False)
    if "BENCH_ACTIVATION_CHECKPOINTING" in os.environ:
        activation_checkpointing = os.environ["BENCH_ACTIVATION_CHECKPOINTING"] == "1"
    checkpoint_every_n = int(os.environ.get(
        "BENCH_CHECKPOINT_EVERY_N",
        getattr(args, "activation_checkpointing_every_n", 1),
    ))
    sync_each_batch = getattr(args, "sync_each_batch", True)
    if "BENCH_SYNC_EACH_BATCH" in os.environ:
        sync_each_batch = os.environ["BENCH_SYNC_EACH_BATCH"] == "1"
    if bench_steps:
        args.exp_name = "_bench"  # throwaway run dir; never touches a real run

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]

    # --- accelerate + FSDP2 ---
    # Wrap each decoder layer as its own unit; flamingo also wraps its DenseXAttn
    # bridge. llava has no such class (its connector rides in the root unit), so
    # gate on arch — wrapping a missing class aborts FSDP setup.
    # Resolved from config (not model) because the FSDP plugin is built before the model.
    _dec_cfg = AutoConfig.from_pretrained(args.decoder_path, local_files_only=True)
    wrap_classes = [layer_cls_for_config(_dec_cfg)]
    if args.arch == "flamingo":
        wrap_classes.append("DenseXAttn")
    # Preserve FP32 optimizer/master shards while casting gathered parameters to
    # the compute dtype. Without an FSDP mixed-precision policy, autocast makes
    # the matmuls BF16 but every layer still all-gathers FP32 parameters.
    reduce_dtype_name = os.environ.get(
        "BENCH_FSDP_REDUCE_DTYPE", getattr(args, "fsdp_reduce_dtype", "float32"))
    reduce_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[reduce_dtype_name]
    use_fsdp_mixed_precision = getattr(
        args, "fsdp_mixed_precision", getattr(args, "fp32_master_weights", False))
    if "BENCH_FSDP_MIXED_PRECISION" in os.environ:
        use_fsdp_mixed_precision = os.environ["BENCH_FSDP_MIXED_PRECISION"] == "1"
    if not use_fsdp_mixed_precision:
        mixed_precision_policy = None
    else:
        mixed_precision_policy = {
            "param_dtype": amp_dtype,
            "reduce_dtype": reduce_dtype,
            "output_dtype": amp_dtype,
        }
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=wrap_classes,
        reshard_after_forward=reshard_after_forward,
        activation_checkpointing=activation_checkpointing and checkpoint_every_n == 1,
        mixed_precision_policy=mixed_precision_policy,
        state_dict_type="SHARDED_STATE_DICT",
    )
    # no_sync retains full gradients across accumulation. With BF16 gathered
    # parameters and selective checkpointing this fits the measured 8192-token
    # worst case, while avoiding a reduction after every microbatch.
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.grad_accum_steps, sync_each_batch=sync_each_batch),
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        fsdp_plugin=fsdp_plugin,
    )
    device = accelerator.device
    args.device = str(device)  # build the model + encoder on this rank's GPU

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.output_dir) / args.exp_name
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics.jsonl"
    txt_path   = run_dir / "metrics.txt"

    t0 = time.time()
    (model, encoder, tokenizer, train_loader, eval_ds,
     optimizer, scheduler, amp_dtype) = initialize_training_objects(args)

    # Warm-start (e.g. stage-2 from a stage-1 checkpoint): overlay the portable
    # trainable.pt weights onto the freshly-built, still-unsharded model, then
    # train from step 0 with a fresh optimizer/scheduler. Distinct from
    # --resume-from (full state restore); skipped when resuming so self-chained
    # slots continue from their own stage-2 checkpoints.
    init_from = getattr(args, "init_from", None)
    if init_from and not args.resume_from:
        sd = torch.load(Path(init_from) / "trainable.pt", map_location="cpu")
        # Portable checkpoints may have been saved while selective activation
        # checkpointing wrapped decoder layers.  Those wrappers are an
        # execution detail, not part of the model's parameter names.
        sd = {
            k.removeprefix("_orig_mod.").replace("._checkpoint_wrapped_module", ""): v
            for k, v in sd.items()
        }
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in init_from trainable.pt: {unexpected[:10]}"
        assert sd, "init_from trainable.pt is empty — nothing to overlay!"
        accelerator.print(f"[init_from] loaded {len(sd)} tensors from {init_from} "
                          f"(missing={len(missing)} kept from base init)")

    # Accelerate checkpoints every wrapped decoder layer. On 80GB GPUs a
    # half-rate policy is a useful middle ground: it retains enough activation
    # headroom for the 8k tail without recomputing all 36 layers.
    if activation_checkpointing and checkpoint_every_n > 1:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        targets = [(name, module) for name, module in model.named_modules()
                   if module.__class__.__name__ == wrap_classes[0]]
        for i, (name, module) in enumerate(targets):
            if i % checkpoint_every_n:
                continue
            parent_name, child_name = name.rsplit(".", 1)
            model.get_submodule(parent_name).register_module(
                child_name, checkpoint_wrapper(module, preserve_rng_state=False))
        accelerator.print(f"Activation checkpointing {len(targets[::checkpoint_every_n])}/"
                          f"{len(targets)} decoder layers (every {checkpoint_every_n})")

    # Shard the model + optimizer + scheduler + dataloader across ranks. The
    # frozen encoder is not trained, so it stays replicated (not prepared).
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    trainable_dtypes = {}
    for p in model.parameters():
        if p.requires_grad:
            trainable_dtypes[str(p.dtype)] = trainable_dtypes.get(str(p.dtype), 0) + p.numel()
    accelerator.print(
        f"FSDP policy: mixed_precision={mixed_precision_policy} "
        f"sync_each_batch={sync_each_batch} reshard_after_forward={reshard_after_forward} "
        f"activation_checkpointing={activation_checkpointing}/every_{checkpoint_every_n} "
        f"master_params={trainable_dtypes}")
    accelerator.print(f"Initialization complete ({time.time() - t0:.1f}s)")
    if accelerator.is_main_process:
        tokenizer.save_pretrained(str(run_dir))  # self-contained: POV tokens + pinned-date template
    use_wandb = wandb is not None and accelerator.is_main_process and os.environ.get("WANDB_DISABLED") != "1"
    if use_wandb:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "chesslm-stage1"), name=args.exp_name,
                   id=args.exp_name, resume="allow", config={k: str(v) for k, v in vars(args).items()})

    if args.compile:
        compile_mode = os.environ.get("BENCH_COMPILE_MODE", getattr(args, "compile_mode", "default"))
        accelerator.print(f"Compiling model with torch.compile (mode={compile_mode})...")
        t0 = time.time()
        model = torch.compile(model, mode=compile_mode)
        accelerator.print(f"Compilation done ({time.time() - t0:.1f}s)")

    # Validation batches have independently varying sequence lengths on every
    # rank. In Flamingo, compiling the HF causal-mask builder in eval mode can
    # incorrectly retain the first batch's cache_position length and then fail
    # an Inductor size assertion on the next batch. Keep compiled training, but
    # use the prepared model underneath OptimizedModule for validation. This is
    # the same FSDP model and weights; only the Dynamo wrapper is bypassed.
    eval_model = getattr(model, "_orig_mod", model)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(accelerator, args.resume_from)
        accelerator.print(f"Resumed from step {start_step}")

    # Teacher-forced val loader (loss-only eval): reuse the exact training collator
    # on the eval split, sharded across ranks. One forward per example, no
    # generation, so it is cheap enough to run at every checkpoint.
    val_loader = accelerator.prepare(DataLoader(
        eval_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer, max_seq_len=args.max_seq_len),
        num_workers=args.num_workers, pin_memory=True,
    ))

    def do_eval(step: int, jsonl_file, txt_file, train_loss: float | None = None) -> None:
        # Runs on every rank — generation forwards trigger FSDP all-gathers, so
        # all ranks must participate — but only the main process logs / saves.
        accelerator.print(f"[eval] starting eval at step {step}...")
        t_eval = time.time()
        eval_model.eval()
        metrics, samples = run_eval(
            eval_model, encoder, eval_ds, tokenizer, device, amp_dtype,
            args.eval_batch_size, args.eval_max_new_tokens,
            pov=args.pov,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            max_examples=args.eval_max_examples,
        )
        diag = get_diagnostics(accelerator.unwrap_model(eval_model))
        full_metrics = ({"train_loss": train_loss} if train_loss is not None else {}) | metrics | diag
        if accelerator.is_main_process:
            step_dir = run_dir / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            if getattr(args, "extra_eval_file", None):   # flag-gated external val metric(s)
                import importlib.util
                _sp = importlib.util.spec_from_file_location("_extra_eval", args.extra_eval_file)
                _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
                full_metrics.update(_m.compute_metrics(samples, args))
            save_generations(step_dir, samples)
            log_metrics(step, full_metrics, jsonl_file, txt_file)
            if use_wandb:
                wandb.log(full_metrics, step=step)
            if verbose and args.log_samples > 0:
                for s in samples[:args.log_samples]:
                    print(f"  [{s['task']}] {s['prompt'][:60]} → {s['generated'][:80]}")
        accelerator.print(f"[eval] done ({time.time() - t_eval:.1f}s) — {len(samples)} generations")
        post_eval(args, accelerator.unwrap_model(eval_model))
        # Generation builds large transient KV-cache activations; the caching
        # allocator keeps those blocks reserved. The next big allocation is the
        # FSDP checkpoint save's all-gather, which can fail to allocate on the
        # fragmented/near-full pool and surface as "NCCL Error 1: unhandled cuda
        # error" inside the save collective. Return the cached blocks to CUDA and
        # resync ranks before saving. Cheap, runs only at eval cadence.
        if torch.cuda.is_available():
            rsv = torch.cuda.memory_reserved(device) / 1e9
            alloc = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_reserved(device) / 1e9
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            rsv2 = torch.cuda.memory_reserved(device) / 1e9
            accelerator.print(
                f"[eval] mem: reserved {rsv:.1f}->{rsv2:.1f} GB  allocated {alloc:.1f} GB  peak_reserved {peak:.1f} GB")
            torch.cuda.reset_peak_memory_stats(device)
        accelerator.wait_for_everyone()

    def do_eval_loss(step: int, jsonl_file, txt_file, train_loss: float | None = None) -> None:
        # Teacher-forced validation loss (token-weighted, reduced across ranks).
        # All ranks run the forwards (FSDP all-gathers need every rank); only the
        # main process logs. Used when the task has no grader — loss is the signal.
        accelerator.print(f"[eval] computing val loss at step {step}...")
        t_eval = time.time()
        eval_model.eval()
        tot_loss = torch.zeros((), device=device)
        tot_tok  = torch.zeros((), device=device)
        with torch.no_grad():
            for batch in val_loader:
                enc_hidden = encode_planes(
                    encoder, batch["planes"], amp_dtype, pov=args.pov, turn=batch["turn"])
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = eval_model(batch["input_ids"], enc_hidden,
                                      batch["attention_mask"], labels=batch["labels"])
                ntok = (batch["labels"] != -100).sum()
                tot_loss += loss.detach().float() * ntok
                tot_tok  += ntok
        tot_loss = accelerator.reduce(tot_loss, reduction="sum")
        tot_tok  = accelerator.reduce(tot_tok,  reduction="sum")
        val_loss = (tot_loss / tot_tok.clamp_min(1)).item()
        post_eval(args, accelerator.unwrap_model(eval_model))
        if accelerator.is_main_process:
            metrics = (({"train_loss": train_loss} if train_loss is not None else {})
                       | {"val_loss": val_loss})
            log_metrics(step, metrics, jsonl_file, txt_file)
            if use_wandb:
                wandb.log(metrics, step=step)
        accelerator.print(
            f"[eval] val_loss={val_loss:.4f} ({time.time() - t_eval:.1f}s, "
            f"{int(tot_tok.item())} sup-tokens)")
        accelerator.wait_for_everyone()

    # Loss-only eval (no grader) vs full generation eval.
    _do_eval = do_eval_loss if getattr(args, "eval_loss_only", False) else do_eval
    save_freq = args.save_freq or args.eval_freq

    # Metrics files are written by the main process only.
    jsonl_file = open(jsonl_path, "a") if accelerator.is_main_process else None
    txt_file   = open(txt_path, "a") if accelerator.is_main_process else None
    trajectory_path = os.environ.get("TRAJECTORY_LOG")
    trajectory_file = (open(trajectory_path, "w")
                       if trajectory_path and accelerator.is_main_process else None)

    if args.eval_at_start:
        _do_eval(start_step, jsonl_file, txt_file)

    # Deterministic resumption: with a seedable sampler, set_epoch(epoch) gives a
    # reproducible shuffle, so re-deriving (epoch, in-epoch offset) from the step
    # and skip_first_batches over the partial epoch replays the exact same data.
    batches_per_epoch = len(train_loader)
    global_step = start_step
    batches_done = start_step * args.grad_accum_steps
    epoch = batches_done // batches_per_epoch
    skip_batches = batches_done % batches_per_epoch

    optimizer.zero_grad()
    accelerator.print(
        f"Training started — {args.n_steps} steps, batch={args.batch_size}, "
        f"grad_accum={args.grad_accum_steps}, world_size={accelerator.num_processes}, "
        f"effective_batch={args.batch_size * args.grad_accum_steps * accelerator.num_processes}"
    )

    pbar = None
    if verbose and accelerator.is_main_process:
        from tqdm import tqdm
        pbar = tqdm(total=args.n_steps, initial=start_step, desc="train", dynamic_ncols=True)

    running_loss = torch.zeros((), dtype=torch.float32, device=device)
    last_log_time = time.time()
    log_freq = int(os.environ.get("LOG_EVERY", "10"))
    bench_t0 = None
    bench_padded_tokens = torch.zeros((), dtype=torch.long, device=device)
    bench_actual_tokens = torch.zeros((), dtype=torch.long, device=device)
    done = global_step >= args.n_steps
    stop_file = Path(args.stop_file) if args.stop_file else None
    while not done:
        train_loader.set_epoch(epoch)
        epoch_iter = train_loader
        if skip_batches:
            epoch_iter = skip_first_batches(train_loader, skip_batches)
            skip_batches = 0
        for batch in epoch_iter:
            if bench_t0 is not None:
                bench_padded_tokens += batch["input_ids"].numel()
                bench_actual_tokens += batch["attention_mask"].sum()
            # accelerate gates grad sync + optimizer step over grad_accum_steps.
            with accelerator.accumulate(model):
                enc_hidden = encode_planes(
                    encoder, batch["planes"], amp_dtype,
                    pov=args.pov, turn=batch["turn"],
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model(batch["input_ids"], enc_hidden,
                                 batch["attention_mask"], labels=batch["labels"])
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Keep loss accumulation on-device. Calling .item() for every
            # microbatch serializes the CPU with the GPU and defeats loader and
            # collective overlap; synchronize only when a step is logged.
            running_loss += loss.detach().float() / args.grad_accum_steps
            if not accelerator.sync_gradients:
                continue

            # One full optimizer step completed.
            global_step += 1
            # Handoff: rank zero checks the request; every rank saves and exits together.
            if stop_file is not None:
                requested = torch.tensor(
                    int(accelerator.is_main_process and stop_file.exists()), device=device)
                requested = accelerator.reduce(requested, reduction="sum")
                if requested.item():
                    save_checkpoint(accelerator, model, tokenizer, run_dir, global_step)
                    accelerator.wait_for_everyone()
                    accelerator.print(f"Checkpointed stop requested by {stop_file}")
                    done = True
                    break
            if trajectory_file is not None:
                trajectory_file.write(json.dumps({
                    "step": global_step,
                    "loss": running_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }) + "\n")
                trajectory_file.flush()
            if bench_steps:
                torch.cuda.synchronize(device)
                rel = global_step - start_step
                if rel == bench_warmup:
                    bench_t0 = time.time()
                    torch.cuda.reset_peak_memory_stats(device)
                    accelerator.print(f"[bench] warmup {bench_warmup} steps done; timing {bench_steps} steps")
                elif bench_t0 is not None and rel >= bench_warmup + bench_steps:
                    dt = time.time() - bench_t0
                    padded = accelerator.reduce(bench_padded_tokens, reduction="sum").item()
                    actual = accelerator.reduce(bench_actual_tokens, reduction="sum").item()
                    peak_gb = torch.cuda.max_memory_reserved(device) / 1e9
                    accelerator.print(
                        f"[bench] RESULT steps={bench_steps} elapsed={dt:.3f}s "
                        f"per_step={dt / bench_steps:.3f}s padded_tok_s={padded / dt:.0f} "
                        f"actual_tok_s={actual / dt:.0f} padding_eff={actual / max(padded, 1):.3f} "
                        f"peak_reserved_gb={peak_gb:.1f}")
                    done = True
                    break
                running_loss = torch.zeros((), device=device)
                continue
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{running_loss.item():.4f}")
            elif global_step % log_freq == 0:
                torch.cuda.synchronize(device)
                elapsed = time.time() - last_log_time
                accelerator.print(
                    f"[step {global_step}/{args.n_steps}] loss={running_loss.item():.4f} "
                    f"elapsed={elapsed:.1f}s ({elapsed / log_freq:.2f}s/step)")
                last_log_time = time.time()
            if use_wandb:
                wandb.log({"train/loss": running_loss.item()}, step=global_step)

            if global_step % args.eval_freq == 0:
                _do_eval(global_step, jsonl_file, txt_file, train_loss=running_loss.item())
            if global_step % save_freq == 0:
                # Return cached blocks before the checkpoint all-gather so the save
                # collective allocates on a clean pool (matches the eval-path cleanup).
                if torch.cuda.is_available():
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
                save_checkpoint(accelerator, model, tokenizer, run_dir, global_step)

            running_loss = torch.zeros((), device=device)
            if global_step >= args.n_steps:
                done = True
                break
            if stop_after_steps and global_step >= start_step + stop_after_steps:
                done = True
                break
        epoch += 1

    if accelerator.is_main_process:
        if jsonl_file is not None:
            jsonl_file.close()
        if txt_file is not None:
            txt_file.close()
        if trajectory_file is not None:
            trajectory_file.close()
    accelerator.print("Training complete.")

    if use_wandb:
        wandb.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
