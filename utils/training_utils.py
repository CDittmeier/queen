"""Initialization and collation utilities for chess-LM training."""
import functools
import json
import os

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from models.encoder import Lc0Bt4HFModel
from models import FlamingoChessLM, LLaVAChessLM
from utils.instance_format import (
    KEY_EXTRA,
    KEY_FEN,
    KEY_HISTORY,
    KEY_PROMPT,
    KEY_RESPONSE,
    to_standard_instance,
    tokenize_instance,
)
from utils.lc0_planes import encode_fen_batch
from utils.special_tokens import (
    maybe_add_special_tokens,
    maybe_init_special_token_embeddings,
)
from utils.utils import turn_tensor


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def _arrow_text_lengths(dataset, columns: tuple[str, ...]) -> np.ndarray:
    """Cheap UTF-8 byte-length proxy without materializing Python strings."""
    total = np.zeros(len(dataset), dtype=np.int32)
    for name in columns:
        parts = []
        for arr in dataset.data.column(name).chunks:
            # Arrow string offsets are int32; large_string offsets are int64.
            offset_dtype = np.int64 if str(arr.type) == "large_string" else np.int32
            offsets = np.frombuffer(arr.buffers()[1], dtype=offset_dtype,
                                    count=arr.offset + len(arr) + 1)
            offsets = offsets[arr.offset:arr.offset + len(arr) + 1]
            parts.append(np.diff(offsets).astype(np.int32, copy=False))
        total += np.concatenate(parts)
    return total


class LengthGroupedSampler(Sampler[int]):
    """Shuffle globally, then make similarly sized batches inside random pools.

    Groups contain one batch per distributed rank. Shuffling groups keeps update
    steps well mixed while adjacent rank batches have comparable sequence
    lengths, avoiding stragglers at FSDP collectives.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, pool_size: int,
                 seed: int, world_size: int, longest_first: bool = False,
                 shortest_first: bool = False, start_percentile: float | None = None):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = max(pool_size, batch_size * world_size)
        self.seed = seed
        self.world_size = world_size
        self.longest_first = longest_first
        self.shortest_first = shortest_first
        self.start_percentile = start_percentile
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.longest_first:
            return iter(np.argsort(self.lengths, kind="stable")[::-1].tolist())
        if self.shortest_first:
            return iter(np.argsort(self.lengths, kind="stable").tolist())
        if self.start_percentile is not None:
            ordered = np.argsort(self.lengths, kind="stable")
            start = int(len(ordered) * self.start_percentile / 100.0)
            return iter(ordered[start:].tolist())
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self), generator=generator).numpy()
        sync_group = self.batch_size * self.world_size
        result = []
        for start in range(0, len(order), self.pool_size):
            pool = order[start:start + self.pool_size]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            groups = [pool[i:i + sync_group] for i in range(0, len(pool), sync_group)]
            group_order = torch.randperm(len(groups), generator=generator).tolist()
            for i in group_order:
                result.extend(groups[i].tolist())
        return iter(result)

def collate_fn(batch: list[dict], *, tokenizer, max_seq_len: int) -> dict:
    """Collate standardized {fen, history, prompt, response, extra} rows.

    Tokenizes prompt+response with the prompt span masked to -100 and EOS
    appended (instance_format.tokenize_instance), right-pads, and builds the lc0
    input planes from fen + history. ``extra`` is carried for eval only and is
    not forwarded to the model.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cap = max_seq_len or None
    std = [to_standard_instance(ex) for ex in batch]
    toks = [tokenize_instance(tokenizer, s[KEY_PROMPT], s[KEY_RESPONSE], max_length=cap) for s in std]

    B = len(toks)
    max_len = max(len(ids) for ids, _ in toks)
    input_ids  = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask  = torch.zeros(B, max_len,          dtype=torch.long)
    labels_out = torch.full((B, max_len), -100,   dtype=torch.long)
    for i, (ids, labs) in enumerate(toks):
        L = len(ids)
        input_ids [i, :L] = torch.tensor(ids,  dtype=torch.long)
        attn_mask [i, :L] = 1
        labels_out[i, :L] = torch.tensor(labs, dtype=torch.long)

    fens      = [s[KEY_FEN] for s in std]
    histories = [s[KEY_HISTORY] or None for s in std]
    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels_out,
        "planes":         encode_fen_batch(fens, histories),
        "turn":           turn_tensor(fens),
        "extra":          [s[KEY_EXTRA] for s in std],
    }


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_model_and_tokenizer(args):
    """Load the chess-LM (flamingo / llava) + lc0 encoder + tokenizer.

    Whether the chess answer tokens are added is decided automatically: if the
    tokenizer already contains them (a chess-resized model) nothing is added
    (n_new_tokens=0); otherwise they are added here and new embeddings are
    trained for them (see utils/special_tokens.py). The token set (POV vs
    board-absolute) follows ``args.pov``.
    """
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_path, local_files_only=True)
    orig_vocab = len(tokenizer)
    n_new_tokens = maybe_add_special_tokens(tokenizer, args)  # 0 if already present

    arch = getattr(args, "arch", "flamingo")
    print(f"Loading model (arch={arch})...")
    if arch == "flamingo":
        x_attn_kwargs = {
            "alpha_init":   getattr(args, "alpha_init",   1.0),
            "wo_rand_init": getattr(args, "wo_rand_init", False),
        }
        model = FlamingoChessLM.from_pretrained(
            args.decoder_path, n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", -1),
            x_attn_kwargs=x_attn_kwargs,
            frozen_vocab=orig_vocab,
            device=device, torch_dtype=amp_dtype, local_files_only=True,
        )
    elif arch == "llava":
        hf_extra = {}
        attn_impl = getattr(args, "attn_implementation", None)
        if attn_impl:
            hf_extra["attn_implementation"] = attn_impl
        model = LLaVAChessLM.from_pretrained(
            args.decoder_path, n_new_tokens=n_new_tokens,
            lora_rank=getattr(args, "lora_rank", 0),
            frozen_vocab=orig_vocab,
            device=device, torch_dtype=amp_dtype, local_files_only=True,
            **hf_extra,
        )
    else:
        raise NotImplementedError(f"arch={arch!r} not yet implemented")

    assert orig_vocab == model.decoder.config.vocab_size, (
        f"Tokenizer base size ({orig_vocab}) != decoder vocab "
        f"({model.decoder.config.vocab_size}); the tokenizer must match the model."
    )

    # Loss path selector: chunked CE by default (works under torch.compile);
    # Liger fused CE opt-in for stage-5 memory pressure. Left explicit rather
    # than derived from `compile`: the chunked path has a different memory
    # profile, so no config silently changes loss kernel.
    model.use_liger_loss = getattr(args, "use_liger_loss", False)

    print("Loading LC0 encoder...")
    encoder = Lc0Bt4HFModel.from_pretrained(args.encoder_path, local_files_only=True)
    encoder.to(device=device, dtype=amp_dtype).eval()

    maybe_init_special_token_embeddings(model, tokenizer, args)  # no-op unless flag set

    model.train()
    # LLaVA has no fp32-sensitive alpha gates, so by default its unfrozen decoder is kept in the
    # requested dtype (FSDP then holds no fp32 copy of the 3B params). BUT at lr 1e-5 a bf16 weight
    # cannot represent the ~1e-6 update — it rounds away and most of the net stops learning. Setting
    # fp32_master_weights keeps trainable params in fp32 (compute stays bf16 via autocast), the
    # standard mixed-precision path that recovers full training quality at the cost of the fp32 copy.
    if arch == "llava" and not getattr(args, "fp32_master_weights", False):
        model.to(dtype=amp_dtype)
    else:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if arch == "flamingo":
        model.x_attn_layers.train()
    # Keep the decoder in eval mode when fully frozen (lora_rank < 0).
    if model.lora_rank < 0:
        model.decoder.eval()

    return model, encoder, tokenizer


def init_datasets_and_dataloader(args, tokenizer):
    """Load the HF (Arrow) train + eval datasets; return (train DataLoader, eval dataset)."""
    print(f"Loading train dataset from {args.train_dataset}...")
    train_ds = load_from_disk(args.train_dataset)
    print(f"  train: {len(train_ds)} examples")
    print(f"Loading eval dataset from {args.eval_dataset}...")
    eval_ds  = load_from_disk(args.eval_dataset)
    print(f"  eval:  {len(eval_ds)} examples")

    cfn = functools.partial(collate_fn, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    sampler = None
    shuffle = True
    length_grouping = getattr(args, "length_grouping", False)
    if "BENCH_LENGTH_GROUPING" in os.environ:
        length_grouping = os.environ["BENCH_LENGTH_GROUPING"] == "1"
    if length_grouping:
        lengths = _arrow_text_lengths(train_ds, (KEY_PROMPT, KEY_RESPONSE))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        sampler = LengthGroupedSampler(
            lengths, batch_size=args.batch_size,
            pool_size=getattr(args, "length_group_pool_size", 512),
            seed=args.seed, world_size=world_size,
            longest_first=os.environ.get("BENCH_LONGEST_BATCHES") == "1",
            shortest_first=os.environ.get("BENCH_SHORTEST_BATCHES") == "1",
            start_percentile=(float(os.environ["BENCH_START_PERCENTILE"])
                              if "BENCH_START_PERCENTILE" in os.environ else None),
        )
        shuffle = False
        print(f"Length grouping enabled (pool={sampler.pool_size}, proxy range="
              f"{lengths.min()}..{lengths.max()} bytes)")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        collate_fn=cfn, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    return train_loader, eval_ds


def post_eval(args, model) -> None:
    """Restore train mode after eval. Mirrors init_model_and_tokenizer's
    train/eval setup so eval cycles don't leave the model in a wrong mode."""
    model.train()
    raw = getattr(model, "_orig_mod", model)
    if getattr(args, "arch", "flamingo") == "flamingo":
        raw.x_attn_layers.train()
    # Keep decoder in eval when fully frozen; LoRA / full-train need train mode.
    if raw.lora_rank < 0:
        raw.decoder.eval()


def init_optimizer_and_scheduler(args, model):
    decoder_lr = getattr(args, "decoder_lr", None)
    embed_lr   = getattr(args, "embed_lr",   None)
    fused_optimizer = getattr(args, "fused_optimizer", False)
    if "BENCH_FUSED_OPTIMIZER" in os.environ:
        fused_optimizer = os.environ["BENCH_FUSED_OPTIMIZER"] == "1"
    optimizer = torch.optim.AdamW(
        model.param_groups(args.lr, decoder_lr=decoder_lr, embed_lr=embed_lr),
        weight_decay=args.weight_decay,
        fused=fused_optimizer and torch.cuda.is_available(),
    )
    warmup_steps = int(getattr(args, "warmup_ratio", 0.0) * args.n_steps)
    if args.scheduler == "cosine":
        min_lr_rate = getattr(args, "min_lr_rate", 0.0)
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer, warmup_steps, args.n_steps, min_lr_rate=min_lr_rate,
        )
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.n_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    return optimizer, scheduler


def _load_dataset_pov(train_dataset_dir: str) -> bool:
    """Read 'pov' from <train_dataset_dir>/dataset_config.json (single source of truth)."""
    with open(os.path.join(train_dataset_dir, "dataset_config.json")) as f:
        return bool(json.load(f)["pov"])


def initialize_training_objects(args):
    """Top-level init. Returns everything the training loop needs."""
    args.pov = _load_dataset_pov(args.train_dataset)
    print(f"Dataset mode: {'POV-relative' if args.pov else 'board-absolute'} (args.pov={args.pov})")
    model, encoder, tokenizer = init_model_and_tokenizer(args)
    train_loader, eval_ds     = init_datasets_and_dataloader(args, tokenizer)
    optimizer, scheduler      = init_optimizer_and_scheduler(args, model)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]
    return model, encoder, tokenizer, train_loader, eval_ds, optimizer, scheduler, amp_dtype
