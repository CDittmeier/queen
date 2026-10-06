# QUEEN inference on Apple silicon

Use PAWN-8 with the default MLX backend. This is an inference port of the released
weights: it does not train, fine-tune, or download a replacement language model.
Both inference backends retain the LC0 encoder, chess vocabulary, and all 16
trained gated cross-attention sublayers before decoder layers 0, 2, …, 30.

## The two models

| Release | Initial explanation data | Improvement rounds | Paper rating |
| --- | --- | --- | --- |
| [PAWN-8](https://huggingface.co/princeton-nlp/queen_pawn-8) | Explanations from a frontier language model | Seven after the initial checkpoint | 2697 |
| [HCE-4](https://huggingface.co/princeton-nlp/queen_hce-4) | Templates built from Stockfish's handcrafted evaluation features | Three after the initial checkpoint | 2497 |

The [paper](https://arxiv.org/pdf/2610.03695), sections 3.3 and 5/table 6, describes
the distinction. They share the same approximately 4B-parameter encoder/decoder
architecture. The suffixes identify checkpoints, not different model sizes.
These are the paper's calibrated playing-strength estimates on a Lichess-related
scale; they are not FIDE ratings or a new benchmark of this Mac port. The CLI
downloads PAWN-8. HCE-4 is not part of the validated Mac setup.

## Why these tools

Research checked October 6, 2026, against primary documentation and source:

| Tool | Fit for this inference task |
| --- | --- |
| [MLX](https://developer.apple.com/videos/play/wwdc2025/315/) and [mlx-lm](https://github.com/ml-explore/mlx-lm) | Apple silicon framework with unified memory and native GPU inference. mlx-lm already implements [SmolLM3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/smollm3.py), including its alternating RoPE/NoPE layers. This fork adds QUEEN's missing board-conditioned layers around that decoder. |
| [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html) | Runs the existing upstream encoder directly on Apple's GPU. It also supplies a reference decoder backend for verifying the MLX port. |
| [Upstream vLLM runner](https://huggingface.co/princeton-nlp/queen_pawn-8) | The released runner requires Linux/NVIDIA/CUDA. This fork uses MLX's single-request inference API without introducing a serving stack. |
| Generic SmolLM3 converters and model GUIs | Loading the decoder alone does not restore QUEEN's LC0 encoder or cross-attention. This follows from the released architecture and is why the fork needs a QUEEN-specific inference adapter. |

The selected architecture runs the unchanged FP32 LC0 encoder once per position,
transfers its 16 × 64 × 1024 hidden states into MLX, and caches each bridge's board
keys and values. Autoregressive generation then stays in MLX with the decoder's
self-attention KV cache. Decoder and bridge inference use BF16. The published
vLLM runner also casts the encoder to BF16; this Mac adapter keeps it in FP32.
The original
safetensors load directly; there is no converted weight copy or quantization.

## Setup and run

Tested on an M4 Max MacBook Pro with 128 GB unified memory and macOS 27.0.1.
Allow approximately 8.4 GiB for the model plus the Python environment. Smaller
Macs have not been validated. Use a recent Apple silicon macOS version with BF16
GPU support. The runner raises an error if MPS is unavailable.

```bash
git clone --branch mac-inference https://github.com/fmhall/queen.git
cd queen
uv sync --frozen --group mac
uv run --frozen --group mac python -m mac_inference --download
```

The CLI downloads the anonymous, pinned PAWN-8 Hugging Face release at
`5027687fac08b64b2403b13aca5b198e68294fe9` and checks every file in its published
release manifest. Weights and caches stay in ignored `.models/`; they are not
committed. Downloads can be resumed with the same command.

Examples after downloading:

```bash
# Analyze the position after these legal UCI moves, retaining real game history.
uv run --frozen --group mac python -m mac_inference \
  --moves e2e4 e7e5 g1f3 b8c6 f1c4

# Analyze any nonterminal legal position supplied as FEN.
uv run --frozen --group mac python -m mac_inference \
  --fen 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3' \
  --output .mac-results/analysis.json

# Reference backend, or deterministic greedy generation for debugging.
uv run --frozen --group mac python -m mac_inference --backend mps
uv run --frozen --group mac python -m mac_inference --temperature 0
```

`--history history.json` accepts chronological prior FENs, excluding the current
FEN. A FEN alone cannot recover game history. `--moves` builds that history for
you. Default generation uses the published prompt, temperature 0.6, top-k 20,
top-p 0.95, 2048 output tokens, and the evaluation's per-game/per-ply seed hash.
Sampling is not bitwise reproducible across runtimes. The runner reports its
actual backend, token limit, finish reason, timing, raw output, and legal best move.

The CLI returns a recommendation only when an explicit `BEST_MOVE` passes
piece, capture, POV, promotion, and legality checks. It reports null for truncated
or unparseable recommendations, instead of the evaluation runner's random legal
fallback. Other analysis prose and variations are model output and can be wrong.
This CLI is not a UCI engine, and the Mac port has not been rated through full games.

## Reproducible environment

The environment was upgraded and locked with the latest stable packages available
on October 6, 2026:

| Tool | Version |
| --- | --- |
| uv | 0.12.23 |
| Python | 3.14.8 |
| MLX / mlx-lm | 0.32.3 / 0.32.0 |
| PyTorch | 2.14.1 |
| Transformers | 5.19.0 |
| huggingface-hub | 1.33.0 |

`uv.lock` records the exact dependency resolution. These are inference dependencies;
the Mac dependency group does not include the upstream training stack. Existing
`requirements.txt` is preserved for the upstream workflow.

## Verification

```bash
uv lock --check
uv pip check
uv run --frozen --group mac --group dev ruff check mac_inference tests/apple
uv run --frozen --group mac --group dev ruff format --check mac_inference tests/apple
uv run --frozen --group mac --group dev pytest -q
uv run --frozen --group mac python -m mac_inference.validate \
  --output .mac-results/validation.json
```

The small tests cover cached cross-attention against the upstream implementation,
MLX/PyTorch bridge agreement and board-cache replacement, black POV, en passant,
castling, underpromotion, illegal or incomplete moves, and the published seed.
The weight-backed validator checks release SHA-256 hashes, finite encoder states,
full-decoder logits against the upstream Flamingo forward, board conditioning,
and a cached next-token step against a complete forward pass. It checks both the
initial position and a black-to-move position with five prior FENs in FP32 to
isolate architecture/layout errors from BF16 rounding.

On the tested Mac, maximum absolute MLX/upstream logit differences were below
0.000063; the MPS hooked decoder matched the upstream forward exactly. MLX's
cached next-token comparisons were below 0.000056. Seventeen small tests passed.

Complete BF16 MLX smoke runs at the default sampled settings produced:

| Position | Recommendation | Output tokens | Generation time | Tokens/second |
| --- | --- | --- | --- | --- |
| Initial position | Nf3 (`g1f3`) | 542 | 10.03 s | 54.0 |
| Italian Game, Black to move, five prior FENs | Nf6 (`g8f6`) | 709 | 12.35 s | 57.4 |

Both stopped on EOS and recommended a legal move. Model-loading time was about
0.68 seconds and board encoding/cache preparation about 0.14–0.15 seconds in
these warm-file-cache runs. MLX's reported peak allocation was 7.23 GB; that
counter excludes PyTorch's encoder allocations and is not total process memory.
These are local smoke measurements, not a representative strength or speed
benchmark.

In a separate fixed-length comparison using the same initial position, prompt,
BF16 weights, greedy decoding, and 128 output tokens, MLX generated at 55.1
tokens/second and MPS at 35.4 tokens/second (about 1.56×). Each backend ran alone,
and rates include prompt processing but exclude model loading and board encoding.
Both runs reached the token limit before the explicit recommendation field;
they therefore reported no best move. This short comparison supports choosing
MLX for this Mac; it does not establish performance on all positions or devices.
