# QUEEN

Extended README and clean code release coming soon! More detailed documentation
will be released after we finish cleaning the code. You can download our models
[here](https://huggingface.co/collections/princeton-nlp/queen-chess-models).

## Apple silicon inference

This fork adds inference with the published PAWN-8 weights on Apple silicon.
MLX runs the SmolLM3 decoder and all 16 trained cross-attention layers; PyTorch
MPS runs the LC0 board encoder once per position. No training or separate chess
engine installation is required.

### Play a live game locally

After downloading the weights with the setup below:

```bash
uv run --frozen --group mac python -m mac_inference.web
```

Open [127.0.0.1:8765](http://127.0.0.1:8765). Play White or Black by clicking
or dragging pieces. QUEEN stays loaded and streams an explanation for each reply.
The game supports promotions, castling, en passant, takebacks, draw claims,
resignation, and PGN export. All moves are checked by the rules engine; a missing
or illegal AI recommendation leaves the board unchanged and offers a retry.

The server listens only on your Mac. The UI uses local assets and requires no
frontend build or additional packages. One game is shared by tabs on the server;
refreshing preserves it, and stopping the server clears it. See
[live game details](docs/mac-inference.md#live-game-demo).

### CLI setup

```bash
uv sync --frozen --group mac
uv run --frozen --group mac python -m mac_inference --download
uv run --frozen --group mac python -m mac_inference \
  --moves e2e4 e7e5 g1f3 b8c6 f1c4
```

The download is approximately 8.4 GiB and is checksum-verified. After the first
download, inference is offline. Use `--fen '...'` for an arbitrary position,
`--backend mps` for the PyTorch reference path, and `--json` or `--output analysis.json`
for structured output. [Mac setup, runtime research, and validation](docs/mac-inference.md)
describe the supported hardware and limits. The original setup below is for the
upstream training and Linux environment, and is not needed for Mac inference.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All launchers use `.venv`. The requirements cover the core dependencies; GPU
inference also requires a compatible vLLM installation, whose combined setup
is still being validated.

## Repository layout

```text
├── models/
│   ├── encoder.py                  # LC0 encoder loading
│   ├── flamingo.py, llava.py       # Model architectures
│   └── vllm/                      # Batched inference and checkpoint exports
├── datagen/
│   ├── sample_positions.py        # Human, puzzle, and gameplay sampling
│   ├── build_qa_dataset.py        # Template data construction
│   ├── build_train_all.py         # Task-dataset concatenation
│   ├── mix_arrows.py              # Curriculum mixtures
│   ├── tasks/                     # Task templates
│   ├── tree/                      # Stockfish/HCE data construction
│   ├── sim/                       # Plan/verdict pipeline
│   ├── self_distill/              # Gameplay, recursion, and consolidation logic
│   └── self_distill_stages.py      # Mining/rebalancing/consolidation CLI
├── eval/
│   ├── benchmark.py               # Tactical/general PV benchmarks
│   ├── critical_line.py           # Benchmark line parsing
│   ├── model_ladder.py            # Batched 32-game model Elo evaluation
│   └── ladder.py                  # UCI engine calibration
├── utils/
│   ├── pack_self_distill.py       # Filtering, eval rewriting, POV translation, Arrow
│   └── ...                        # Shared training, token, and board utilities
├── scripts/
│   ├── environment.sh             # Shared repository/venv setup
│   ├── datagen/
│   │   ├── common/                # Generic sampling/building/packing/mixing
│   │   ├── stage1/                # Stage 1 preparation and builder
│   │   ├── stage2/                # Stage 2 preparation, sampling, building, mixing
│   │   ├── stage3/                # Stage 3 preparation, sampling, building, mixing
│   │   ├── stage4/                # Stage 4 preparation, sampling, building, mixing
│   │   ├── hce/                   # HCE preparation, games/puzzles, and packing
│   │   └── si_iteration/          # One self-improvement round through Arrow packing
│   ├── train/
│   │   ├── common/                # Generic SFT launcher
│   │   ├── stage1/                # Flamingo and LLaVA stage 1
│   │   ├── stage2/                # Flamingo and LLaVA stage 2
│   │   ├── stage3/                # Flamingo and LLaVA stage 3
│   │   ├── stage4/                # Flamingo and LLaVA stage 4
│   │   ├── hce/                   # HCE SFT and LLaVA ablations
│   │   ├── sol_distill/           # Initial Sol distillation SFT
│   │   └── si_iteration/          # Iterative SFT
│   └── eval/
│       ├── benchmarks/            # Tactical/general benchmark array
│       ├── ladder/                # Resumable model Elo evaluation
│       └── native/                # Native Arrow/task and regret evaluations
├── configs/                       # Data, training, iteration, and evaluation YAMLs
├── train.py                       # Training entry point
└── eval.py                        # Native Arrow evaluation entry point
```
