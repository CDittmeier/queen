# QUEEN

Extended README and clean code release coming soon! More detailed documentation
will be released after we finish cleaning the code. You can download our models
[here](https://huggingface.co/collections/princeton-nlp/queen-chess-models).

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
