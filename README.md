Download our trained models on [Huggingface](https://huggingface.co/collections/princeton-nlp/queen-chess-models)

```
├── models/            # Contains code for specific ChessLM architectures (LoRA + FSDP2 compatabible)
│   ├── base.py            # Base protocol that fits into training and inference code
│   ├── flamingo.py        # Implementation of Flamingo XAttention arch
│   └── llava.py           # Implementation of LLaVa embedding projection arch
|
├── datagen/               # Contains code for data generation pipeline
|   ├── prose.py               # Contains general utility functions for formatting strings from tokens
|   ├── position_features.py   # Canonical feature extraction from each position
|   ├── sample_positions.py    # Code for position sampling from raw .jsonl.zst files into .jsonl format
|   ├── build_qa_dataset.py    # Code for generating QA SFT datasets from .jsonl position list
|   ├── build_train_all.py     # Code for concatenating separate task datasets
|   ├── mix_arrows.py          # Code for dataset mixing logic
|   └── tasks/                 # Directory containing task specific templates and weighting functions
|
├── utils/                    # Contains code for supplementary util functions
|   ├── utils.py                  # General utility functions; encoder pass, custom tokens
│   ├── board_representation.py   # Canonical board representation for data processing and evaluation
│   ├── instance_format.py        # Canonical data format expected by trainers from dataloaders
│   ├── lc0_planes.py             # Utility functions to control (frozen) encoder forward passes
│   ├── special_tokens.py         # Tokenizer utility functions to add tokens and initialize new embeddings
|   ├── training_utils.py         # Training utility functions for data, model, tokenizer, optim, fsdp wrap
|   └── eval_utils.py             # Evaluation utility functions for generation, answer checking
|
├── encoder/          # Code for weights and loading of LC0 encoder
├── scripts/          # Contains scripts for training and inference
├── configs/          # Contains configs for input into scripts
├── train.py          # Main training loop
└── eval.py           # Main evaluation loop

```
