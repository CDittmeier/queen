"""vLLM serving stack for the stage-5 SmolLM3 llava model.

  merge_weights.py  checkpoint -> standalone SmolLM3 HF dir + bridge.pt
  bridge.py         FEN(s)     -> 64 board-prefix embeddings
  llava.py          vLLM model: SmolLM3 with prefix RoPE positions pinned to 0
  generate.py       ChessVLLMGenerator: prefix + text embeds -> vLLM -> tokens

SmolLM3-specific: the merge assumes an identity embedding scale (true for SmolLM3,
false for Gemma). Everything mirrors models/llava.py's board-prefix scheme.
"""
