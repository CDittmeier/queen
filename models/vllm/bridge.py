"""Board bridge: FEN(s) -> 64 board-prefix embeddings (B, 64, decoder_dim).

Replicates step 1 of models/llava.py::LLaVAChessLM.forward exactly:
  enc = concat over 16 encoder layers             -> (B, 64, 16384)
  prefix = connector(enc)                          -> (B, 64, decoder_dim)
  prefix += file_embed(sq % 8) + rank_embed(sq // 8)

These are the decoder's first 64 tokens. For vLLM they are concatenated with the
text-token embeddings and fed via prompt_embeds; the decoder itself is the merged
SmolLM3 (see merge_weights.py). Runs in bf16 to match the reference eval's autocast.

Only needs torch + the repo's encoder / plane code — no vLLM — so it is reusable
from a plain training venv as well as the serving venv.
"""
import torch
import torch.nn as nn

from models.encoder import Lc0Bt4HFModel
from utils.lc0_planes import encode_fen_batch
from utils.utils import encode_planes, turn_tensor

# Board geometry; must match models/llava.py (N_ENC_LAYERS / N_ENC_SQUARES / N_FILES).
N_ENC_LAYERS = 16
N_SQUARES = 64
N_FILES = 8
N_RANKS = 8


class BoardBridge:
    """Connector + spatial embeds (+ optionally the lc0 encoder), loaded from a
    merged checkpoint's bridge.pt. Layer widths are inferred from the tensors."""

    def __init__(self, bridge_path, encoder_path=None, device="cuda",
                 dtype=torch.bfloat16, pov=True):
        self.device = torch.device(device)
        self.dtype = dtype
        self.pov = pov
        w = torch.load(bridge_path, map_location="cpu")

        in_dim = w["connector.0.weight"].shape[0]        # 16384 = 16 layers * 1024
        mlp_hidden = w["connector.1.weight"].shape[0]
        decoder_dim = w["connector.3.weight"].shape[0]
        self.connector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mlp_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(mlp_hidden, decoder_dim, bias=False),
        )
        self.connector.load_state_dict({
            "0.weight": w["connector.0.weight"], "0.bias": w["connector.0.bias"],
            "1.weight": w["connector.1.weight"], "3.weight": w["connector.3.weight"],
        })
        self.file_embed = nn.Embedding(N_FILES, decoder_dim)
        self.rank_embed = nn.Embedding(N_RANKS, decoder_dim)
        self.file_embed.load_state_dict({"weight": w["file_embed.weight"]})
        self.rank_embed.load_state_dict({"weight": w["rank_embed.weight"]})

        for module in (self.connector, self.file_embed, self.rank_embed):
            module.to(device=self.device, dtype=self.dtype).eval()

        self.encoder = None
        if encoder_path is not None:
            self.encoder = Lc0Bt4HFModel.from_pretrained(encoder_path, local_files_only=True)
            self.encoder.to(device=self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def prefix_from_hidden(self, enc_hidden: torch.Tensor) -> torch.Tensor:
        """enc_hidden (B, 16, 64, 1024) -> prefix (B, 64, decoder_dim)."""
        enc = torch.cat([enc_hidden[:, e] for e in range(N_ENC_LAYERS)], dim=-1)
        prefix = self.connector(enc.to(self.device, self.dtype))
        sq = torch.arange(N_SQUARES, device=self.device)
        return prefix + self.file_embed(sq % N_FILES) + self.rank_embed(sq // N_FILES)

    @torch.no_grad()
    def prefix_from_fens(self, fens, histories=None) -> torch.Tensor:
        """FEN list -> prefix (B, 64, decoder_dim). Requires encoder_path at init."""
        assert self.encoder is not None, "BoardBridge built without an encoder"
        if histories is None:
            histories = [None] * len(fens)
        planes = encode_fen_batch(fens, histories).to(self.device)
        enc_hidden = encode_planes(self.encoder, planes, self.dtype,
                                   pov=self.pov, turn=turn_tensor(fens))
        return self.prefix_from_hidden(enc_hidden)
