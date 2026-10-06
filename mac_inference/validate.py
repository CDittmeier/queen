"""Validate the downloaded weights and complete Mac inference paths; no training."""

import argparse
import json
from pathlib import Path

import chess
import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.cache import make_prompt_cache

from models.adapters import adapter_for
from models.flamingo import FlamingoChessLM

from .common import BoardEncoder, prompt_ids, verify_release
from .mlx import MLXRunner
from .mps import MPSRunner


def error(actual, expected):
    difference = np.abs(actual - expected)
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
    }


@torch.inference_mode()
def validate(directory: Path):
    verify_release(directory)
    encoder = BoardEncoder(directory)
    _, ids, _ = prompt_ids(directory)
    # FP32 comparisons isolate architecture/layout/cache errors from BF16 rounding.
    mps = MPSRunner(directory)
    mps.model.float()
    mps.bridges.float()
    mlx = MLXRunner(directory)
    mlx.model.set_dtype(mx.float32)
    mx.eval(mlx.model.parameters())
    manual = FlamingoChessLM(mps.model, adapter_for(mps.model), n_new_tokens=0)
    manual.x_attn_layers = mps.bridges
    manual.eval()
    report = {
        "release_checksums": "passed",
        "comparison_dtype": "float32",
        "positions": [],
    }
    board = chess.Board()
    history = []
    positions = [(board.fen(), [])]
    for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"):
        history.append(board.fen())
        board.push_uci(move)
    positions.append((board.fen(), history))
    first_logits = None
    for fen, history in positions:
        print(
            f"Checking encoder, cross-attention, full decoder, and KV cache: {fen}",
            flush=True,
        )
        states = encoder.encode(fen, history)
        cpu_states = states.cpu().numpy()
        mps.clear_board()
        inputs = torch.tensor([ids], device="mps")
        hidden = manual._hidden_states(inputs, states)
        reference = mps.model.lm_head(hidden[:, -1:]).float().cpu().numpy()
        mps.set_board(states)
        actual = (
            mps.model(inputs, use_cache=False, logits_to_keep=1)
            .logits.float()
            .cpu()
            .numpy()
        )
        mps_error = error(actual, reference)
        np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-3)
        for index, bridge in enumerate(mlx.model.x_attn_layers):
            bridge.set_board(mx.array(cpu_states[:, index]))
        mlx_inputs = mx.array([ids])
        mlx_logits = np.array(mlx.model(mlx_inputs)[:, -1:])
        mlx_error = error(mlx_logits, reference)
        np.testing.assert_allclose(mlx_logits, reference, atol=3e-3, rtol=1e-3)
        # Verify a cached next-token step agrees with a complete forward pass.
        cache = make_prompt_cache(mlx.model)
        prefix = mlx.model(mlx_inputs, cache=cache)
        mx.eval(prefix, [item.state for item in cache])
        token = int(np.argmax(mlx_logits[0, 0]))
        cached = np.array(mlx.model(mx.array([[token]]), cache=cache))
        full = np.array(mlx.model(mx.array([ids + [token]]))[:, -1:])
        cache_error = error(cached, full)
        np.testing.assert_allclose(cached, full, atol=3e-3, rtol=1e-3)
        if first_logits is None:
            first_logits = mlx_logits.copy()
        else:
            assert np.max(np.abs(first_logits - mlx_logits)) > 0.1, (
                "Board conditioning had no effect"
            )
        report["positions"].append(
            {
                "fen": fen,
                "history_length": len(history),
                "encoder_shape": list(states.shape),
                "mps_vs_upstream": mps_error,
                "mlx_vs_upstream": mlx_error,
                "mlx_cache_vs_full": cache_error,
            }
        )
    report["board_conditioning"] = "passed"
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".models/queen_pawn-8"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
