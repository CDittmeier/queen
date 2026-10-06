import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import chess

from .common import BoardEncoder, download_model, prompt_ids, request_seed, result


def main():
    parser = argparse.ArgumentParser(
        description="Run board-conditioned QUEEN on Apple silicon"
    )
    parser.add_argument("--backend", choices=["mlx", "mps"], default="mlx")
    parser.add_argument("--model", choices=["pawn-8"], default="pawn-8")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--download", action="store_true", help="Download and verify the pinned release"
    )
    position = parser.add_mutually_exclusive_group()
    position.add_argument("--fen", default=chess.STARTING_FEN)
    position.add_argument(
        "--moves",
        nargs="+",
        help="UCI moves from the initial position; retains history",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="JSON array of prior FENs, excluding current position",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--game-id", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Save JSON including raw text and timing"
    )
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192:
        parser.error("--max-tokens must be between 1 and 8192")
    if not 0 <= args.temperature <= 5:
        parser.error("--temperature must be between 0 and 5")
    if args.moves and args.history:
        parser.error("--moves already builds history; omit --history")
    try:
        board = chess.Board(args.fen)
        history = json.loads(args.history.read_text()) if args.history else []
        if not isinstance(history, list) or any(
            not isinstance(f, str) for f in history
        ):
            raise ValueError("History must be a JSON list of FEN strings")
        for fen in history:
            if not chess.Board(fen).is_valid():
                raise ValueError(f"Invalid history FEN: {fen}")
        for move in args.moves or []:
            history.append(board.fen())
            board.push_uci(move)
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        if board.is_game_over():
            raise ValueError("The position is terminal; no move can be recommended")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    directory = (
        args.model_dir
        or Path(__file__).resolve().parents[1] / ".models" / "queen_pawn-8"
    )
    if args.download:
        os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
        download_model(directory, args.model)
    if not (directory / "xattn.pt").is_file():
        parser.error("Model is missing; first run with --download")
    tokenizer, ids, prompt = prompt_ids(directory, args.prompt)
    print(f"Loading QUEEN ({args.backend}, BF16)…", file=sys.stderr, flush=True)
    start = perf_counter()
    encoder = BoardEncoder(directory)
    if args.backend == "mlx":
        from .mlx import MLXRunner

        runner = MLXRunner(directory)
    else:
        from .mps import MPSRunner

        runner = MPSRunner(directory)
    load_seconds = perf_counter() - start
    start = perf_counter()
    states = encoder.encode(board.fen(), history)
    runner.set_board(states)
    encoder_seconds = perf_counter() - start
    print("Analyzing position…", file=sys.stderr, flush=True)
    seed = request_seed(args.seed, args.game_id, board)
    raw, timing = runner.generate(
        tokenizer, ids, args.max_tokens, args.temperature, seed
    )
    answer = result(
        board,
        raw,
        backend=args.backend,
        dtype="bfloat16",
        prompt=prompt,
        temperature=args.temperature,
        top_k=20,
        top_p=0.95,
        request_seed=seed,
        max_tokens=args.max_tokens,
        load_seconds=load_seconds,
        encoder_seconds=encoder_seconds,
        **timing,
    )
    encoded = json.dumps(answer, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.json:
        print(encoded)
    else:
        print(answer["text"])
        recommendation = answer["best_move_san"] or "No explicit legal recommendation"
        print(f"\nBest move: {recommendation}")
        print(
            f"{timing['generated_tokens']} tokens, "
            f"{timing['tokens_per_second']:.1f} tokens/s ({args.backend})"
        )


if __name__ == "__main__":
    main()
