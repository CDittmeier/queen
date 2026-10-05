#!/usr/bin/env python3
"""Place one analysis model on the fixed engine Elo ladder.

One target checkpoint plays four color-balanced games against every configured
opponent.  All live model turns are generated together through one persistent
vLLM instance; opponent moves are grouped by engine and the groups run in
parallel.  The output directory is resumable at one batched-ply cycle.

The runner writes ``state.json``, ``model_moves.jsonl``, ``games.pgn`` and
``results.json`` after every cycle.  Opponent Elos are fixed anchors and only
the target rating is fitted.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import yaml
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn


REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datagen.self_distill.analysis import (
    ANALYSIS_PROMPT, legal_structured_best, legal_root_candidates, parse_critical,
)

def history_fens(board: chess.Board) -> list[str]:
    replay = board.root()
    history = []
    for move in board.move_stack:
        history.append(replay.fen())
        replay.push(move)
    return history

def stable_seed(seed: int, game: int, ply: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{game}|{ply}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)

def choose_model_move(fen: str, generation: str, seed: int) -> tuple[chess.Move, str, dict]:
    board = chess.Board(fen)
    best = legal_structured_best(fen, generation)
    critical = parse_critical(fen, generation)
    critical_move = (
        chess.Move.from_uci(critical["legal_steps"][0]["uci"])
        if critical["legal_steps"] else None
    )
    candidates = legal_root_candidates(fen, generation)
    if best is not None:
        move, source = best, "best_move"
    elif critical_move is not None:
        move, source = critical_move, "critical_line"
    elif candidates:
        move, source = candidates[0], "promising_or_analysis"
    else:
        move = random.Random(seed).choice(list(board.legal_moves))
        source = "random_legal_fallback"
    return move, source, {
        "structured_best_uci": best.uci() if best else None,
        "critical_first_uci": critical_move.uci() if critical_move else None,
        "critical_illegal_at": critical["illegal_at"],
        "candidate_uci": [move.uci() for move in candidates],
    }

SCHEMA_VERSION = 1
DEFAULT_OPENINGS = (
    {"name": "italian_setup", "moves_san": ["e4", "e5", "Nf3", "Nc6"]},
    {"name": "qgd_setup", "moves_san": ["d4", "d5", "c4", "e6"]},
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, rows: list[dict]) -> int:
    if rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return path.stat().st_size


def truncate(path: Path, size: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(size)


def canonical_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    return str(path.resolve())


def opening_uci(opening: dict) -> list[str]:
    board = chess.Board()
    moves = []
    for san in opening["moves_san"]:
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    return moves


def initial_games(config: dict, target: dict) -> list[dict]:
    """Two openings, each played once with each color, per opponent."""
    openings = config.get("openings") or list(DEFAULT_OPENINGS)
    if len(openings) != 2:
        raise ValueError("exactly two openings are required for four games/opponent")
    games = []
    for opponent_index, opponent in enumerate(config["opponents"]):
        for opening_index, opening in enumerate(openings):
            moves = opening_uci(opening)
            for model_white in (True, False):
                game_id = len(games)
                games.append({
                    "game_id": game_id,
                    "opponent_index": opponent_index,
                    "opponent": opponent["name"],
                    "opponent_elo": float(opponent["elo"]),
                    "opening_index": opening_index,
                    "opening": opening["name"],
                    "model_white": model_white,
                    "moves": list(moves),
                    "move_sources": ["opening"] * len(moves),
                    "result": None,
                    "termination": None,
                })
    return games


def board_for(game: dict) -> chess.Board:
    board = chess.Board()
    for uci in game["moves"]:
        board.push_uci(uci)
    return board


def finish_game(game: dict, board: chess.Board, max_plies: int) -> bool:
    capped = board.ply() >= max_plies
    outcome = board.outcome(claim_draw=True)
    if outcome is None and not capped:
        return False
    if capped or outcome is None or outcome.winner is None:
        game["result"] = "1/2-1/2"
        game["termination"] = (
            f"ply_cap_{max_plies}" if capped else outcome.termination.name.lower()
        )
    else:
        game["result"] = "1-0" if outcome.winner else "0-1"
        game["termination"] = outcome.termination.name.lower()
    return True


def white_score(result: str) -> float:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]


def model_score(game: dict) -> float:
    score = white_score(game["result"])
    return score if game["model_white"] else 1.0 - score


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def fit_rating(rows: list[tuple[float, float, int]]) -> dict:
    """One-dimensional Bradley-Terry MLE with fixed opponent ratings."""
    ratings = range(-200, 3601)

    def log_likelihood(rating: float) -> float:
        value = 0.0
        for score, opponent_rating, games in rows:
            probability = min(max(expected_score(rating, opponent_rating), 1e-12),
                              1.0 - 1e-12)
            value += (score * math.log(probability)
                      + (games - score) * math.log(1.0 - probability))
        return value

    likelihoods = [log_likelihood(float(rating)) for rating in ratings]
    best_index = max(range(len(likelihoods)), key=likelihoods.__getitem__)
    cutoff = likelihoods[best_index] - 1.92
    interval = [rating for rating, value in zip(ratings, likelihoods)
                if value >= cutoff]
    return {
        "elo": float(ratings[best_index]),
        "ci95_low": float(min(interval)),
        "ci95_high": float(max(interval)),
        "anchored": False,
    }


class FixedUciPlayer:
    """A persistent UCI process searched to one fixed node budget."""

    def __init__(self, spec: dict, stockfish: Path, lc0: Path):
        self.name = spec["name"]
        self.nodes = int(spec["nodes"])
        if spec["kind"] == "stockfish":
            command = [str(stockfish)]
        else:
            command = [str(lc0), f"--weights={spec['weights']}",
                       *spec.get("extra_args", [])]
        self.engine = chess.engine.SimpleEngine.popen_uci(command)
        if spec["kind"] == "stockfish":
            self.engine.configure({
                "Threads": int(spec.get("threads", 1)),
                "Hash": int(spec.get("hash_mb", 64)),
            })
        self.game_token = 0

    def choose_move(self, board: chess.Board) -> chess.Move:
        self.game_token += 1
        result = self.engine.play(
            board, chess.engine.Limit(nodes=self.nodes), game=self.game_token
        )
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move")
        return result.move

    def close(self) -> None:
        self.engine.quit()


class OpponentPool:
    """One persistent process per opponent, with opponents run concurrently."""

    def __init__(self, specs: list[dict], stockfish: Path, lc0: Path):
        self.specs = specs
        self.players = {}
        for spec in specs:
            self.players[spec["name"]] = FixedUciPlayer(spec, stockfish, lc0)
        self.executor = ThreadPoolExecutor(max_workers=len(self.players))

    def _group(self, opponent: str, games: list[dict]) -> list[tuple[int, str]]:
        player = self.players[opponent]
        result = []
        for game in games:
            board = board_for(game)
            # Treat every board as an independent search.  Interleaved games
            # must not inherit engine state from one another.
            move = player.choose_move(board)
            result.append((game["game_id"], move.uci()))
        return result

    def choose(self, games: list[dict]) -> dict[int, str]:
        grouped = defaultdict(list)
        for game in games:
            grouped[game["opponent"]].append(game)
        futures = {
            self.executor.submit(self._group, opponent, rows): opponent
            for opponent, rows in grouped.items()
        }
        result = {}
        for future in as_completed(futures):
            for game_id, uci in future.result():
                result[game_id] = uci
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for player in self.players.values():
            player.close()


def fit_target(games: list[dict], target_name: str,
               opponents: list[dict]) -> dict | None:
    rows_for_fit = []
    for opponent in opponents:
        rows = [game for game in games
                if game["opponent"] == opponent["name"] and game["result"]]
        if rows:
            rows_for_fit.append((
                sum(model_score(game) for game in rows),
                float(opponent["elo"]), len(rows),
            ))
    if not rows_for_fit:
        return None
    return fit_rating(rows_for_fit)


def summary(state: dict, target: dict, config: dict) -> dict:
    games = state["games"]
    breakdown = []
    for opponent in config["opponents"]:
        rows = [game for game in games
                if game["opponent"] == opponent["name"] and game["result"]]
        score = sum(model_score(game) for game in rows)
        breakdown.append({
            "opponent": opponent["name"],
            "opponent_elo": float(opponent["elo"]),
            "finished": len(rows),
            "games": 4,
            "score": score,
            "score_rate": score / len(rows) if rows else None,
        })
    finished = sum(game["result"] is not None for game in games)
    return {
        "schema_version": SCHEMA_VERSION,
        "target": target["name"],
        "model": target["model"],
        "finished": finished,
        "games": len(games),
        "complete": finished == len(games),
        "score": sum(model_score(game) for game in games if game["result"]),
        "rating": fit_target(games, target["name"], config["opponents"]),
        "opponents": breakdown,
        "model_positions": state["model_positions"],
        "generated_tokens": state["generated_tokens"],
        "truncated_generations": state["truncated_generations"],
        "random_fallbacks": state["random_fallbacks"],
        "cycles": state["cycles"],
        "pipeline_seconds": state["pipeline_seconds"],
        "model_generation_seconds": state["model_generation_seconds"],
        "opponent_seconds": state["opponent_seconds"],
        "model_positions_per_second": (
            state["model_positions"] / state["model_generation_seconds"]
            if state["model_generation_seconds"] else None
        ),
        "tokens_per_second": (
            state["generated_tokens"] / state["model_generation_seconds"]
            if state["model_generation_seconds"] else None
        ),
    }


def render_pgn(path: Path, games: list[dict], target_name: str) -> None:
    rendered = []
    for row in games:
        game = chess.pgn.Game()
        game.headers.update(
            Event="QUEEN fixed Elo ladder",
            Round=str(row["game_id"] + 1),
            White=target_name if row["model_white"] else row["opponent"],
            Black=row["opponent"] if row["model_white"] else target_name,
            Result=row["result"] or "*",
            Termination=row["termination"] or "unterminated",
            Opening=row["opening"],
            OpponentElo=str(round(row["opponent_elo"])),
        )
        node = game
        for uci in row["moves"]:
            node = node.add_main_variation(chess.Move.from_uci(uci))
        rendered.append(game.accept(chess.pgn.StringExporter(
            headers=True, variations=False, comments=False
        )).strip())
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text("\n\n".join(rendered) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_config(raw: dict) -> dict:
    config = json.loads(json.dumps(raw))
    config["encoder"] = canonical_path(config["encoder"])
    config["stockfish"] = canonical_path(config["stockfish"])
    config["lc0"] = canonical_path(config["lc0"])
    for target in config["targets"]:
        target["model"] = canonical_path(target["model"])
    for opponent in config["opponents"]:
        if opponent["kind"] == "lc0":
            opponent["weights"] = canonical_path(opponent["weights"])
        opponent["elo"] = float(opponent["elo"])
    config.setdefault("prompt", ANALYSIS_PROMPT)
    config.setdefault("max_plies", 400)
    config.setdefault("max_output_tokens", 2048)
    config.setdefault("temperature", 0.6)
    config.setdefault("top_k", 20)
    config.setdefault("top_p", 0.95)
    config.setdefault("seed", 20260823)
    config.setdefault("gpu_memory_utilization", 0.72)
    config.setdefault("max_num_seqs", 32)
    config.setdefault("openings", list(DEFAULT_OPENINGS))
    return config


def validate_config(config: dict) -> None:
    if len(config["opponents"]) != 8:
        raise ValueError("the fixed ladder panel must contain exactly eight opponents")
    if len({row["name"] for row in config["opponents"]}) != 8:
        raise ValueError("opponent names must be unique")
    for key in ("encoder", "stockfish", "lc0"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"missing {key}: {config[key]}")
    for opponent in config["opponents"]:
        if opponent["kind"] not in ("stockfish", "lc0"):
            raise ValueError(f"unsupported opponent kind: {opponent['kind']}")
        if opponent["kind"] == "lc0" and not Path(opponent["weights"]).exists():
            raise FileNotFoundError(opponent["weights"])
    initial_games(config, config["targets"][0])


def select_opponent_shard(config: dict, index: int, count: int) -> dict:
    """Return one disjoint, order-preserving shard of the fixed opponent panel."""
    if count < 1:
        raise ValueError("opponent-shard-count must be positive")
    if not 0 <= index < count:
        raise ValueError(
            f"opponent-shard-index {index} is outside 0..{count - 1}"
        )
    sharded = json.loads(json.dumps(config))
    sharded["opponents"] = [
        opponent for opponent_index, opponent in enumerate(config["opponents"])
        if opponent_index % count == index
    ]
    if not sharded["opponents"]:
        raise ValueError(f"opponent shard {index}/{count} is empty")
    return sharded


def target_for(config: dict, index: int) -> dict:
    if not 0 <= index < len(config["targets"]):
        raise ValueError(f"invalid target-index: {index}")
    target = config["targets"][index]
    if not Path(target["model"]).exists():
        raise FileNotFoundError(f"missing merged target model: {target['model']}")
    return target


def run(args: argparse.Namespace, config: dict, target: dict) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    state_path = output / "state.json"
    events_path = output / "model_moves.jsonl"
    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "target": target,
        "config": {key: value for key, value in config.items() if key != "targets"},
    }
    # Hardware placement does not affect play; a resume may move between GPUs.
    def comparable(fp):
        return {**fp, "config": {k: v for k, v in fp["config"].items()
                                 if k != "gpu_memory_utilization"}}
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        manifest = json.loads(manifest_path.read_text())
        if comparable(manifest["fingerprint"]) != comparable(fingerprint):
            raise RuntimeError("resume configuration does not match manifest")
        state = json.loads(state_path.read_text())
        truncate(events_path, state["events_bytes"])
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"output directory is nonempty: {output}")
        events_path.write_text("")
        state = {
            "games": initial_games(config, target),
            "cycles": 0,
            "model_positions": 0,
            "generated_tokens": 0,
            "truncated_generations": 0,
            "random_fallbacks": 0,
            "pipeline_seconds": 0.0,
            "model_generation_seconds": 0.0,
            "opponent_seconds": 0.0,
            "events_bytes": 0,
        }
        atomic_json(manifest_path, {"fingerprint": fingerprint})
        atomic_json(state_path, state)

    if all(game["result"] is not None for game in state["games"]):
        atomic_json(output / "results.json", summary(state, target, config))
        render_pgn(output / "games.pgn", state["games"], target["name"])
        print("[done] all games already complete", flush=True)
        return

    from models.vllm.flamingo_generate import ChessFlamingoGenerator

    print(f"[load] {target['name']} <- {target['model']}", flush=True)
    generator = ChessFlamingoGenerator(
        target["model"], config["encoder"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_output_tokens"] + 256,
        max_num_seqs=config["max_num_seqs"],
        seed=config["seed"], enforce_eager=True, use_v1_vllm=True,
    )
    opponents = OpponentPool(
        config["opponents"], Path(config["stockfish"]), Path(config["lc0"])
    )
    invocation_started = time.perf_counter()
    cycles_this_run = 0
    while any(game["result"] is None for game in state["games"]):
        cycle_started = time.perf_counter()
        live = [game for game in state["games"] if game["result"] is None]
        for game in live:
            finish_game(game, board_for(game), config["max_plies"])
        live = [game for game in state["games"] if game["result"] is None]

        engine_games = [
            game for game in live
            if board_for(game).turn != game["model_white"]
        ]
        if engine_games:
            started = time.perf_counter()
            moves = opponents.choose(engine_games)
            state["opponent_seconds"] += time.perf_counter() - started
            for game in engine_games:
                board = board_for(game)
                move = chess.Move.from_uci(moves[game["game_id"]])
                if move not in board.legal_moves:
                    raise RuntimeError(
                        f"{game['opponent']} returned illegal {move} at {board.fen()}"
                    )
                game["moves"].append(move.uci())
                game["move_sources"].append("opponent")
                board.push(move)
                finish_game(game, board, config["max_plies"])

        model_games = [game for game in state["games"]
                       if game["result"] is None]
        event_rows = []
        if model_games:
            boards = [board_for(game) for game in model_games]
            if not all(board.turn == game["model_white"]
                       for board, game in zip(boards, model_games, strict=True)):
                raise RuntimeError("scheduler left a live opponent turn in the model batch")
            fens = [board.fen() for board in boards]
            histories = [history_fens(board) for board in boards]
            started = time.perf_counter()
            outputs = generator.generate(
                fens, [config["prompt"]] * len(fens), histories,
                temperature=config["temperature"],
                top_k=config["top_k"], top_p=config["top_p"],
                max_tokens=config["max_output_tokens"],
                seeds=[stable_seed(config["seed"], game["game_id"], board.ply())
                       for game, board in zip(model_games, boards, strict=True)],
            )
            state["model_generation_seconds"] += time.perf_counter() - started
            for game, board, fen, history, model_output in zip(
                    model_games, boards, fens, histories, outputs, strict=True):
                seed = stable_seed(config["seed"], game["game_id"], board.ply())
                move, source, parsing = choose_model_move(
                    fen, model_output["text"], seed
                )
                if move not in board.legal_moves:
                    raise RuntimeError(f"move parser returned illegal {move} at {fen}")
                event_rows.append({
                    "game_id": game["game_id"], "opponent": game["opponent"],
                    "ply": board.ply(), "fen": fen, "history": history,
                    "uci": move.uci(), "san": board.san(move),
                    "source": source, "parsing": parsing,
                    "generation": model_output["text"],
                    "token_count": len(model_output["token_ids"]),
                    "finish_reason": model_output["finish_reason"],
                })
                game["moves"].append(move.uci())
                game["move_sources"].append("model")
                board.push(move)
                finish_game(game, board, config["max_plies"])
                state["model_positions"] += 1
                state["generated_tokens"] += len(model_output["token_ids"])
                state["truncated_generations"] += (
                    model_output["finish_reason"] == "length"
                )
                state["random_fallbacks"] += source == "random_legal_fallback"

        state["cycles"] += 1
        cycles_this_run += 1
        state["events_bytes"] = append_jsonl(events_path, event_rows)
        state["pipeline_seconds"] += time.perf_counter() - cycle_started
        atomic_json(state_path, state)
        current = summary(state, target, config)
        atomic_json(output / "results.json", current)
        render_pgn(output / "games.pgn", state["games"], target["name"])
        rating = current["rating"]
        elo = "—" if rating is None else f"{rating['elo']:.0f}"
        total_games = len(state["games"])
        print(
            f"[cycle {state['cycles']}] finished={current['finished']}/{total_games} "
            f"active={total_games - current['finished']} positions={state['model_positions']} "
            f"tokens={state['generated_tokens']} elo={elo} "
            f"fallbacks={state['random_fallbacks']} "
            f"truncated={state['truncated_generations']}",
            flush=True,
        )
        if args.max_cycles and cycles_this_run >= args.max_cycles:
            print(f"[stop] reached --max-cycles={args.max_cycles}", flush=True)
            break
        if args.max_seconds and time.perf_counter() - invocation_started >= args.max_seconds:
            print(f"[stop] reached --max-seconds={args.max_seconds}", flush=True)
            break
    opponents.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="clean smoke-test stop after N new cycles")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="clean stop after approximately this many seconds")
    parser.add_argument("--opponent-shard-index", type=int, default=0,
                        help="zero-based opponent shard to run")
    parser.add_argument("--opponent-shard-count", type=int, default=1,
                        help="number of disjoint opponent shards")
    args = parser.parse_args()

    config = normalized_config(yaml.safe_load(args.config.read_text()))
    validate_config(config)
    config = select_opponent_shard(
        config, args.opponent_shard_index, args.opponent_shard_count
    )
    if args.validate_only:
        print(json.dumps({
            "targets": len(config["targets"]),
            "opponents": len(config["opponents"]),
            "games_per_target": len(initial_games(config, config["targets"][0])),
            "schedule": [
                {"opponent": row["opponent"], "opening": row["opening"],
                 "model_white": row["model_white"]}
                for row in initial_games(config, config["targets"][0])
            ],
        }, indent=2))
        return
    target = target_for(config, args.target_index)
    if args.output is None:
        args.output = REPO / "runs" / "eval" / "ladder" / target["name"]
    run(args, config, target)


if __name__ == "__main__":
    main()
