"""CPU scoring and selection of durable, caller-named gameplay shards."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from datagen.self_distill.play_engines import StockfishPool as _StockfishPool
from datagen.self_distill.play_storage import (
    _atomic_json, file_digest, fingerprint, publish_jsonl, scored_path, shard_lock, sync_directory,
)


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    stockfish: Path
    oracle_nodes: int = 100_000
    stockfish_workers: int = 6

    def validate(self):
        if self.oracle_nodes <= 0 or self.stockfish_workers <= 0:
            raise ValueError("oracle_nodes and stockfish_workers must be positive")
        if not self.stockfish.is_file():
            raise FileNotFoundError(self.stockfish)


@dataclass(frozen=True, slots=True)
class PlayPosition:
    """One played position and its stronger-oracle comparison."""

    fen: str
    history: tuple[str, ...]
    oracle_move: str
    played_move: str
    winrate_drop: float
    oracle_cp: int
    played_cp: int
    mover: str
    game: int
    ply: int
    model_white: bool
    opponent_nodes: int
    move_source: str
    cached_root: dict | None = field(default=None, compare=False, hash=False)

    @property
    def true_move(self) -> str:
        """Compatibility name for the move actually played in the game."""
        return self.played_move


@dataclass(frozen=True, slots=True)
class PlaySamplingResult:
    positions: tuple[PlayPosition, ...]
    games: tuple[dict, ...]
    statistics: dict



def score_play_positions(config: ScoreConfig, source: Path):
    """Read a raw shard in batches; no model, scratch state, or GPU is required."""
    config.validate()
    positions, games = [], []
    started = time.perf_counter()
    with source.open() as handle:
        metadata = json.loads(next(handle))["metadata"]
        def records():
            for line in handle:
                row = json.loads(line)
                if "game" in row:
                    games.append(row["game"])
                else:
                    yield row["position"]
        iterator = records()
        pool = _StockfishPool(config.stockfish, config.stockfish_workers)
        try:
            while batch := list(islice(iterator, config.stockfish_workers * 16)):
                scores = pool.audit([(row["fen"], row["move"]) for row in batch], config.oracle_nodes)
                for row, (best, best_cp, played_cp, drop) in zip(batch, scores, strict=True):
                    positions.append(PlayPosition(
                        fen=row["fen"], history=tuple(row["history"]), oracle_move=best,
                        played_move=row["move"], winrate_drop=drop, oracle_cp=best_cp,
                        played_cp=played_cp, mover=row["mover"], game=row["game"],
                        ply=row["ply"], model_white=row["model_white"],
                        opponent_nodes=row["opponent_nodes"], move_source=row["move_source"],
                        cached_root=row["cached_root"],
                    ))
                print(f"[oracle] {len(positions):,}/{metadata['statistics']['positions']:,}", flush=True)
        finally:
            pool.close()
    if len(positions) != metadata["statistics"]["positions"] or len(games) != metadata["statistics"]["games"]:
        raise RuntimeError("raw shard row counts differ from gameplay statistics")
    stats = dict(metadata["statistics"])
    stats["oracle_seconds"] = time.perf_counter() - started
    stats["wall_seconds"] += stats["oracle_seconds"]
    return PlaySamplingResult(tuple(positions), tuple(games), stats), metadata["specification"]


def normalized_fen(fen: str) -> str:
    return " ".join(fen.split()[:4])


def _play_eval_bucket(cp: int) -> str:
    value = abs(cp) / 100.0
    if value < 0.7:
        return "0.0-0.7"
    if value < 1.5:
        return "0.7-1.5"
    if value < 2.5:
        return "1.5-2.5"
    return "2.5+"


def select_play_positions(positions, cfg: dict, rng: random.Random):
    """Keep mistakes plus randomly chosen per-game ordinary-position quotas."""
    # Quotas count played plies before deduplication or mistake filtering.
    force_drop = float(cfg.get("force_keep_winrate_drop", 0.05))
    high_downsample = int(cfg.get("high_eval_downsample", 4))
    if "sample_fraction" in cfg or "target_positions" in cfg:
        raise ValueError("sample_fraction/target_positions were replaced by per-game ten-ply quotas")
    if force_drop < 0.0 or high_downsample <= 0:
        raise ValueError("invalid play filtering configuration")

    played_plies = defaultdict(set)
    for row in positions:
        played_plies[row.game].add(row.ply)
    quotas = {game: math.ceil(len(plies) / 10) for game, plies in played_plies.items()}

    # deduplication: retain the occurrence with the largest observed move error.
    by_fen = {}
    for row in positions:
        key = normalized_fen(row.fen)
        incumbent = by_fen.get(key)
        rank = (row.winrate_drop, row.game, row.ply, row.mover)
        if incumbent is None or rank > (
            incumbent.winrate_drop,
            incumbent.game,
            incumbent.ply,
            incumbent.mover,
        ):
            by_fen[key] = row
    unique = list(by_fen.values())
    # forced examples: always retain positions where the played move lost enough win rate.
    forced = [row for row in unique if row.winrate_drop >= force_drop]
    ordinary = [row for row in unique if row.winrate_drop < force_drop]
    high = [row for row in ordinary if _play_eval_bucket(row.oracle_cp) == "2.5+"]
    lower = [row for row in ordinary if _play_eval_bucket(row.oracle_cp) != "2.5+"]
    rng.shuffle(high)
    high = high[: math.ceil(len(high) / high_downsample)]
    eligible_by_game = defaultdict(list)
    for row in lower + high:
        eligible_by_game[row.game].append(row)
    # Sample across the whole game, not separately within consecutive ply windows.
    ordinary_selected = []
    selected_per_game = {}
    for game in sorted(quotas):
        eligible = eligible_by_game[game]
        rng.shuffle(eligible)
        chosen = eligible[:quotas[game]]
        ordinary_selected.extend(chosen)
        selected_per_game[str(game)] = len(chosen)
    selected = forced + ordinary_selected
    rng.shuffle(selected)
    statistics = {
        "raw_positions": len(positions),
        "unique_positions": len(unique),
        "duplicates_removed": len(positions) - len(unique),
        "ordinary_target_positions": sum(quotas.values()),
        "ordinary_selected_positions": len(ordinary_selected),
        "ordinary_shortfall": sum(quotas.values()) - len(ordinary_selected),
        "ordinary_quota_by_game": {str(game): quota for game, quota in sorted(quotas.items())},
        "ordinary_selected_by_game": selected_per_game,
        "force_keep_winrate_drop": force_drop,
        "forced_positions": len(forced),
        "ordinary_lower_eval_positions": len(lower),
        "ordinary_high_eval_positions": len(ordinary) - len(lower),
        "ordinary_high_eval_eligible": len(high),
        "high_eval_downsample": high_downsample,
        "ordinary_policy": "ceil_played_plies_div_10_random_per_game",
        "selected_positions": len(selected),
        "selected_eval_buckets": dict(Counter(
            _play_eval_bucket(row.oracle_cp) for row in selected
        )),
        "selected_movers": dict(Counter(row.mover for row in selected)),
    }
    return selected, statistics



def run_score(source: Path, cfg: dict, stockfish: Path) -> Path:
    source = source.resolve()
    output_path = scored_path(source)
    manifest_path = output_path.with_suffix(".manifest.json")
    config = ScoreConfig(
        stockfish=stockfish.resolve(),
        oracle_nodes=int(cfg.get("oracle_nodes", 100_000)),
        stockfish_workers=int(cfg.get("stockfish_workers", 6)),
    )
    config.validate()
    scoring = {"selection_policy_version": 2, "oracle_nodes": config.oracle_nodes, "stockfish_sha256": fingerprint(config.stockfish),
               "selection": cfg}
    with shard_lock(source):
        if manifest_path.exists():
            # A prior attempt may have finished publication but not input cleanup.
            manifest = json.loads(manifest_path.read_text())
            if (manifest["configuration"]["scoring"] != scoring
                    or manifest["source"]["path"] != str(source)
                    or file_digest(output_path) != manifest["output"]["sha256"]):
                raise RuntimeError("existing scored shard does not match this request")
            if source.exists():
                if file_digest(source) != manifest["source"]["sha256"]:
                    raise RuntimeError("raw shard changed since scoring")
                source.unlink()
                sync_directory(source.parent)
            return output_path
        source_digest = file_digest(source)
        sampled, specification = score_play_positions(config, source)
        _publish(config, source, source_digest, specification, scoring, output_path,
                 manifest_path, sampled, cfg, specification["settings"]["seed"])
        source.unlink()
        sync_directory(source.parent)
    return output_path


def _publish(config, source, source_digest, specification, scoring, output_path,
             manifest_path, sampled, cfg, seed):
    # position selection: deduplicate and apply forced-error and evaluation filtering.
    selected, selection = select_play_positions(
        sampled.positions, cfg, random.Random(seed)
    )
    # serialization: attach game metadata and cached model analyses to selected positions.
    games_by_id = {row["game"]: row for row in sampled.games}
    rows = []
    for row in selected:
        game = games_by_id[row.game]
        record = {
            "record_id": f"play_s{seed}_g{row.game:08d}_p{row.ply:03d}",
            "fen": row.fen,
            "history": list(row.history),
            "extra": {
                "source_kind": "self_play",
                "normalized_fen": normalized_fen(row.fen),
                "seed": seed,
                "game": row.game,
                "ply": row.ply,
                "mover": row.mover,
                "model_white": row.model_white,
                "opponent_stockfish_nodes": row.opponent_nodes,
                "oracle_stockfish_nodes": config.oracle_nodes,
                "oracle_move_uci": row.oracle_move,
                "played_move_uci": row.played_move,
                "winrate_drop": row.winrate_drop,
                "oracle_cp": row.oracle_cp,
                "played_cp": row.played_cp,
                "eval_bucket": _play_eval_bucket(row.oracle_cp),
                "forced_keep": row.winrate_drop >= float(
                    cfg.get("force_keep_winrate_drop", 0.05)
                ),
                "move_source": row.move_source,
                "game_result": game["result"],
                "game_termination": game["termination"],
            },
        }
        if row.cached_root is not None:
            record["cached_root"] = row.cached_root
        rows.append(record)

    # output: write the selected JSONL and a reproducibility manifest beside it.
    digest = publish_jsonl(output_path, rows, allow_identical=True)
    manifest = {
        "mode": "play",
        "seed": seed,
        "source": {"path": str(source), "sha256": source_digest},
        "configuration": {"gameplay": specification, "scoring": scoring},
        "play_statistics": sampled.statistics,
        "selection": selection,
        "output": {"path": str(output_path), "rows": len(rows), "sha256": digest},
    }
    _atomic_json(manifest_path, manifest)
    # Publish both files durably and verify the data before deleting this artifact only.
    for path in (output_path, manifest_path):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    sync_directory(output_path.parent)
    if file_digest(output_path) != digest or json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("gameplay output verification failed; raw shard retained")
    print(json.dumps(manifest, indent=2, sort_keys=True), file=sys.stderr)
