"""Run an Elo ladder end to end: play the configured matches, then fit Elos.

Config (JSON)::

    {
      "anchors":  {"random_legal": 0},          # engines pinned to a fixed Elo
      "engines":  [ {engine spec}, ... ],        # see eval.engines.build_player
      "pairings": "chain"                        # or "round_robin",
                                                 # or "star:<name>",
                                                 # or [["a","b"], ...]
    }

Each pairing is a 10-game match (5 openings x 2 colors). Outputs
``<out>/elos.json`` (ratings + CI95 + per-match scores) and ``<out>/games.pgn``.
Resumable at match granularity: a pairing already at 10 games is skipped, so a
killed run just re-launches. Engines are opened lazily and closed as soon as
their last pairing is done, so an lc0 net loads its weights once and a large
ladder never holds every GPU engine open at the same time.

    python -m eval.ladder --config eval/configs/example_ladder.json --out runs/ladder
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .elo import Match, fit_elos
from .engines import REPO_ROOT, Player, build_player
from .match import play_match


def _key(a: str, b: str) -> str:
    return f"{a}||{b}"


def _resolve_specs(engines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make lc0 weights paths absolute (relative paths resolve from repo root)."""
    out = []
    for spec in engines:
        spec = dict(spec)
        if spec.get("kind") == "lc0":
            w = Path(spec["weights"])
            spec["weights"] = str(w if w.is_absolute() else REPO_ROOT / w)
        out.append(spec)
    return out


def _pairings(pairing_spec: Any, names: list[str]) -> list[tuple[str, str]]:
    if isinstance(pairing_spec, list):
        return [(a, b) for a, b in pairing_spec]
    if pairing_spec == "chain":
        return list(zip(names, names[1:]))
    if pairing_spec == "round_robin":
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if isinstance(pairing_spec, str) and pairing_spec.startswith("star:"):
        hub = pairing_spec.split(":", 1)[1]
        return [(hub, n) for n in names if n != hub]
    raise ValueError(f"bad pairings: {pairing_spec!r}")


def _load_matches(elos_path: Path) -> dict[str, dict[str, Any]]:
    if not elos_path.exists():
        return {}
    try:
        return json.loads(elos_path.read_text(encoding="utf-8")).get("matches", {})
    except Exception:
        return {}


def _write(
    elos_path: Path,
    specs: list[dict[str, Any]],
    anchors: dict[str, float],
    matches: dict[str, dict[str, Any]],
    ratings: dict[str, dict] | None = None,
) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "anchors": anchors,
        "engines": specs,
        "matches": matches,
        "ratings": ratings or {},
    }
    elos_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_ladder(config: dict[str, Any], out_dir: str | Path) -> dict[str, dict]:
    specs = _resolve_specs(config["engines"])
    by_name = {s["name"]: s for s in specs}
    names = list(by_name)
    anchors = {k: float(v) for k, v in config.get("anchors", {}).items()}
    pairings = _pairings(config.get("pairings", "chain"), names)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    elos_path = out_dir / "elos.json"
    pgn_path = out_dir / "games.pgn"

    matches = _load_matches(elos_path)
    todo = [(a, b) for a, b in pairings if matches.get(_key(a, b), {}).get("games", 0) < 10]
    remaining = Counter(x for pair in todo for x in pair)

    pool: dict[str, Player] = {}

    def acquire(name: str) -> Player:
        if name not in pool:
            pool[name] = build_player(by_name[name])
        return pool[name]

    def release(name: str) -> None:
        player = pool.pop(name, None)
        if player is not None:
            player.close()

    try:
        for a, b in todo:
            print(f"[match] {a} vs {b}", flush=True)
            result = play_match(acquire(a), acquire(b))
            matches[_key(a, b)] = {
                "a": a, "b": b, "games": result["games"], "score_a": result["score_a"]
            }
            with pgn_path.open("a", encoding="utf-8") as fh:
                for rec in result["records"]:
                    fh.write(rec["pgn"] + "\n\n")
            _write(elos_path, specs, anchors, matches)  # checkpoint after every match
            for x in (a, b):
                remaining[x] -= 1
                if remaining[x] <= 0:
                    release(x)
    finally:
        for name in list(pool):
            release(name)

    ratings = fit_elos(
        (Match(m["a"], m["b"], m["score_a"], m["games"]) for m in matches.values()),
        anchors,
    )
    _write(elos_path, specs, anchors, matches, ratings)
    return ratings


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an engine Elo ladder and rate it.")
    parser.add_argument("--config", required=True, help="ladder config JSON")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ratings = run_ladder(config, args.out)
    for name, r in sorted(ratings.items(), key=lambda kv: kv[1]["elo"]):
        ci = "" if r["ci95_low"] is None else f"  [{r['ci95_low']:.0f}, {r['ci95_high']:.0f}]"
        tag = "  (anchor)" if r["anchored"] else ""
        print(f"{name:<28} {r['elo']:7.1f}{ci}{tag}")


if __name__ == "__main__":
    main()
