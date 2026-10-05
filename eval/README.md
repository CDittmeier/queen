# `eval/` — engine Elo benchmarking

Rate chess engines on a shared Elo ladder by playing them against each other and
fitting a logistic (Bradley-Terry) model. Supports **Stockfish** (by node
budget), **lc0 networks** (by weights file), and a **random-legal** baseline that
anchors the floor at 0 Elo.

This is a compact reconstruction of the earlier calibration suite
(`data/tests/engine_elo_calibration/`). It deliberately supports only engines for
now; LLM move-pickers plug in later as a new `Player` subclass (see below).

## Layout

| file | purpose |
|---|---|
| `engines.py` | `Player` ABC + `RandomPlayer`, `UciPlayer`, and `stockfish()` / `lc0()` factories; `build_player(spec)` from config |
| `match.py`   | fixed openings, `play_game`, `play_match` (10 games = 5 openings × 2 colors, 400-ply cap, forfeit on error/illegal) |
| `elo.py`     | `fit_elos(matches, anchors)` — joint MLE via coordinate ascent, profile-likelihood CI95 |
| `ladder.py`  | orchestration + CLI: play a configured set of matches (resumable), then rate |

## Run

```bash
python -m eval.ladder --config eval/configs/example_ladder.json --out runs/ladder
```

Writes `runs/ladder/elos.json` (ratings + CI95 + per-match scores) and
`runs/ladder/games.pgn`. Re-running resumes: any pairing already at 10 games is
skipped.

## Config

```json
{
  "anchors":  {"random_legal": 0},
  "pairings": "chain",
  "engines": [
    {"name": "random_legal",     "kind": "random",    "seed": 12345},
    {"name": "stockfish_n1000",  "kind": "stockfish", "nodes": 1000},
    {"name": "lc0_BT4",          "kind": "lc0",       "weights": "data/engines/lc0_networks/BT4-1740.pb.gz", "nodes": 1}
  ]
}
```

- **`anchors`** — engines pinned to a fixed Elo (held constant during the fit).
  Pin `random_legal = 0` for an absolute-ish floor, or `stockfish_n10000 = 2475`
  to anchor to a human rating.
- **`pairings`** — who plays whom: `"chain"` (consecutive engines in listed
  order), `"round_robin"` (all pairs), `"star:<name>"` (everyone vs one hub), or
  an explicit `[["a","b"], ...]` list. lc0 weights paths may be relative to the
  repo root.

## Engine binaries

Defaults resolve through repo-local symlinks (override with `$STOCKFISH_BIN` /
`$LC0_BIN`):

- `data/engines/stockfish_25080907_x64_avx2`
- `data/engines/lc0/build/release/lc0`

## Notes

- **Rating bounds** `[-200, 3600]`: a clean sweep (e.g. `stockfish_n1` beating
  `random` 10/10) is an unbounded Elo gap; the bound pins it to the boundary
  rather than ±∞, so the ladder floor is compressed/loose by construction.
- **Resume granularity** is one match; a match interrupted mid-way replays from
  game 1.

## Adding an LLM player later

Subclass `Player`, implement `choose_move(board) -> chess.Move` (and optionally
`new_game` / `close`), then add a `kind` branch to `build_player`. The match and
rating code are engine-agnostic and need no changes.
## Analysis-model evaluation

For the paper's model benchmarks use `python -m eval.benchmark --config
configs/eval/benchmark.yaml` (one command line). For the fixed 32-game LM Elo
test use `python -m eval.model_ladder --config configs/eval/model_ladder.yaml`.
The repository README documents input formats, exact metrics, opponents,
checkpoint exports, Slurm launchers, and resume behavior. The UCI-only tools
above remain for engine calibration; the batched LM runner does not require
modifying their per-move interface.
