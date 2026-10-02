"""Maximum-likelihood Elo from match scores.

Given pairwise match results and a set of pinned anchors (engines whose Elo is
held fixed — e.g. ``random_legal = 0``), fit every other engine's Elo by
maximizing the logistic (Bradley-Terry) likelihood. We use coordinate ascent:
repeatedly move each free engine to the 1-D likelihood maximum given the
current ratings of its opponents. One engine vs fixed-rating opponents reduces
to the classic 1-D scan. CI95 is the profile-likelihood interval (ratings whose
log-likelihood is within 1.92 of the maximum).

The rating search is bounded to ``[-200, 3600]``, which is what keeps a clean
sweep (e.g. stockfish_n1 beating random 10/10, an unbounded Elo gap) from
running off to infinity — it pins to the boundary instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

RATING_LO, RATING_HI = -200, 3600
_GRID = list(range(RATING_LO, RATING_HI + 1))
_CHI2_95_1DOF_HALF = 1.92  # half the 95% chi-square cutoff at 1 dof


@dataclass
class Match:
    a: str
    b: str
    score_a: float  # A's points over the match
    games: int


def expected_score(ra: float, rb: float) -> float:
    """Logistic expected score for A (rating ``ra``) vs B (rating ``rb``)."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _loglik(rating: float, opponents: list[tuple[float, float, int]]) -> float:
    """Bernoulli LL for one engine; opponents = [(score, opp_rating, games)]."""
    ll = 0.0
    for score, opp_rating, games in opponents:
        p = min(max(expected_score(rating, opp_rating), 1e-12), 1.0 - 1e-12)
        ll += score * math.log(p) + (games - score) * math.log(1.0 - p)
    return ll


def _argmax_rating(opponents: list[tuple[float, float, int]]) -> tuple[float, list[float]]:
    lls = [_loglik(float(r), opponents) for r in _GRID]
    best_i = max(range(len(lls)), key=lls.__getitem__)
    return float(_GRID[best_i]), lls


def fit_elos(
    matches: Iterable[Match],
    anchors: dict[str, float],
    *,
    max_sweeps: int = 100,
) -> dict[str, dict]:
    """Fit Elos for all engines appearing in ``matches`` (anchors held fixed)."""
    matches = list(matches)
    names = sorted({m.a for m in matches} | {m.b for m in matches} | set(anchors))
    elo = {n: float(anchors.get(n, 1500.0)) for n in names}
    free = [n for n in names if n not in anchors]

    # Per engine: the matches it played, as (its_score, opponent_name, games).
    played: dict[str, list[tuple[float, str, int]]] = {n: [] for n in names}
    for m in matches:
        played[m.a].append((m.score_a, m.b, m.games))
        played[m.b].append((m.games - m.score_a, m.a, m.games))

    for _ in range(max_sweeps):
        moved = False
        for n in free:
            opp = [(s, elo[o], g) for s, o, g in played[n]]
            if not opp:
                continue
            best, _ = _argmax_rating(opp)
            if best != elo[n]:
                elo[n] = best
                moved = True
        if not moved:
            break

    out: dict[str, dict] = {}
    for n in names:
        if n in anchors or not played[n]:
            out[n] = {"elo": elo[n], "ci95_low": None, "ci95_high": None,
                      "anchored": n in anchors}
            continue
        opp = [(s, elo[o], g) for s, o, g in played[n]]
        _, lls = _argmax_rating(opp)
        cutoff = max(lls) - _CHI2_95_1DOF_HALF
        valid = [_GRID[i] for i, ll in enumerate(lls) if ll >= cutoff]
        out[n] = {"elo": elo[n], "ci95_low": float(min(valid)),
                  "ci95_high": float(max(valid)), "anchored": False}
    return out
