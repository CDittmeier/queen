"""Task registry: question_type name -> task module.

Each task module exposes:
    NAME: str
    MAX_UNIQUE_QUERIES: int
    sample_n(board, frequency, rng, n) -> list[dict]
    _choose_entity / _render          — internal helpers

`sample_one` and `sample_all` are trivial wrappers over `sample_n` (identical
across every task), so they're synthesized here and attached to each module
at import time rather than duplicated in seven files. Existing call sites
(`module.sample_one(...)`, `module.sample_all(...)`) keep working unchanged.

`_render_forward` and `sample_n_forward` (stage-3/4 lookahead) are likewise
synthesized — the renderer and sampler loops are identical across the
forward-capable tasks (those declaring `SUPPORTS_FORWARD = True`). A task that
wants distinctness-balanced forward sampling additionally declares
`_choose_entity_forward` + `_distinctness_changed`; the rest fall back to the
static `_choose_entity`. See `_attach_forward`.
"""
from datagen.position_features import final_board_repr, sequence_move_dicts
from datagen.prose import compose_forward, distinctness_marker, parts_from_render

from datagen.tasks import (
    capture_moves,
    check_parries,
    checking_moves,
    hce_motif_annotation,
    hce_rationale,
    is_checkmate,
    plan_eval,
    material_count,
    piece_count,
    piece_moves,
    piece_on_diagonal,
    piece_on_file,
    piece_on_rank,
    piece_on_square,
    sim_rationale,
    square_attackers,
    square_of_piece,
    verdict_eval,
)

_TASK_MODULES = [
    piece_on_square,
    square_of_piece,
    piece_on_file,
    piece_on_rank,
    piece_on_diagonal,
    piece_count,
    material_count,
    piece_moves,
    square_attackers,
    checking_moves,
    capture_moves,
    check_parries,
    is_checkmate,
    hce_motif_annotation, # stage-5.motif: render pre-mined structured JSONL rows
    hce_rationale,        # stage-5: answer produced by the search pipeline, not sample_n
    verdict_eval,         # stage-5.a: answer produced by the verdict classifier
    plan_eval,            # stage-5.b: plans derived from stored selfplay games
    sim_rationale,        # stage-5.sim: GPU-sharded plan-model search narrative
]


def _attach_wrappers(module) -> None:
    """Synthesize sample_one + sample_all from the module's sample_n + MAX_UNIQUE_QUERIES."""
    def sample_one(board, frequency, rng):
        return module.sample_n(board, frequency, rng, 1)[0]

    def sample_all(board, frequency, rng):
        return module.sample_n(board, frequency, rng, module.MAX_UNIQUE_QUERIES)

    module.sample_one = sample_one
    module.sample_all = sample_all


def _attach_forward(module) -> None:
    """Synthesize the stage-3/4 forward API (`_render_forward`, `sample_n_forward`)
    from the module's static `_render` + `_choose_entity`.

    `_render_forward` renders `entity` on the FINAL board reached after the move
    sequence (via the module's own `_render`), then frames it with the sequence
    prompt + move-by-move CoT. `sample_n_forward` is the standard accumulate loop.

    Two opt-in hooks, independent of each other:
      * `_choose_entity_forward(board, final, move_sequence, frequency, rng, exclude)`
        — a forward-aware entity chooser (picks on the FINAL board, may backtrack
        through the sequence). Returns None when no fresh entity remains, which
        stops the loop. Modules without it fall back to the static `_choose_entity`
        on the presented board (fine when the entity is board-agnostic — a species,
        file or rank — but not when it is a square whose occupant the moves change).
      * `_distinctness_changed(board, final, entity) -> bool` — when present, each
        record's answer_class is tagged with the changed/unchanged marker so the
        builder can balance the bucket ~50/50. Independent of the chooser: a task
        may choose on the final board without wanting distinctness balancing."""
    has_fwd_chooser = hasattr(module, "_choose_entity_forward")
    has_distinct = hasattr(module, "_distinctness_changed")

    def render_forward(entity, board, rng, move_sequence):
        final = final_board_repr(board, move_sequence)
        parts = parts_from_render(module._render(entity, final, rng))
        return compose_forward(parts, sequence_move_dicts(board, move_sequence), rng)

    def sample_n_forward(board, move_sequence, frequency, rng, n):
        n = min(n, module.MAX_UNIQUE_QUERIES)
        final = final_board_repr(board, move_sequence)
        move_dicts = sequence_move_dicts(board, move_sequence)
        seen = set()
        out = []
        while len(out) < n:
            if has_fwd_chooser:
                e = module._choose_entity_forward(board, final, move_sequence,
                                                  frequency, rng, exclude=seen)
                if e is None:           # forward chooser exhausted the position
                    break
            else:
                # `None` from the static chooser is the "no entity" sentinel
                # (e.g. checking_moves) — render once, as the static path does.
                e = module._choose_entity(board, frequency, rng, exclude=seen)
            seen.add(e)
            render = module._render(e, final, rng)
            if has_distinct:
                changed = module._distinctness_changed(board, final, e)
                render["answer_class"] = render["answer_class"] + [distinctness_marker(changed)]
            out.append(compose_forward(parts_from_render(render), move_dicts, rng))
        return out

    module._render_forward = render_forward
    module.sample_n_forward = sample_n_forward


for _m in _TASK_MODULES:
    if (getattr(_m, "PARALLEL_ANALYZE", False)
            or getattr(_m, "MINED_RECORDS", False)):
        continue                    # analysis tasks answer via the search pipeline, not sample_n
    _attach_wrappers(_m)
    if getattr(_m, "SUPPORTS_FORWARD", False):
        _attach_forward(_m)


TASKS = {m.NAME: m for m in _TASK_MODULES}
