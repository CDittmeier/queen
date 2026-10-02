# `datagen/tasks/` — per-task QA generators

Each module owns one Stage-1 task: the entity space it samples over, the
weighter that picks which entity to query, and the prose / parse_tag
rendering for a single fixed entity. Registered in
[`__init__.py:TASKS`](__init__.py).

| Task | Entity | MAX_UNIQUE_QUERIES | Notes |
|---|---|---|---|
| `piece_on_square` | board sq (0..63) | 64 | answer = piece on that square (or `<EMPTY>`) |
| `square_of_piece` | piece species | 12 | answer = list of squares holding that piece |
| `piece_on_file`   | file tuple | 8 | line-task: piece multiset on the file |
| `piece_on_rank`   | rank tuple | 8 | line-task: piece multiset on the rank |
| `piece_on_diagonal` | diagonal tuple | 26 (13 up-right + 13 up-left) | line-task; length-proportional pick |
| `piece_count`     | none | 1 | deterministic — both sides listed |
| `material_count`  | none | 1 | deterministic — both sides listed |
| `piece_moves`     | own piece (by square) | 16 | stage-2; lists the piece's legal moves (prose CoT + flattened tag), species-balanced |
| `square_attackers` | board sq (0..63) | 64 | stage-2; lists attackers (+ defenders if occupied); 50% empty / 50% occupant-balanced |
| `checking_moves`  | none | 1 | stage-2; legal moves that give check to the opponent king (mate case = empty list) |
| `capture_moves`   | none | 1 | stage-2; legal capture moves available to side-to-move |
| `check_parries`   | none | 1 | stage-2; in-check positions only — all legal parries, classified capture→block→king |
| `is_checkmate`    | none | 1 | stage-2; yes/no on mate with CoT (not_check / stalemate / check_parry / checkmate branches) |
| `hce_motif_annotation` | mined motif record | 1 | stage-5.motif; game rows use 0--8-ply lookahead + seven LC0 history states, puzzle rows use their genuine setup history, globally ordered parseable motif bullets |

---

## Module contract

Each task module exposes exactly:

```python
NAME: str                                              # e.g. "piece_on_square"
MAX_UNIQUE_QUERIES: int                                # max distinct queries per position

def _choose_entity(board, frequency, rng, exclude: set) -> EntityT:
    """Task-specific weighter over `entities \\ exclude`. Read-only on
    `frequency`; called once per query."""

def _render(entity, board, rng) -> dict:
    """Template + prose rendering for a fixed entity. Returns one record:
        {question, answer, question_type, answer_class}
    """

def sample_n(board, frequency, rng, n: int) -> list[dict]:
    """Standard loop — accumulates `seen`, repeatedly calls _choose_entity
    + _render until n records are collected (capped at MAX_UNIQUE_QUERIES)."""
```

`sample_one(board, frequency, rng)` and
`sample_all(board, frequency, rng)` are **not implemented per task** —
they're synthesized in [`__init__.py`](__init__.py) from `sample_n` +
`MAX_UNIQUE_QUERIES` and attached to each module at import time. Call
sites use them exactly as if they were defined in the module:

```python
from datagen.tasks.piece_on_square import sample_one, sample_all
# both work — provided by __init__.py
```

This removes ~12 lines of identical boilerplate from each task.

### Forward (stage-3/4 lookahead) API — also synthesized

The forward API is opt-in and synthesized in [`__init__.py`](__init__.py)
(`_attach_forward`), so no task implements it by hand. A task opts in with:

```python
SUPPORTS_FORWARD = True
```

The synthesizer then attaches:

```python
def _render_forward(entity, board, rng, move_sequence) -> dict:
    """`board` is the INITIAL position, `move_sequence` the played continuation.
    Renders `entity` on the FINAL board via the task's own `_render`, then frames
    it: the prompt gets an "After the moves <m1>, ..., and <mN>, ..." prefix
    (parse-tag-move format, oxford-comma divider); the answer leads with the
    move-by-move CoT, then the usual body + parse tag — all on the FINAL board,
    POV-anchored to the INITIAL side-to-move so tokens don't flip mid-sequence."""

def sample_n_forward(board, move_sequence, frequency, rng, n) -> list[dict]:
    """Forward analogue of sample_n (final board + move dicts computed once)."""
```

**Distinctness balancing (optional).** A forward-capable task that wants ~50/50
changed-vs-unchanged answers additionally defines:

```python
def _choose_entity_forward(board, final, frequency, rng, exclude) -> EntityT:
    """Pick the changed/unchanged bucket first (frequency-balanced coin over the
    DISTINCT_CHANGED/DISTINCT_SAME counters), then balance within it."""

def _distinctness_changed(board, final, entity) -> bool:
    """Did this entity's answer change over the sequence?"""
```

When present, the synthesized `sample_n_forward` uses `_choose_entity_forward`
and appends the distinctness marker (`<<changed>>`/`<<unchanged>>`) to each
record's `answer_class`. Only `piece_on_square` and `square_of_piece` do this;
the others (line tasks, counts) fall back to the static `_choose_entity` since
their natural changed rate is already well-mixed (~40–55%).

### The `frequency` dict

A shared `dict[str, int]` counter the driver maintains per task per split.
After each call to `sample_n`, the driver walks each returned record's
`answer_class` and bumps the counter for every token. The next
`_choose_entity` call reads these counts to shape weights — that's what
keeps the answer-class distribution balanced across many calls.

For tasks whose weighter doesn't read frequency
(`piece_on_file`/`rank`/`diagonal`, `piece_count`, `material_count`), the
arg is accepted but ignored.

### The `seen` set inside `sample_n`

`sample_n(board, frequency, rng, n)` ensures the `n` returned records
query `n` *distinct* entities for that position. The pattern:

```python
def sample_n(board, frequency, rng, n):
    n = min(n, MAX_UNIQUE_QUERIES)
    seen, out = set(), []
    while len(out) < n:
        e = _choose_entity(board, frequency, rng, exclude=seen)
        seen.add(e)
        out.append(_render(e, board, rng))
    return out
```

The per-task weighter re-runs over `entities \ exclude` on every call, so
class balance is preserved across the n draws.

---

## Adding a new task

1. Create `datagen/tasks/<new_task>.py` with:
   - `NAME`, `MAX_UNIQUE_QUERIES`
   - `_choose_entity(board, frequency, rng, exclude)` — weighted pick
   - `_render(entity, board, rng)` — template + prose
   - `sample_n(board, frequency, rng, n)` — the standard loop above
2. Register in `__init__.py`:
   - import the module
   - add to `_TASK_MODULES`
3. If the task needs a non-default grader, add it to
   `utils/eval_utils.py:_GRADERS`.

`sample_one` / `sample_all` are attached automatically by
`__init__.py:_attach_wrappers`.
