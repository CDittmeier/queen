"""Stage-5 task: a machine-readable HCE rationale for a position.

Unlike the QA tasks (which sample question/answer pairs off a BoardRepr), the answer here is
the full search+verbalization analysis of the presented position —
`get_tree(fen).string("token")` — a machine-readable narrative that never spells out pieces,
colors, files, ranks, diagonals, or moves in English (notation.py reuses the same BoardRepr
token vocabulary the QA tasks use: <PIECE_M*/O*>, <SQUARE_1..64>, <PIECE><from><to> moves).

Because get_tree is expensive (~5 s/position), the builder computes these across CPUs via
datagen/tree/parallelize.py rather than through a per-board `sample_n`. So this module carries
no sampler — it is a marker (`PARALLEL_ANALYZE`) + prompt that build_qa_dataset recognizes and
routes through the analysis pipeline. From the config's point of view it is just another task.
"""

NAME = "hce_rationale"
MAX_UNIQUE_QUERIES = 1
PARALLEL_ANALYZE = True     # build_qa_dataset routes this task through datagen.tree.parallelize
# The side to move is always the pov side (POV mode), which the QA tasks name with the plain
# word "player" (mover_label; there is no side-to-play special token). Match that convention.
PROMPT = "Analyze the position carefully and find the best move for player."
