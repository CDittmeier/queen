"""Mix N HuggingFace arrow datasets in a target proportion into one arrow.

Generalizes the previous stage-2/stage-1 90/10 mixer to N inputs with a
caller-supplied weight vector. Weights are post-normalized to a probability
distribution; the input with the largest weight is the "anchor" and is used
whole at its native row count, and the remaining inputs are uniformly shuffled
and trimmed to ``N_anchor * p_i / p_anchor`` rows so the realized mix matches
``--dist`` up to integer rounding.

Aborts if any non-anchor input lacks enough rows to hit its target (reduce
that input's weight or raise the anchor's). All inputs must agree on column
names and on the ``pov`` flag in their ``dataset_config.json`` — pov is
written through to the output.

    python -m datagen.mix_arrows \\
        --inputs data/stage2/train_all.arrow data/stage1/train_all.arrow \\
        --dist   9 1 \\
        --out    data/stage2/train_with_stage1_mix.arrow \\
        --seed   0
"""
import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def mix_arrows(inputs: list[str], dist: list[float], out: str, seed: int = 0) -> None:
    """Concatenate N arrows in proportion to ``dist`` (post-normalized).

    Anchor (largest-weight input) is used whole; each other input is
    sub-sampled to match its target share against the anchor. ``dist`` does
    not need to sum to 1 — it is normalized internally.
    """
    assert len(inputs) >= 2, f"need >= 2 inputs, got {len(inputs)}"
    assert len(dist) == len(inputs), (
        f"--dist length ({len(dist)}) must match --inputs ({len(inputs)})"
    )
    assert all(w > 0 for w in dist), f"all --dist weights must be > 0, got {dist}"

    s = sum(dist)
    probs = [w / s for w in dist]
    anchor = max(range(len(inputs)), key=lambda i: probs[i])
    p_anchor = probs[anchor]

    dsets = [load_from_disk(p) for p in inputs]
    anchor_cols = dsets[anchor].column_names
    for i, d in enumerate(dsets):
        if i == anchor:
            continue
        assert d.column_names == anchor_cols, (
            f"column mismatch: {inputs[i]} cols={d.column_names} "
            f"!= anchor {inputs[anchor]} cols={anchor_cols}"
        )

    n_anchor = len(dsets[anchor])
    pieces = []
    realized_counts: list[int] = []
    for i, d in enumerate(dsets):
        if i == anchor:
            pieces.append(d)
            realized_counts.append(n_anchor)
            continue
        target = round(n_anchor * probs[i] / p_anchor)
        assert target <= len(d), (
            f"need {target:,} rows from {inputs[i]} but only {len(d):,} available; "
            f"reduce its --dist weight or raise the anchor's"
        )
        pieces.append(d.shuffle(seed=seed).select(range(target)))
        realized_counts.append(target)

    mixed = concatenate_datasets(pieces)
    total = sum(realized_counts)
    print(f"[mix] anchor=#{anchor} ({inputs[anchor]})  rows={n_anchor:,}")
    for i, (p_in, n_taken, target) in enumerate(zip(inputs, realized_counts, probs)):
        realized = n_taken / total
        marker = " *" if i == anchor else "  "
        print(f"[mix] {marker} #{i}: {Path(p_in).name}  rows={n_taken:,}  "
              f"target={target:.4f}  realized={realized:.4f}")
    print(f"[mix] total={total:,}")

    out_path = Path(out)
    mixed.save_to_disk(str(out_path))

    # pov must agree across all inputs — silent "first one wins" would mask
    # a mismatch where the trainer reads the wrong embedding convention.
    povs = []
    for p in inputs:
        cfg = Path(p) / "dataset_config.json"
        assert cfg.exists(), f"missing dataset_config.json beside input {p}"
        povs.append(bool(json.load(open(cfg))["pov"]))
    assert len(set(povs)) == 1, (
        "pov disagreement across inputs: "
        + ", ".join(f"{Path(p).name}={v}" for p, v in zip(inputs, povs))
    )
    pov = povs[0]
    json.dump({"pov": pov}, open(out_path / "dataset_config.json", "w"))
    print(f"[mix] wrote {out_path}  (pov={pov})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths to >= 2 input arrow datasets.")
    ap.add_argument("--dist", nargs="+", type=float, required=True,
                    help="Weights (>0) per input, equal length to --inputs; "
                         "post-normalized to a probability distribution. "
                         "Integer or float; e.g. `--dist 9 1` == `--dist 0.9 0.1`.")
    ap.add_argument("--out", required=True, help="Output arrow path.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shuffle seed for non-anchor sub-samples.")
    args = ap.parse_args()
    mix_arrows(args.inputs, args.dist, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
