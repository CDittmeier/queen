"""Concatenate per-task train arrows into a single ``train_all.arrow``.

Reads ``<in-dir>/training_datasets/*.arrow`` (each a HuggingFace arrow with
its own ``dataset_config.json``), concatenates them in sorted-filename order,
and writes ``<in-dir>/train_all.arrow`` plus a top-level ``dataset_config.json``
that carries forward the shared ``pov`` flag and lists the task names.

The trainer reads only ``pov`` from this top-level config; everything else is
diagnostic. Per-stage call sites (the stage-1 / stage-2 sbatches) used to
inline a heredoc that did exactly this — kept as a real module for editor /
import / test visibility.

Usage:

    python -m datagen.build_train_all --in-dir data/stage1
    python -m datagen.build_train_all --in-dir data/stage2 --note "..."
"""
import argparse
import glob
import json
import os

from datasets import concatenate_datasets, load_from_disk


def build_train_all(in_dir: str, note: str | None = None) -> str:
    """Concatenate per-task arrows under ``in_dir`` into ``in_dir/train_all.arrow``.

    Asserts every per-task ``dataset_config.json`` agrees on ``pov``; the shared
    value is written through to the top-level config (silent "first one wins"
    would mask a mismatch). Returns the output path.
    """
    parts = sorted(glob.glob(f"{in_dir}/training_datasets/*.arrow"))
    if not parts:
        raise FileNotFoundError(f"no per-task arrows found under {in_dir}/training_datasets/")

    dss = [load_from_disk(p) for p in parts]
    print("[concat] per-task sizes:", {os.path.basename(p): len(d) for p, d in zip(parts, dss)})

    povs = [json.load(open(f"{p}/dataset_config.json"))["pov"] for p in parts]
    assert len(set(povs)) == 1, (
        f"per-task pov disagreement under {in_dir}: "
        + ", ".join(f"{os.path.basename(p)}={v}" for p, v in zip(parts, povs))
    )
    pov = povs[0]

    combined = concatenate_datasets(dss)
    out = f"{in_dir}/train_all.arrow"
    combined.save_to_disk(out)

    config = {
        "pov":      pov,
        "split":    "train",
        "records":  len(combined),
        "tasks":    [os.path.basename(p).replace(".arrow", "") for p in parts],
    }
    if note:
        config["note"] = note
    with open(f"{out}/dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[concat] -> {out}  ({len(combined):,} records, pov={pov})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--in-dir", required=True,
                   help="Stage directory containing training_datasets/*.arrow (e.g. data/stage1).")
    p.add_argument("--note", default=None,
                   help="Optional note string written into the top-level dataset_config.json.")
    args = p.parse_args()
    build_train_all(args.in_dir, note=args.note)


if __name__ == "__main__":
    main()
