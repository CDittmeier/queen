"""Streaming inputs, durable JSON, and deterministic shard rebalancing."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from datagen.self_distill.play_storage import sync_directory

from datagen.self_distill.schema import (
    accepted_identity,
    normalize_seed,
    partition_for,
    position_identity,
)


def atomic_json(path: Path, value: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    separators = (",", ":") if compact else None
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=None if compact else 2, separators=separators) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


@contextmanager
def stop_after_batch():
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def resolve_jsonls(patterns: Iterable[str], accepted_name: str | None = None) -> list[Path]:
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        if not matches:
            raise FileNotFoundError(f"no inputs matched {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path.is_dir():
                if accepted_name is None:
                    raise ValueError(f"expected a JSONL, got directory {path}")
                path = path / accepted_name
            if not path.is_file():
                raise FileNotFoundError(path)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _input_rows(path: Path):
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if raw.strip():
                yield line_number, raw, json.loads(raw)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seeds(
    paths: list[Path],
    part: int,
    parts: int,
    limit: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Normalize, deduplicate, and deterministically partition seed JSONLs."""
    selected = []
    seen_positions = set()
    eligible = 0
    source_manifests = []
    rng = random.Random(seed)

    # input scan: hash every source while selecting this task's stable partition.
    for path in paths:
        rows = 0
        for line_number, raw, value in _input_rows(path):
            rows += 1
            record = normalize_seed(value, path, line_number)
            identity = position_identity(record)
            if partition_for(identity, parts) != part or identity in seen_positions:
                continue
            seen_positions.add(identity)
            eligible += 1
            record["position_id"] = identity
            if limit <= 0 or len(selected) < limit:
                selected.append(record)
            else:
                replacement = rng.randrange(eligible)
                if replacement < limit:
                    selected[replacement] = record
        source_manifests.append(
            {"path": str(path), "rows": rows, "sha256": _file_sha256(path)}
        )
    rng.shuffle(selected)
    manifest = {
        "sources": source_manifests,
        "partition": {"index": part, "count": parts},
        "eligible_unique_positions": eligible,
        "selected_positions": len(selected),
        "limit": limit,
        "seed": seed,
    }
    manifest["identity"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return selected, manifest


def truncate_file(path: Path, committed_bytes: int) -> None:
    if committed_bytes < 0 or (path.exists() and path.stat().st_size < committed_bytes):
        raise RuntimeError(f"{path} is shorter than its committed offset or offset is negative")
    if not path.exists():
        if committed_bytes:
            raise RuntimeError(f"checkpoint expects {committed_bytes} bytes in missing {path}")
        path.write_text("")
        return
    with path.open("r+b") as handle:
        handle.truncate(committed_bytes)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_record_digest(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def rebalance(
    inputs: list[Path],
    output: Path,
    shards: int | None,
    rows_per_shard: int | None,
    overwrite: bool,
) -> dict:
    """Deduplicate accepted records and repartition their original JSON bytes."""
    if (shards is None) == (rows_per_shard is None):
        raise ValueError("choose exactly one of shards or rows_per_shard")
    if shards is not None and shards <= 0:
        raise ValueError("shards must be positive")
    if rows_per_shard is not None and rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    inputs = [path.resolve() for path in inputs]
    output = output.resolve()
    if any(path.is_relative_to(output) for path in inputs):
        raise ValueError("rebalance output must not contain its inputs")
    if overwrite:
        raise ValueError("rebalance overwrite is disabled; choose a new output directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)

    # first pass: validate accepted records and detect exact or conflicting duplicates.
    identities = {}
    source_manifests = []
    nonaccepted = 0
    duplicate_rows = 0
    for path in inputs:
        before = _file_sha256(path)
        input_rows = accepted_rows = 0
        for _, raw, record in _input_rows(path):
            input_rows += 1
            if record.get("outcome") != "accept":
                nonaccepted += 1
                continue
            accepted_rows += 1
            identity = accepted_identity(record)
            digest = _canonical_record_digest(record)
            if identity in identities:
                if identities[identity] != digest:
                    raise ValueError(f"conflicting accepted records for identity {identity}")
                duplicate_rows += 1
            else:
                identities[identity] = digest
        source_manifests.append(
            {
                "path": str(path),
                "rows": input_rows,
                "accepted_rows": accepted_rows,
                "sha256": _file_sha256(path),
            }
        )
        if source_manifests[-1]["sha256"] != before:
            raise RuntimeError(f"rebalance input changed during first pass: {path}")
    unique_rows = len(identities)
    shard_count = shards or max(1, math.ceil(unique_rows / rows_per_shard))

    # second pass: preserve each first-seen JSON line and assign it by stable identity hash.
    shard_paths = [temporary / f"shard_{index:05d}.jsonl" for index in range(shard_count)]
    handles = [path.open("wb") for path in shard_paths]
    hashes = [hashlib.sha256() for _ in shard_paths]
    counts = [0 for _ in shard_paths]
    emitted = set()
    try:
        for path, source in zip(inputs, source_manifests, strict=True):
            if _file_sha256(path) != source["sha256"]:
                raise RuntimeError(f"rebalance input changed between passes: {path}")
            for _, raw, record in _input_rows(path):
                if record.get("outcome") != "accept":
                    continue
                identity = accepted_identity(record)
                if identity not in identities or _canonical_record_digest(record) != identities[identity]:
                    raise RuntimeError(f"rebalance record changed between passes: {path}")
                if identity in emitted:
                    continue
                if not raw.endswith(b"\n"):
                    raise ValueError(f"JSONL record lacks a trailing newline in {path}")
                index = partition_for(identity, shard_count)
                handles[index].write(raw)
                hashes[index].update(raw)
                counts[index] += 1
                emitted.add(identity)
            if _file_sha256(path) != source["sha256"]:
                raise RuntimeError(f"rebalance input changed during second pass: {path}")
        if emitted != set(identities):
            raise RuntimeError("rebalance lost records between passes")
    finally:
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    manifest = {
        "schema_version": 1,
        "mode": "rebalance",
        "sources": source_manifests,
        "unique_accepted_records": unique_rows,
        "duplicate_records_removed": duplicate_rows,
        "nonaccepted_records_skipped": nonaccepted,
        "partition": {
            "method": "blake2b_identity_modulo",
            "shards": shard_count,
            "requested_rows_per_shard": rows_per_shard,
        },
        "outputs": [
            {
                "path": path.name,
                "rows": count,
                "sha256": digest.hexdigest(),
            }
            for path, count, digest in zip(shard_paths, counts, hashes)
        ],
    }
    atomic_json(temporary / "manifest.json", manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    sync_directory(temporary)
    temporary.replace(output)
    sync_directory(output.parent)
    return manifest
