"""Caller-named gameplay checkpoints and durable, single-writer shard publication."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Hash artifact contents, not machine-local paths or modification times."""
    if path.is_file():
        return file_digest(path)
    files = sorted(item for item in path.rglob("*") if item.is_file()
                   and not any(part.startswith(".") for part in item.relative_to(path).parts))
    inventory = [(str(item.relative_to(path)), file_digest(item)) for item in files]
    return identity(inventory)


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class PlayStore:
    """One artifact lock covers recovery, appending, reading, or precise cleanup.

    Lock files live outside artifact directories and are never unlinked: deleting
    a live lock's inode would let a second process lock a different inode.
    """

    streams = ("positions", "games")

    def __init__(self, scratch_dir: Path, job: str, specification: dict | None = None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job):
            raise ValueError("job must be a single safe directory name")
        self.root = scratch_dir.resolve()
        self.path = self.root / job
        self.job = job
        self.artifact_id = identity(specification) if specification is not None else None
        self.specification = specification
        self.state = None
        self.handles = {}
        self.lock = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        locks = self.root / ".locks"
        locks.mkdir(exist_ok=True)
        self.lock = (locks / f"{self.job}.lock").open("a+b")
        try:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"gameplay artifact is in use: {self.artifact_id}") from error
            self._recover()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _recover(self) -> None:
        if self.path.is_symlink():
            raise ValueError(f"refusing symlinked artifact directory: {self.path}")
        manifest = self.path / "manifest.json"
        if not manifest.exists():
            if self.specification is None:
                raise FileNotFoundError(manifest)
            self.path.mkdir(exist_ok=True)
            if any(self.path.glob("*.json*")):
                raise RuntimeError(f"gameplay files exist without a manifest: {self.path}")
            _atomic_json(manifest, {"artifact_id": self.artifact_id, "specification": self.specification})
            sync_directory(self.root)
        for name in ("manifest.json", "state.json", "positions.jsonl", "games.jsonl"):
            if (self.path / name).is_symlink():
                raise ValueError(f"refusing symlinked gameplay file: {name}")
        saved = json.loads(manifest.read_text())
        if saved["artifact_id"] != identity(saved["specification"]):
            raise RuntimeError("gameplay manifest identity mismatch")
        if self.specification is not None and self.specification != saved["specification"]:
            raise RuntimeError("gameplay specification mismatch")
        self.specification = saved["specification"]
        self.artifact_id = saved["artifact_id"]
        state_path = self.path / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            if self.state["artifact_id"] != self.artifact_id:
                raise RuntimeError("gameplay state identity mismatch")
        for name in self.streams:
            path = self.path / f"{name}.jsonl"
            expected = self.state["committed_bytes"][name] if self.state else 0
            actual = path.stat().st_size if path.exists() else 0
            if expected < 0 or actual < expected:
                raise RuntimeError(f"{path} is shorter than its committed offset {expected}")
            if self.state is None and actual:
                raise RuntimeError(f"gameplay records exist without a state file: {path}")
            if actual > expected:
                with path.open("r+b") as handle:
                    handle.truncate(expected)
                    handle.flush()
                    os.fsync(handle.fileno())

    def append(self, name: str, record: dict) -> None:
        if self.complete:
            raise RuntimeError("cannot append to a complete gameplay artifact")
        self._handle(name).write(json.dumps(record, separators=(",", ":")).encode() + b"\n")

    def _handle(self, name: str):
        if name not in self.streams:
            raise ValueError(f"unknown gameplay stream: {name}")
        if name not in self.handles:
            self.handles[name] = (self.path / f"{name}.jsonl").open("ab")
        return self.handles[name]

    @property
    def complete(self) -> bool:
        return bool(self.state and self.state["complete"])

    def checkpoint(self, state: dict) -> None:
        committed = {}
        for name in self.streams:
            handle = self._handle(name)
            handle.flush()
            os.fsync(handle.fileno())
            committed[name] = handle.tell()
        state = {**state, "artifact_id": self.artifact_id, "committed_bytes": committed}
        _atomic_json(self.path / "state.json", state)
        self.state = state

    def records(self, name: str):
        if not self.complete:
            raise RuntimeError("gameplay must finish before downstream processing")
        if name not in self.streams:
            raise ValueError(f"unknown gameplay stream: {name}")
        with (self.path / f"{name}.jsonl").open() as handle:
            for line in handle:
                yield json.loads(line)

    def delete(self) -> None:
        """Caller must first publish and verify its durable downstream output."""
        if not self.complete:
            raise RuntimeError("refusing to delete incomplete gameplay")
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        shutil.rmtree(self.path)
        sync_directory(self.root)

    def __exit__(self, *_):
        try:
            for handle in self.handles.values():
                handle.close()
        finally:
            if self.lock is not None:
                self.lock.close()


@contextmanager
def shard_lock(games_path: Path):
    """One persistent lock inode protects both phases of a named shard."""
    if not games_path.name.endswith(".games.jsonl"):
        raise ValueError("raw shard path must end in .games.jsonl")
    games_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = games_path.with_name("." + games_path.name.removesuffix(".games.jsonl") + ".lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"shard is in use: {games_path}") from error
        yield


def scored_path(games_path: Path) -> Path:
    if not games_path.name.endswith(".games.jsonl"):
        raise ValueError("raw shard path must end in .games.jsonl")
    return games_path.with_name(games_path.name.removesuffix(".games.jsonl") + ".scored.jsonl")


def publish_jsonl(path: Path, rows, *, allow_identical: bool = False) -> str:
    """Fsync a complete temporary file, then publish without clobbering a shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                line = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                handle.write(line)
                digest.update(line)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard link is atomic and fails if the target already exists.
        if allow_identical and path.exists():
            if file_digest(path) != digest.hexdigest():
                raise FileExistsError(f"existing shard differs from regenerated output: {path}")
        else:
            os.link(temporary, path)
        sync_directory(path.parent)
        if file_digest(path) != digest.hexdigest():
            raise RuntimeError(f"published shard verification failed: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()
