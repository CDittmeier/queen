"""Config-backed, indexed tasks for the existing self-distillation launcher."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys

import yaml

from datagen.self_distill.play_storage import _atomic_json, file_digest, fingerprint


ROOT = Path(__file__).resolve().parents[2]


def root_path(value) -> Path:
    """All relative paths are repository-root-relative, never YAML-relative."""
    return (ROOT / value).resolve()


def load_tasks(path: Path, command: str, start: int | None, end: int | None):
    config = yaml.safe_load(root_path(path).read_text())
    allowed = {"stage", "input_dir", "output_dir", "output_suffix", "tasks",
               "parameters", "start_index", "end_index"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("invalid task config fields")
    if config.get("stage") != command:
        raise ValueError("config stage must match the launcher subcommand")
    input_dir = root_path(config.get("input_dir", "."))
    output_dir = root_path(config["output_dir"])
    entries = config["tasks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("tasks must be a nonempty ordered list")
    tasks = []
    for index, entry in enumerate(entries):
        entry = {"input": entry} if isinstance(entry, str) else entry
        if not isinstance(entry, dict) or set(entry) - {"input", "output"}:
            raise ValueError(f"invalid task entry at index {index}")
        names = entry["input"]
        names = names if isinstance(names, list) else [names]
        if not names or (command == "consolidate" and len(names) != 1):
            raise ValueError("each consolidation task requires exactly one input")
        inputs = [(input_dir / name).resolve() for name in names]
        for source in inputs:
            if any(char in str(source) for char in "*?[]"):
                raise ValueError("task inputs must be explicit filenames, not globs")
        if "output" not in entry and len(inputs) != 1:
            raise ValueError("multi-input tasks need an explicit output name")
        name = entry.get("output", inputs[0].stem + config.get("output_suffix", "_" + command))
        output = (output_dir / name).resolve()
        if any(output == prior["output"] or output.is_relative_to(prior["output"])
               or prior["output"].is_relative_to(output) for prior in tasks):
            raise ValueError(f"duplicate or nested task output: {output}")
        tasks.append({"index": index, "inputs": inputs, "output": output})
    for task in tasks:
        if any(source.is_relative_to(task["output"]) for other in tasks for source in other["inputs"]):
            raise ValueError("task outputs must not contain task inputs")
    start = config.get("start_index", 0) if start is None else start
    end = config.get("end_index") if end is None else end
    end = len(tasks) if end is None else end
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(tasks):
        raise ValueError("index range must satisfy 0 <= start <= end <= task count")
    selected = tasks[start:end]
    for task in selected:
        for source in task["inputs"]:
            if not source.is_file():
                raise FileNotFoundError(source)
    return config, selected


def task_arguments(command: str, task: dict, parameters: dict):
    # Parse through the existing CLI to retain its defaults and validation.
    from datagen.self_distill_stages import parser
    forbidden = {"input", "output", "config", "start_index", "end_index", "command"}
    if set(parameters) & forbidden:
        raise ValueError("task paths and index ranges do not belong in parameters")
    arguments = [command, "--input", *(str(path) for path in task["inputs"]),
                 "--output", str(task["output"])]
    for key, value in parameters.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            arguments.append("--" + (flag if value else "no-" + flag))
        else:
            if key in {"model", "encoder", "stockfish"}:
                value = root_path(value)
            arguments.extend(["--" + flag, str(value)])
    parsed = vars(parser().parse_args(arguments))
    if parsed.get("overwrite"):
        raise ValueError("config tasks do not allow overwrite; use a new output directory")
    if parsed.get("resume") is False:
        raise ValueError("config tasks require resume for interruption recovery")
    resolved = {key: str(value) if isinstance(value, Path) else value
                for key, value in parsed.items()}
    return arguments, resolved


def output_inventory(output: Path) -> dict:
    return {str(path.relative_to(output)): file_digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()}


def semantic_specification(specification: dict) -> dict:
    parameters = dict(specification["parameters"])
    if parameters.get("command") == "mine":
        parameters.setdefault("use_v1_vllm", True)  # Missing in legacy V1 receipts.
    # These do not alter request seeds or the sequence of mining boundaries.
    for key in ("sf_workers", "checkpoint_every"):
        parameters.pop(key, None)
    return {**specification, "parameters": parameters}


def run_config(path: Path, command: str, start: int | None = None, end: int | None = None,
               *, use_v1_vllm: bool | None = None):
    config, tasks = load_tasks(path, command, start, end)
    parameters = dict(config.get("parameters", {}))
    if use_v1_vllm is not None:
        if command != "mine":
            raise ValueError("use_v1_vllm applies only to mining")
        parameters["use_v1_vllm"] = use_v1_vllm
    # Validate every selected command before starting costly work.
    prepared = [(task, *task_arguments(command, task, parameters))
                for task in tasks]
    artifacts = {}
    for _, _, parameters in prepared:
        for key in ("model", "encoder", "stockfish"):
            if key in parameters and parameters[key] not in artifacts:
                artifact = Path(parameters[key])
                if not artifact.exists():
                    raise FileNotFoundError(artifact)
                artifacts[str(artifact)] = fingerprint(artifact)
    for task, arguments, parameters in prepared:
        output = task["output"]
        records = output.parent / ".tasks"
        records.mkdir(parents=True, exist_ok=True)
        manifest_path = records / (output.name + ".json")
        with (records / (output.name + ".lock")).open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"task output is already owned: {output}") from error
            specification = {
                "task_index": task["index"], "parameters": parameters,
                "inputs": {str(source): file_digest(source) for source in task["inputs"]},
                "artifacts": artifacts,
            }
            previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            if previous is not None:
                if semantic_specification(previous["specification"]) != semantic_specification(specification):
                    raise RuntimeError(f"task specification changed: {output}")
                if previous["complete"]:
                    if output_inventory(output) != previous["outputs"]:
                        raise RuntimeError(f"completed task outputs changed: {output}")
                    print(f"[skip task {task['index']}] {output}", flush=True)
                    continue
            elif output.exists():
                raise FileExistsError(f"output exists without a task manifest: {output}")
            manifest = {"config_path": str(root_path(path)), "specification": specification,
                        "complete": False}
            _atomic_json(manifest_path, manifest)
            print(f"[run task {task['index']}] {output}", flush=True)
            # Process isolation releases GPU resources between tasks and preserves
            # the established stage entrypoints; no alternate algorithm is used.
            if command == "rebalance" and (output / "manifest.json").exists():
                # Rebalance publishes its directory atomically. Recover a parent
                # interruption between that publication and our completion receipt.
                published = json.loads((output / "manifest.json").read_text())
                if ({row["path"]: row["sha256"] for row in published["sources"]}
                        != specification["inputs"]
                        or any(file_digest(output / row["path"]) != row["sha256"]
                               for row in published["outputs"])):
                    raise RuntimeError("published rebalance output failed verification")
            else:
                subprocess.run([sys.executable, "-m", "datagen.self_distill_stages", *arguments],
                               cwd=ROOT, check=True, pass_fds=(lock.fileno(),))
            if any(file_digest(source) != specification["inputs"][str(source)] for source in task["inputs"]):
                raise RuntimeError("task input changed while processing")
            outputs = output_inventory(output)
            required = {"mine": "summary.json", "consolidate": "consolidated.jsonl",
                        "rebalance": "manifest.json"}[command]
            if required not in outputs:
                raise RuntimeError(f"stage did not publish its final output: {output}")
            _atomic_json(manifest_path, {**manifest, "complete": True, "outputs": outputs})
