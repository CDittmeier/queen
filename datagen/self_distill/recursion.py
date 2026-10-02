"""Batched FIFO recursion for self-distillation supervision mining."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess

from datagen.self_distill.analysis import (
    ANALYSIS_PROMPT,
    comparison_boundary,
    expected_winrate,
    finish_critical,
    first_line_divergence,
    legal_root_candidates,
    legal_structured_best,
    parse_critical,
    parse_model_evaluation,
    parsed_sequence,
    terminal_analysis,
    terminal_attempt,
    terminal_evaluation,
    truncate_critical_line,
)
from datagen.self_distill.io import atomic_json, truncate_file, stop_after_batch
from datagen.self_distill.play_storage import fingerprint
from datagen.self_distill.oracle import ParallelOracle


@dataclass(frozen=True)
class MiningConfig:
    output: Path
    model: Path
    encoder: Path
    stockfish: Path
    input_identity: str
    seed: int = 20260815
    sf_workers: int = 8
    sf_nodes: int = 100_000
    max_recursions: int = 5
    max_root_retries: int = 3
    max_child_retries: int = 0
    min_clean_ply: int = 1
    work_batch_size: int = 1024
    checkpoint_every: int = 4096
    max_output_tokens: int = 4096
    max_num_seqs: int = 128
    use_v1_vllm: bool = False
    gpu_memory_utilization: float = 0.70
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    inject_drop: float = 0.05
    mistake_drop: float = 0.10
    eval_truth_limit: float = 0.10
    resume: bool = True

    def validate(self) -> None:
        if self.sf_workers <= 0 or self.sf_nodes <= 0:
            raise ValueError("Stockfish workers and nodes must be positive")
        if self.max_recursions < 0 or self.max_root_retries < 0 or self.max_child_retries < 0:
            raise ValueError("retry and recursion limits must be nonnegative")
        if self.min_clean_ply <= 0:
            raise ValueError("min_clean_ply must be positive")
        if self.work_batch_size <= 0 or self.checkpoint_every <= 0:
            raise ValueError("batch and checkpoint sizes must be positive")
        if self.max_output_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("generation limits must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        for path in (self.model, self.encoder, self.stockfish):
            if not path.exists():
                raise FileNotFoundError(path)

    def identity(self) -> str:
        values = asdict(self)
        if self.use_v1_vllm:
            values.pop("use_v1_vllm")  # Preserve legacy V1 checkpoint identities.
        values.pop("output")
        values.pop("resume")
        values.pop("checkpoint_every")
        values.pop("sf_workers")
        for name in ("model", "encoder", "stockfish"):
            values[name] = fingerprint(values[name])
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class Timings:
    seconds: dict[str, float]
    counts: Counter

    @classmethod
    def create(cls) -> "Timings":
        return cls(defaultdict(float), Counter())

    def add(self, name: str, seconds: float, count: int = 0) -> None:
        self.seconds[name] += seconds
        self.counts[name] += count

    def as_dict(self) -> dict:
        return {"seconds": dict(self.seconds), "counts": dict(self.counts)}


class RecursiveMiner:
    def __init__(self, config: MiningConfig):
        config.validate()
        self.config = config
        self.output = config.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.timings = Timings.create()
        self.generator = None
        self.oracle = ParallelOracle(
            config.stockfish,
            config.sf_workers,
            config.sf_nodes,
        )

    def _load_generator(self) -> None:
        if self.generator is not None:
            return
        from models.vllm.flamingo_generate import ChessFlamingoGenerator

        # model setup: eager execution and disabled prefix caching preserve FEN conditioning.
        started = time.perf_counter()
        self.generator = ChessFlamingoGenerator(
            self.config.model,
            self.config.encoder,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_output_tokens + 256,
            max_num_seqs=self.config.max_num_seqs,
            seed=self.config.seed,
            enforce_eager=True,
            use_v1_vllm=self.config.use_v1_vllm,
        )
        self.timings.add("model_load", time.perf_counter() - started)

    def _stable_seed(self, row: dict, wave: int) -> int:
        identity = "|".join(
            (str(self.config.seed), row["fen"], str(row.get("depth", 0)), str(wave))
        )
        digest = hashlib.blake2b(identity.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (2**31 - 1)

    def _generate(self, rows: list[dict], wave: int) -> list[dict]:
        if not rows:
            return []
        self._load_generator()
        started = time.perf_counter()
        outputs = []
        # generation: bound encoder activations by matching vLLM's live-sequence capacity.
        for offset in range(0, len(rows), self.config.max_num_seqs):
            chunk = rows[offset : offset + self.config.max_num_seqs]
            outputs.extend(
                self.generator.generate(
                    [row["fen"] for row in chunk],
                    [ANALYSIS_PROMPT] * len(chunk),
                    [row.get("history") or [] for row in chunk],
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_output_tokens,
                    seeds=[self._stable_seed(row, wave) for row in chunk],
                )
            )
        self.timings.add("model_generation", time.perf_counter() - started, len(rows))
        self.timings.counts["generated_tokens"] += sum(
            len(output["token_ids"]) for output in outputs
        )
        self.timings.counts["truncated_generations"] += sum(
            output["finish_reason"] == "length" for output in outputs
        )
        return outputs

    def _generate_roots(self, active: list[dict], batch_number: int) -> tuple[list[dict], list[dict]]:
        pending = [dict(row, root_attempts=[]) for row in active]
        ready = []
        uncached = []

        # cached roots: reuse the child analysis that led to a recursed position.
        for item in pending:
            cached = item.pop("cached_root", None)
            if cached is None:
                uncached.append(item)
                continue
            text = cached["generation"]
            evaluation, error = parse_model_evaluation(text, chess.Board(item["fen"]).turn)
            if evaluation is None:
                item["root_attempts"].append(
                    {"attempt": 0, "error": error, "generation": text, "reused_child": True}
                )
                uncached.append(item)
                continue
            item.update(
                root_generation=text,
                root_evaluation=evaluation,
                root_candidates=legal_root_candidates(item["fen"], text),
                initial_move=legal_structured_best(item["fen"], text),
                root_generation_attempt=0,
                root_token_count=cached.get("token_count", 0),
                root_finish_reason=cached.get("finish_reason", "reused_child"),
                root_reused_from_child=True,
            )
            self.timings.counts["reused_root_analyses"] += 1
            ready.append(item)

        # root retries: regenerate only outputs whose numerical evaluation is malformed.
        pending = uncached
        for retry in range(self.config.max_root_retries + 1):
            if not pending:
                break
            outputs = self._generate(pending, 100 + batch_number * 10 + retry)
            next_pending = []
            for item, output in zip(pending, outputs):
                text = output["text"]
                evaluation, error = parse_model_evaluation(text, chess.Board(item["fen"]).turn)
                if evaluation is None:
                    item["root_attempts"].append(
                        {"attempt": retry + 1, "error": error, "generation": text}
                    )
                    next_pending.append(item)
                    continue
                item.update(
                    root_generation=text,
                    root_evaluation=evaluation,
                    root_candidates=legal_root_candidates(item["fen"], text),
                    initial_move=legal_structured_best(item["fen"], text),
                    root_generation_attempt=retry + 1,
                    root_token_count=len(output["token_ids"]),
                    root_finish_reason=output["finish_reason"],
                )
                ready.append(item)
            pending = next_pending
        return ready, pending

    def _prepare_candidates(self, items: list[dict]) -> None:
        if not items:
            return
        started = time.perf_counter()
        requests = []
        for item in items:
            board = chess.Board(item["fen"])
            item["required_candidates"] = min(3, board.legal_moves.count())
            requests.append((item["fen"], item["required_candidates"]))
        suggestions = self.oracle.suggestions(requests)
        comparisons = []

        # candidate repair: fill illegal or missing model candidates from Stockfish MultiPV.
        for item in items:
            rows = suggestions[(item["fen"], item["required_candidates"])]
            best = chess.Move.from_uci(rows[0]["move"])
            repaired = list(item["root_candidates"])
            repair_moves = []
            for row in rows:
                if len(repaired) >= item["required_candidates"]:
                    break
                move = chess.Move.from_uci(row["move"])
                if move not in repaired:
                    repaired.append(move)
                    repair_moves.append(move)
            if len(repaired) < item["required_candidates"]:
                raise RuntimeError("Stockfish returned too few root repair moves")
            item["repaired_candidates"] = repaired[: item["required_candidates"]]
            item["repair_moves"] = repair_moves
            item["sf_best"] = best
            compared = list(item["repaired_candidates"])
            if best not in compared:
                compared.append(best)
            item["comparison_moves"] = tuple(move.uci() for move in compared)
            comparisons.append((item["fen"], item["comparison_moves"]))

        # candidate injection: replace the weakest model move when all model moves miss badly.
        scores_by_item = self.oracle.comparisons(comparisons)
        for item in items:
            scores = scores_by_item[(item["fen"], item["comparison_moves"])]
            best = item["sf_best"]
            best_wr = expected_winrate(scores[best.uci()]["cp"])
            predicted = [
                {
                    "move": move,
                    "cp": scores[move.uci()]["cp"],
                    "winrate": expected_winrate(scores[move.uci()]["cp"]),
                }
                for move in item["repaired_candidates"]
            ]
            predicted.sort(key=lambda row: row["winrate"], reverse=True)
            
            # if all moves drop >= 5% (configurable) WR, then inject SF move replacing the worst model move.
            # Keys 'x', 'y', 'z' are used to label the candidates
            inject = (
                best not in item["repaired_candidates"]
                and best_wr - predicted[0]["winrate"] >= self.config.inject_drop
            )
            chosen = (
                [{"move": best, "cp": scores[best.uci()]["cp"], "winrate": best_wr}, *predicted[:2]]
                if inject
                else predicted[:3]
            )
            chosen.sort(key=lambda row: row["winrate"], reverse=True)
            for label, row in zip("xyz", chosen):
                row["label"] = label
            initial = item["initial_move"]
            item.update(
                final_candidates=chosen,
                injected_stockfish=inject,
                true_root_cp=scores[best.uci()]["cp"],
                true_root_winrate=best_wr,
                initial_move_winrate=(
                    expected_winrate(scores[initial.uci()]["cp"])
                    if initial is not None
                    else 0.0
                ),
            )
        self.timings.add("root_oracle", time.perf_counter() - started, len(items))

    def _audit_outputs(self, branches: list[dict], outputs: list[dict]) -> None:
        # The audit basically adds bookkeeping info like fens to the parsed critical line, truncates the critical line at the first mistake
        # before consolidation, etc.
        started = time.perf_counter()
        prepared = []
        requests = []
        for branch, output in zip(branches, outputs):
            text = output["text"]
            evaluation, error = parse_model_evaluation(text, chess.Board(branch["fen"]).turn)
            parsed = parse_critical(branch["fen"], text)
            requests.extend((step["fen"], step["uci"]) for step in parsed["legal_steps"])
            prepared.append(
                (branch, text, evaluation, error, parsed, len(output["token_ids"]), output["finish_reason"])
            )
        oracle_started = time.perf_counter()
        audits = self.oracle.move_audits(requests) if requests else {}
        oracle_seconds = time.perf_counter() - oracle_started
        self.timings.add("child_oracle", oracle_seconds, len(requests))
        for branch, text, evaluation, error, parsed, tokens, finish_reason in prepared:
            sanitation = finish_critical(
                parsed,
                audits,
                self.config.mistake_drop,
                self.config.min_clean_ply,
            )
            branch["attempts"].append(
                {
                    "attempt": len(branch["attempts"]) + 1,
                    "generation": text,
                    "token_count": tokens,
                    "finish_reason": finish_reason,
                    "evaluation": evaluation,
                    "evaluation_error": error,
                    "sanitation": sanitation,
                }
            )
        self.timings.add("child_parsing", time.perf_counter() - started - oracle_seconds, len(branches))

    @staticmethod
    def _usable(branch: dict) -> list[dict]:
        return [
            attempt
            for attempt in branch["attempts"]
            if attempt["evaluation"] is not None
            and attempt["sanitation"]["valid_checked_prefix"]
        ]

    def _sample_children(self, items: list[dict], batch_number: int) -> None:
        started = time.perf_counter()
        branches = []
        by_owner = defaultdict(list)
        for item in items:
            for candidate in item["final_candidates"]:
                board = chess.Board(item["fen"])
                board.push(candidate["move"])
                branch = {
                    "owner": item["lineage_id"],
                    "label": candidate["label"],
                    "candidate": candidate,
                    "fen": board.fen(),
                    "depth": item["depth"],
                    "history": item.get("history", []) + [item["fen"]],
                    "attempts": [],
                }
                if board.is_game_over(claim_draw=False):
                    branch["attempts"].append(terminal_attempt(board))
                branches.append(branch)
                by_owner[item["lineage_id"]].append(branch)

        # child generation: overlap Stockfish auditing of one chunk with generation of the next.
        pending = [branch for branch in branches if not branch["attempts"]]
        retry = 0
        while pending:
            futures = []
            with ThreadPoolExecutor(max_workers=1) as audit_executor:
                for offset in range(0, len(pending), self.config.max_num_seqs):
                    chunk = pending[offset : offset + self.config.max_num_seqs]
                    outputs = self._generate(chunk, 200 + batch_number * 10 + retry)
                    futures.append(audit_executor.submit(self._audit_outputs, chunk, outputs))
                for future in futures:
                    future.result()
            retry += 1
            pending = [
                branch
                for branch in pending
                if not self._usable(branch)
                and len(branch["attempts"]) < 1 + self.config.max_child_retries
            ]

        # child records: expose the first usable attempt and retain every failed attempt for audit.
        for item in items:
            rows = by_owner[item["lineage_id"]]
            item["child_sampling_valid"] = all(self._usable(branch) for branch in rows)
            children = []
            for branch in rows:
                usable = self._usable(branch)
                chosen = usable[0] if usable else None
                candidate = branch["candidate"]
                child = {
                    "label": branch["label"],
                    "move_uci": candidate["move"].uci(),
                    "fen": branch["fen"],
                    "history": branch["history"],
                    "true_root_cp": candidate["cp"],
                    "true_root_winrate": candidate["winrate"],
                    "attempts": branch["attempts"],
                    "selected_attempt": chosen["attempt"] if chosen else None,
                    "valid": chosen is not None,
                }
                evaluation_attempt = chosen or next(
                    (
                        attempt
                        for attempt in branch["attempts"]
                        if attempt["evaluation"] is not None
                    ),
                    None,
                )
                if evaluation_attempt:
                    evaluation = evaluation_attempt["evaluation"]
                    child.update(
                        evaluation_attempt=evaluation_attempt["attempt"],
                        evaluation=evaluation,
                        predicted_root_winrate=1.0 - evaluation["winrate"],
                        predicted_root_order=[-value for value in evaluation["order"]],
                    )
                if chosen:
                    retained = chosen["sanitation"]["accepted_ply"]
                    child.update(
                        generation=truncate_critical_line(chosen["generation"], retained),
                        original_generation=chosen["generation"],
                        sanitation=chosen["sanitation"],
                    )
                children.append(child)
            item["children"] = children
        self.timings.add("child_generation_audit_wall", time.perf_counter() - started, len(branches))

    def _prepare_divergences(self, items: list[dict]) -> None:
        # peek ahead at what the inferred critical line will be as per the children, then compare it to the root critical line
        # if better at first divergence => accept
        # if worse at first divergence => reject (and stop)
        # if root is a subsequence of the inferred critical line => accept as long as next move is not illegal or a mistake
        # if root is a supersequence => reject (and stop)
        started = time.perf_counter()
        records = []
        requests = []
        for item in items:
            children = sorted(item["children"], key=lambda row: "xyz".index(row["label"]))
            if any("predicted_root_order" not in child for child in children):
                continue
            if not item["child_sampling_valid"]:
                continue
            selected = max(children, key=lambda row: tuple(row["predicted_root_order"]))
            old = parsed_sequence(parse_critical(item["fen"], item["root_generation"]))
            new = [{"uci": selected["move_uci"], "legal": True}]
            new.extend(
                {"uci": move, "legal": True}
                for move in selected["sanitation"]["accepted_uci"]
            )
            divergence = first_line_divergence(item["fen"], old, new)
            boundary = comparison_boundary(selected["sanitation"])

            # hidden boundary: compare a truncated bad move instead of treating it as a clean prefix.
            if (
                divergence["kind"] == "no_divergence"
                and len(new) < len(old)
                and boundary is not None
            ):
                divergence = first_line_divergence(item["fen"], old, [*new, boundary])
                divergence["used_truncation_boundary"] = True
            item.update(
                original_critical_sequence=old,
                inferred_critical_sequence=new,
                inferred_truncation_boundary=boundary,
                first_divergence=divergence,
            )
            records.append((item, divergence))
            if divergence["kind"] == "divergence":
                old_move = divergence["old"]
                new_move = divergence["new"]
                if old_move["legal"] and new_move["legal"] and old_move["uci"] != new_move["uci"]:
                    requests.append(
                        (divergence["fen"], tuple(dict.fromkeys((old_move["uci"], new_move["uci"]))))
                    )

        # divergence oracle: score both differing moves in one restricted root search.
        oracle_started = time.perf_counter()
        comparisons = self.oracle.comparisons(requests) if requests else {}
        oracle_seconds = time.perf_counter() - oracle_started
        self.timings.add("divergence_oracle", oracle_seconds, len(requests))
        for item, divergence in records:
            kind = divergence["kind"]
            if kind == "divergence":
                old_move = divergence["old"]
                new_move = divergence["new"]
                if old_move["legal"] and not new_move["legal"]:
                    accepted = False
                    reason = "new line becomes illegal at first divergence"
                elif not old_move["legal"] and new_move["legal"]:
                    accepted = True
                    reason = "new line repairs illegal first divergence"
                elif not old_move["legal"] and not new_move["legal"]:
                    accepted = False
                    reason = "both lines are illegal at first divergence"
                else:
                    moves = tuple(dict.fromkeys((old_move["uci"], new_move["uci"])))
                    rows = comparisons[(divergence["fen"], moves)]
                    old_cp = rows[old_move["uci"]]["cp"]
                    new_cp = rows[new_move["uci"]]["cp"]
                    divergence.update(old_cp=old_cp, new_cp=new_cp)
                    accepted = new_cp > old_cp
                    reason = (
                        "new move is better at first divergence"
                        if accepted
                        else "new move is not better at first divergence"
                    )
            elif kind == "identical_illegal":
                accepted = False
                reason = "same illegal line"
            else:
                old_error = abs(
                    item["root_evaluation"]["winrate"] - item["true_root_winrate"]
                )
                selected = max(
                    item["children"],
                    key=lambda row: tuple(row.get("predicted_root_order", (-math.inf,))),
                )
                new_error = abs(
                    selected["predicted_root_winrate"] - item["true_root_winrate"]
                )
                divergence.update(old_eval_error=old_error, new_eval_error=new_error)
                accepted = new_error < old_error
                reason = (
                    "prefix-equivalent line has a closer evaluation"
                    if accepted
                    else "prefix-equivalent line does not improve evaluation"
                )
            divergence.update(accept=accepted, reason=reason)
        self.timings.add("divergence_parsing", time.perf_counter() - started - oracle_seconds, len(records))

    def _decide(self, item: dict) -> dict:
        children = sorted(item["children"], key=lambda row: "xyz".index(row["label"]))
        if any("predicted_root_winrate" not in child for child in children):
            offending = next(
                child for child in children if "predicted_root_winrate" not in child
            )
            return {
                "action": "recurse",
                "reason": "child evaluation remained malformed",
                "recurse_fen": offending["fen"],
                "recurse_history": offending["history"],
                "offending_label": offending["label"],
                "offending_move_uci": offending["move_uci"],
            }

        # inference: choose the child with the strongest evaluation from the root POV.
        selected = max(children, key=lambda row: tuple(row["predicted_root_order"]))
        best = children[0]
        inferred = selected["predicted_root_winrate"]
        details = {
            "selected_label": selected["label"],
            "selected_move_uci": selected["move_uci"],
            "e_prime": inferred,
            "new_eval_error": abs(inferred - item["true_root_winrate"]),
            "old_eval_error": abs(
                item["root_evaluation"]["winrate"] - item["true_root_winrate"]
            ),
        }

        # mandatory recursion: fix a bad child ordering or a root evaluation outside tolerance.
        ordering_bad = (
            selected["label"] != "x"
            and best["true_root_winrate"] - selected["true_root_winrate"]
            >= self.config.inject_drop
        )
        if ordering_bad:
            offending = max(
                children,
                key=lambda row: abs(
                    row["predicted_root_winrate"] - row["true_root_winrate"]
                ),
            )
            return {
                "action": "recurse",
                "reason": "incorrect inaccurate child ordering",
                "recurse_fen": offending["fen"],
                "recurse_history": offending["history"],
                **details,
            }
        if details["new_eval_error"] >= self.config.eval_truth_limit:
            return {
                "action": "recurse",
                "reason": "inferred evaluation outside truth limit",
                "recurse_fen": best["fen"],
                "recurse_history": best["history"],
                **details,
            }
        if not item["child_sampling_valid"]:
            offending = next(child for child in children if not child["valid"])
            return {
                "action": "recurse",
                "reason": "child exhausted clean-prefix attempts",
                "recurse_fen": offending["fen"],
                "recurse_history": offending["history"],
                "offending_label": offending["label"],
                "offending_move_uci": offending["move_uci"],
                **details,
            }

        # acceptance: the proposed line must win at its first meaningful divergence.
        divergence = item["first_divergence"]
        if divergence["accept"]:
            return {"action": "accept", "reason": divergence["reason"], **details}
        return {
            "action": "recurse",
            "reason": divergence["reason"],
            "recurse_fen": selected["fen"],
            "recurse_history": selected["history"],
            **details,
        }

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, chess.Move):
            return value.uci()
        if isinstance(value, dict):
            return {
                key: RecursiveMiner._clean(item)
                for key, item in value.items()
                if key != "candidate"
            }
        if isinstance(value, (list, tuple)):
            return [RecursiveMiner._clean(item) for item in value]
        return value

    @staticmethod
    def _cached_child_root(item: dict, decision: dict) -> dict | None:
        child = next(
            (
                row
                for row in item.get("children", [])
                if row["fen"] == decision.get("recurse_fen")
            ),
            None,
        )
        if child is None or "evaluation" not in child:
            return None
        attempt_number = child.get("evaluation_attempt")
        attempt = next(
            (
                row
                for row in child.get("attempts", [])
                if row["attempt"] == attempt_number
            ),
            None,
        )
        if attempt is None:
            return None
        return {
            "generation": attempt["generation"],
            "token_count": attempt.get("token_count", 0),
            "finish_reason": attempt.get("finish_reason", "reused_child"),
            "from_fen": item["fen"],
            "from_move_uci": child["move_uci"],
        }

    def _recurse_item(self, item: dict, decision: dict) -> dict:
        return {
            "lineage_id": item["lineage_id"],
            "source": item["source"],
            "start_fen": item["start_fen"],
            "fen": decision["recurse_fen"],
            "history": decision.get("recurse_history", item.get("history", [])),
            "depth": item["depth"] + 1,
            "trajectory": item["trajectory"],
            "cached_root": self._cached_child_root(item, decision),
        }

    def _finalize(self, item: dict, outcome: str, reason: str) -> dict:
        final_step = item["trajectory"][-1]
        return self._clean(
            {
                "lineage_id": item["lineage_id"],
                "source": item["source"],
                "start_fen": item["start_fen"],
                "source_kind": item["source"].get("extra", {}).get(
                    "source_kind", "unknown"
                ),
                "final_fen": final_step["fen"],
                "final_depth": final_step["depth"],
                "final_true_root_cp": final_step.get("true_root_cp"),
                "final_true_root_winrate": final_step.get("true_root_winrate"),
                "outcome": outcome,
                "outcome_reason": reason,
                "trajectory": item["trajectory"],
            }
        )

    @staticmethod
    def _update_stats(stats: dict, record: dict) -> None:
        outcome = record["outcome"]
        reason = record["outcome_reason"]
        kind = record["source_kind"]
        stats["outcomes"][outcome] = stats["outcomes"].get(outcome, 0) + 1
        stats["outcome_reasons"][reason] = stats["outcome_reasons"].get(reason, 0) + 1
        by_source = stats["outcomes_by_source"].setdefault(kind, {})
        by_source[outcome] = by_source.get(outcome, 0) + 1
        if outcome == "accept":
            depth = str(record["final_depth"])
            depths = stats["accepted_recursion_depth_by_source"].setdefault(kind, {})
            depths[depth] = depths.get(depth, 0) + 1

    @staticmethod
    def _new_stats() -> dict:
        return {
            "processed_boundaries": 0,
            "stockfish_injections": 0,
            "stockfish_candidate_repairs": 0,
            "outcomes": {},
            "outcome_reasons": {},
            "outcomes_by_source": {},
            "accepted_recursion_depth_by_source": {},
        }

    def run(self, seeds: list[dict]) -> dict:
        with stop_after_batch() as stop:
            return self._run(seeds, stop)

    def _run(self, seeds: list[dict], stop: list[bool]) -> dict:
        state_path = self.output / "state.json"
        accepted_path = self.output / "accepted.jsonl"
        discarded_path = self.output / "discarded.jsonl"
        config_identity = self.config.identity()

        # resume: validate input and policy identities, then truncate outputs to committed bytes.
        if state_path.exists() and self.config.resume:
            state = json.loads(state_path.read_text())
            if state.get("schema_version") != 4:
                raise RuntimeError("checkpoint schema does not match this miner")
            if state["input_identity"] != self.config.input_identity:
                raise RuntimeError("checkpoint inputs do not match this run")
            if state["config_identity"] != config_identity:
                raise RuntimeError("checkpoint configuration does not match this run")
            queue = state["work_queue"]
            stats = state["stats"]
            initial_positions = state["initial_positions"]
            committed = state["committed_bytes"]
            truncate_file(accepted_path, committed["accepted"])
            truncate_file(discarded_path, committed["discarded"])
            prior_seconds = state.get("pipeline_seconds", 0.0)
            batch_number = state["batch_number"]
            for name, value in state.get("timings", {}).get("seconds", {}).items():
                self.timings.seconds[name] += value
            self.timings.counts.update(state.get("timings", {}).get("counts", {}))
        else:
            if state_path.exists() or accepted_path.exists() or discarded_path.exists():
                raise FileExistsError(f"refusing to overwrite an existing mining run in {self.output}")
            queue = [
                {
                    "lineage_id": f"sd_{seed['position_id']}",
                    "source": seed,
                    "start_fen": seed["fen"],
                    "fen": seed["fen"],
                    "history": seed.get("history") or [],
                    "depth": 0,
                    "trajectory": [],
                }
                for seed in seeds
            ]
            initial_positions = len(queue)
            stats = self._new_stats()
            accepted_path.write_text("")
            discarded_path.write_text("")
            committed = {"accepted": 0, "discarded": 0}
            prior_seconds = 0.0
            batch_number = 0
        if not queue and not initial_positions:
            raise ValueError("input contains no positions")

        run_started = time.perf_counter()
        since_checkpoint = 0
        buffers = {"accept": [], "discard": []}

        def checkpoint(force: bool = False) -> None:
            nonlocal since_checkpoint
            if not force and since_checkpoint < self.config.checkpoint_every:
                return
            for outcome, path in (("accept", accepted_path), ("discard", discarded_path)):
                if buffers[outcome]:
                    with path.open("a") as handle:
                        for record in buffers[outcome]:
                            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    buffers[outcome].clear()
                committed[outcome + "ed"] = path.stat().st_size
            atomic_json(
                state_path,
                {
                    "schema_version": 4,
                    "batch_number": batch_number,
                    "input_identity": self.config.input_identity,
                    "config_identity": config_identity,
                    "initial_positions": initial_positions,
                    "work_queue": queue,
                    "committed_bytes": committed,
                    "stats": stats,
                    "pipeline_seconds": prior_seconds + time.perf_counter() - run_started,
                    "timings": self.timings.as_dict(),
                },
                compact=True,
            )
            since_checkpoint = 0

        checkpoint(force=True)
        while queue and not stop[0]:
            batch_number += 1
            active = queue[: self.config.work_batch_size]
            del queue[: len(active)]
            since_checkpoint += len(active)
            stats["processed_boundaries"] += len(active)

            # terminal roots: accept deterministic game outcomes without calling the model.
            terminal = [
                item
                for item in active
                if chess.Board(item["fen"]).is_game_over(claim_draw=False)
            ]
            for item in terminal:
                board = chess.Board(item["fen"])
                reason = "terminal position resolved without generation"
                item["trajectory"].append(
                    {
                        "depth": item["depth"],
                        "fen": item["fen"],
                        "root_generation": terminal_analysis(board),
                        "root_evaluation": terminal_evaluation(board),
                        "terminal": True,
                        "decision": {"action": "accept", "reason": reason},
                    }
                )
                record = self._finalize(item, "accept", reason)
                self._update_stats(stats, record)
                buffers["accept"].append(record)
            terminal_ids = {item["lineage_id"] for item in terminal}
            active = [item for item in active if item["lineage_id"] not in terminal_ids]
            if not active:
                checkpoint()
                continue

            print(
                f"[batch {batch_number}] active={len(active)} "
                f"depths={dict(Counter(item['depth'] for item in active))} queued={len(queue)}",
                flush=True,
            )
            ready, malformed = self._generate_roots(active, batch_number)

            # malformed roots: append the same position to the FIFO until its recursion limit.
            for item in malformed:
                reason = "root evaluation remained malformed"
                decision = {
                    "action": "recurse",
                    "reason": reason,
                    "recurse_fen": item["fen"],
                    "recurse_history": item.get("history", []),
                }
                item["trajectory"].append(
                    {
                        "depth": item["depth"],
                        "fen": item["fen"],
                        "root_attempts": item.get("root_attempts", []),
                        "decision": decision,
                    }
                )
                if item["depth"] < self.config.max_recursions:
                    queue.append(self._recurse_item(item, decision))
                else:
                    record = self._finalize(item, "discard", "recursion limit exceeded")
                    self._update_stats(stats, record)
                    buffers["discard"].append(record)

            # recursive boundary: repair candidates, sample children, compare, then accept or requeue.
            self._prepare_candidates(ready)
            stats["stockfish_injections"] += sum(item["injected_stockfish"] for item in ready)
            stats["stockfish_candidate_repairs"] += sum(len(item["repair_moves"]) for item in ready)
            self._sample_children(ready, batch_number)
            self._prepare_divergences(ready)
            for item in ready:
                decision = self._decide(item)
                step = self._clean(
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "lineage_id",
                            "source",
                            "start_fen",
                            "trajectory",
                            "depth",
                            "cached_root",
                        }
                    }
                )
                step["depth"] = item["depth"]
                step["decision"] = decision
                item["trajectory"].append(step)
                if decision["action"] == "recurse" and item["depth"] < self.config.max_recursions:
                    queue.append(self._recurse_item(item, decision))
                    continue
                if decision["action"] == "recurse":
                    outcome = "discard"
                    reason = "recursion limit exceeded"
                else:
                    outcome = decision["action"]
                    reason = decision["reason"]
                record = self._finalize(item, outcome, reason)
                self._update_stats(stats, record)
                buffers[outcome].append(record)

            checkpoint()
            print(f"[batch {batch_number}] outcomes={stats['outcomes']} queued={len(queue)}", flush=True)

        checkpoint(force=True)
        if queue:
            raise InterruptedError("mining checkpointed after the current batch; resume to continue")
        pipeline_seconds = prior_seconds + time.perf_counter() - run_started
        accepted = stats["outcomes"].get("accept", 0)
        generations = self.timings.counts["model_generation"]
        generation_seconds = self.timings.seconds.get("model_generation", 0.0)
        summary = {
            "schema_version": 1,
            "positions": initial_positions,
            "outcomes": stats["outcomes"],
            "outcome_reasons": stats["outcome_reasons"],
            "outcomes_by_source": stats["outcomes_by_source"],
            "accepted_recursion_depth_by_source": stats[
                "accepted_recursion_depth_by_source"
            ],
            "survival_rate": accepted / initial_positions,
            "pipeline_seconds": pipeline_seconds,
            "positions_per_second": initial_positions / pipeline_seconds,
            "boundaries": stats["processed_boundaries"],
            "boundaries_per_second": stats["processed_boundaries"] / pipeline_seconds,
            "stockfish_injections": stats["stockfish_injections"],
            "stockfish_candidate_repairs": stats["stockfish_candidate_repairs"],
            "generation": {
                "responses": generations,
                "tokens": self.timings.counts["generated_tokens"],
                "tokens_per_second": (
                    self.timings.counts["generated_tokens"] / generation_seconds
                    if generation_seconds
                    else 0.0
                ),
                "truncated": self.timings.counts["truncated_generations"],
                "truncation_rate": (
                    self.timings.counts["truncated_generations"] / generations
                    if generations
                    else 0.0
                ),
            },
            "timings": self.timings.as_dict(),
            "stockfish": {
                "workers": self.oracle.workers,
                "queries": self.oracle.query_count,
                "node_budget": self.oracle.node_budget,
                "batch_wall_seconds": self.oracle.wall_seconds,
            },
            "input_identity": self.config.input_identity,
            "config_identity": config_identity,
        }
        atomic_json(self.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary

    def close(self) -> None:
        self.oracle.close()
