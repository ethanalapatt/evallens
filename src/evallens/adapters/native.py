"""Native adapters over the tiny fixture.

Two adapters, and the difference between them is the whole point:

``ReferenceAdapter``
    Always correct, always full-prefix. Stateless batches run as one padded forward pass;
    cached and session cases run *each request independently, unpadded, at batch size one*,
    so the reference is correct and request-isolated by construction rather than by
    assumption. There is no `Behavior` parameter: it cannot be made faulty.

``CandidateAdapter``
    The optimized path. Stateless batches still run as one padded forward pass (there is no
    cached batched path in scope), while ``cached_decode`` and ``session`` run prefill plus
    per-token incremental decoding against a KV cache. Constructed with ``Behavior()`` it is
    the known-good cached implementation; the mutant corpus supplies faulty variants.

Comparing full-prefix against incremental is the realistic scenario this project targets:
nothing crashes, shapes agree, and the optimized path is quietly wrong.

Every ``run`` begins by resetting adapter state. State *within* a session case is intentional
and serialized; state must never leak between separate cases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from evallens.adapters.encoding import (
    encode_cached_steps,
    encode_stateless_batch,
    encode_unpadded_single,
    validate_case,
)
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.fixtures.tiny_transformer import KVCache, TinyTransformer, build_model
from evallens.types import (
    AdapterExecutionError,
    Case,
    Checkpoint,
    CheckpointAddress,
    CheckpointKind,
    ExecutionMode,
    ExecutionResult,
    InvalidCaseError,
    Request,
    summarize_tensor,
)


@dataclass(frozen=True, slots=True)
class CaptureBudget:
    """Bounds on diagnostic capture.

    Capture is deliberately bounded and deliberately separate from the comparison path.
    Retaining every activation of every generated test would dominate both memory and time,
    and would quietly land in only one of the two timing baselines.
    """

    max_checkpoints: int = 4096
    max_values_per_checkpoint: int = 4096

    @staticmethod
    def disabled() -> CaptureBudget:
        return CaptureBudget(max_checkpoints=0, max_values_per_checkpoint=0)


@dataclass(slots=True)
class _CaptureSink:
    """Collects checkpoints for one adapter call under a fixed budget."""

    budget: CaptureBudget
    checkpoints: list[Checkpoint]
    truncated: bool = False
    # (row, column, request_id, logical_position) entries to extract from each recorded tensor
    mapping: tuple[tuple[int, int, str, int], ...] = ()

    def record(self, layer_name: str, kind: CheckpointKind, tensor: torch.Tensor) -> None:
        if self.truncated:
            return
        values = tensor.detach().numpy()
        for row, column, request_id, position in self.mapping:
            if len(self.checkpoints) >= self.budget.max_checkpoints:
                self.truncated = True
                return
            vector = np.ascontiguousarray(values[row, column], dtype=np.float32)
            retained = vector if vector.size <= self.budget.max_values_per_checkpoint else None
            self.checkpoints.append(
                Checkpoint(
                    address=CheckpointAddress(request_id, layer_name, position, kind),
                    values=retained,
                    summary=summarize_tensor(vector),
                )
            )


class _NativeAdapterBase:
    """Shared machinery: weight ownership, validation, and checkpoint plumbing."""

    role = "native"

    def __init__(
        self,
        config: ModelConfig,
        weights: WeightDict,
        behavior: Behavior | None = None,
        *,
        capture_budget: CaptureBudget | None = None,
        label: str = "",
    ) -> None:
        self.config = config
        self.behavior = behavior or Behavior()
        self.weights_sha256 = weights_sha256(weights)
        self.capture_budget = capture_budget or CaptureBudget()
        self.label = label
        self._model: TinyTransformer = build_model(config, weights, self.behavior)
        self._model_calls = 0

    @property
    def adapter_id(self) -> str:
        suffix = f"/{self.label}" if self.label else ""
        return (
            f"{self.role}/{self.config.config_id}/w{self.weights_sha256[:8]}"
            f"/b{self._behavior_tag()}{suffix}"
        )

    def _behavior_tag(self) -> str:
        if self.behavior.is_reference:
            return "ref"
        parts = [
            f"{k}={v}"
            for k, v in sorted(self.behavior.to_dict().items())
            if v not in ("correct", 0.0)
        ]
        return ",".join(parts)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "role": self.role,
            "model_config_id": self.config.config_id,
            "weights_sha256": self.weights_sha256,
            "behavior": self.behavior.to_dict(),
            "behavior_description": self.behavior.describe(),
            "device": "cpu",
            "dtype": "float32",
        }

    def reset(self) -> None:
        """Return to clean state. Every replay starts here."""
        self._model_calls = 0

    def validate(self, case: Case) -> None:
        validate_case(case, self.config, weights_sha256=self.weights_sha256)

    def _forward(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        valid: np.ndarray,
        cache: KVCache | None,
        sink: _CaptureSink | None,
    ) -> np.ndarray:
        if positions.max(initial=0) >= self.config.max_position:
            raise AdapterExecutionError(
                f"position id {int(positions.max())} exceeds the fixture context limit "
                f"{self.config.max_position}"
            )
        self._model_calls += 1
        logits = self._model(
            torch.from_numpy(np.ascontiguousarray(tokens)),
            torch.from_numpy(np.ascontiguousarray(positions)),
            torch.from_numpy(np.ascontiguousarray(valid)),
            cache,
            sink.record if sink is not None else None,
        )
        values: np.ndarray = logits.detach().numpy().astype(np.float32)
        return values

    def _result(
        self,
        case: Case,
        outputs: dict[str, np.ndarray],
        sink: _CaptureSink | None,
        started_ns: int,
        extra: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        meta: dict[str, Any] = {
            "device": "cpu",
            "dtype": "float32",
            "behavior": self.behavior.to_dict(),
            "model_calls": self._model_calls,
            "capture_truncated": bool(sink.truncated) if sink else False,
            "n_checkpoints": len(sink.checkpoints) if sink else 0,
        }
        if extra:
            meta.update(extra)
        return ExecutionResult(
            adapter_id=self.adapter_id,
            case_id=case.case_id,
            outputs=outputs,
            checkpoints=tuple(sink.checkpoints) if sink else (),
            capture_enabled=sink is not None,
            wall_time_ns=time.perf_counter_ns() - started_ns,
            meta=meta,
        )

    def _new_sink(self, capture: bool) -> _CaptureSink | None:
        if not capture or self.capture_budget.max_checkpoints <= 0:
            return None
        return _CaptureSink(budget=self.capture_budget, checkpoints=[])

    def _run_stateless_batch(self, case: Case, sink: _CaptureSink | None) -> dict[str, np.ndarray]:
        encoding = encode_stateless_batch(case)
        if sink is not None:
            sink.mapping = tuple(
                (layout.row, column, layout.request_id, column - layout.col_start)
                for layout in encoding.layouts
                for column in range(layout.col_start, layout.col_stop)
            )
        logits = self._forward(
            encoding.token_ids, encoding.position_ids, encoding.key_valid, None, sink
        )
        return {
            layout.request_id: np.asarray(logits[layout.row, layout.col_start : layout.col_stop])
            for layout in encoding.layouts
        }


class ReferenceAdapter(_NativeAdapterBase):
    """Correct full-prefix execution. Cannot be given a fault.

    For cached and session cases each request is executed independently, unpadded, at batch
    size one. That makes request isolation a property of the reference's *construction*, so
    a candidate that leaks state between requests has something trustworthy to fail against.
    """

    role = "native-reference"

    def __init__(
        self,
        config: ModelConfig,
        weights: WeightDict,
        *,
        capture_budget: CaptureBudget | None = None,
        label: str = "",
    ) -> None:
        super().__init__(config, weights, Behavior(), capture_budget=capture_budget, label=label)

    def run(self, case: Case, capture: bool = False) -> ExecutionResult:
        self.reset()
        self.validate(case)
        started = time.perf_counter_ns()
        sink = self._new_sink(capture)

        if case.execution_mode is ExecutionMode.STATELESS_BATCH:
            outputs = self._run_stateless_batch(case, sink)
        else:
            outputs = {}
            for request in case.requests:
                encoding = encode_unpadded_single(request)
                if sink is not None:
                    sink.mapping = tuple(
                        (0, position, request.request_id, position)
                        for position in range(request.n_valid)
                    )
                logits = self._forward(
                    encoding.token_ids, encoding.position_ids, encoding.key_valid, None, sink
                )
                outputs[request.request_id] = logits[0]

        return self._result(case, outputs, sink, started)


class CandidateAdapter(_NativeAdapterBase):
    """The optimized path under test.

    With ``Behavior()`` this is the known-good cached implementation and its comparison
    against the reference is a control, not a detection.
    """

    role = "native-candidate"

    def __init__(
        self,
        config: ModelConfig,
        weights: WeightDict,
        behavior: Behavior | None = None,
        *,
        capture_budget: CaptureBudget | None = None,
        label: str = "",
    ) -> None:
        super().__init__(config, weights, behavior, capture_budget=capture_budget, label=label)
        self._cache: KVCache | None = None

    def reset(self) -> None:
        super().reset()
        self._cache = None

    def begin_request(self) -> None:
        """Clear per-request cache state between requests of one session.

        ``Behavior.reset`` decides whether that actually happens. A serving loop that forgets
        this is the classic request-isolation bug, and it is invisible until the second
        request of a session produces subtly wrong tokens.
        """
        if self._cache is None:
            self._cache = self._model.new_cache()
            return
        mode = self.behavior.reset
        if mode == "none":
            return
        if mode == "partial":
            self._cache.clear_layer(0)
            return
        self._cache.clear()

    def _decode_positions(self, positions: np.ndarray, request: Request) -> np.ndarray:
        """Apply the decode-position behavior to one cached step's position IDs."""
        mode = self.behavior.decode_pos
        if mode == "correct":
            return positions
        if mode == "minus_one":
            shifted: np.ndarray = np.maximum(positions - 1, 0)
            return shifted
        if mode == "restart":
            restarted: np.ndarray = np.maximum(positions - request.prefix_length, 0)
            return restarted
        raise AdapterExecutionError(f"unknown decode_pos mode {mode!r}")

    def _run_cached_request(self, request: Request, sink: _CaptureSink | None) -> np.ndarray:
        self.begin_request()
        assert self._cache is not None
        collected: list[np.ndarray] = []

        for index, (tokens, positions, logical) in enumerate(encode_cached_steps(request)):
            step_positions = positions if index == 0 else self._decode_positions(positions, request)
            valid = np.ones_like(tokens, dtype=bool)
            if sink is not None:
                sink.mapping = tuple(
                    (0, column, request.request_id, position)
                    for column, position in enumerate(logical)
                )
            collected.append(self._forward(tokens, step_positions, valid, self._cache, sink)[0])

        stacked: np.ndarray = np.concatenate(collected, axis=0)
        return stacked

    def run(self, case: Case, capture: bool = False) -> ExecutionResult:
        self.reset()
        self.validate(case)
        started = time.perf_counter_ns()
        sink = self._new_sink(capture)

        if case.execution_mode is ExecutionMode.STATELESS_BATCH:
            outputs = self._run_stateless_batch(case, sink)
        else:
            self._cache = self._model.new_cache()
            outputs = {}
            for request in case.requests:
                logits = self._run_cached_request(request, sink)
                if logits.shape[0] != request.n_valid:
                    raise AdapterExecutionError(
                        f"cached execution produced {logits.shape[0]} positions for request "
                        f"{request.request_id!r}, expected {request.n_valid}"
                    )
                outputs[request.request_id] = logits

        return self._result(case, outputs, sink, started)


def build_adapter_pair(
    config: ModelConfig,
    weights: WeightDict,
    behavior: Behavior | None = None,
    *,
    capture_budget: CaptureBudget | None = None,
    label: str = "",
) -> tuple[ReferenceAdapter, CandidateAdapter]:
    """Build a reference/candidate pair over the same config and the same canonical weights."""
    return (
        ReferenceAdapter(config, weights, capture_budget=capture_budget),
        CandidateAdapter(config, weights, behavior, capture_budget=capture_budget, label=label),
    )


__all__ = [
    "AdapterExecutionError",
    "CandidateAdapter",
    "CaptureBudget",
    "InvalidCaseError",
    "ReferenceAdapter",
    "build_adapter_pair",
]
