"""Core contracts for EvalLens: cases, execution results, checkpoints, and verdicts.

This module is the resolved schema referenced by SPEC.md. Every other module depends on
it; it depends on nothing inside EvalLens, and it must not import torch.

Axis conventions
----------------
All tensors that cross the adapter boundary are ``numpy`` arrays, never torch tensors, so
that comparison, hashing, and serialization never depend on a framework version.

``ExecutionResult.outputs[request_id]``
    float32, shape ``[n_valid_tokens, vocab_size]``. Row ``p`` holds the next-token logits
    produced *after consuming* logical token ``p`` of that request. Padding columns are
    never present here: the adapter is responsible for stripping them, so a comparison can
    never accidentally treat a padded position as meaningful.

``Checkpoint.values``
    float32, shape ``[hidden_size]`` for per-token activations, or ``[vocab_size]`` for
    ``logits`` checkpoints. One checkpoint is one (request, layer, logical position, kind)
    address, so no checkpoint tensor carries a batch or time axis.

Position-ID convention
----------------------
A request's logical token positions are ``0 .. n_valid - 1`` and are assigned by valid-token
index, never by column index in a padded batch tensor. Left padding therefore shifts a
token's *column* but not its *position ID*. This is what makes padding invariance a testable
property rather than an accident of implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = 1

PAD_TOKEN_ID = 0
"""Reserved vocabulary slot used as the padding token by every native fixture.

Valid generated tokens are drawn from ``1 .. vocab_size - 1`` so that a padding column is
always distinguishable from a real token during validation.
"""

CANONICAL_TOKEN_ID = 1
"""The single token value the reducer simplifies toward.

Token-value simplification must converge, so the canonical set is one fixed element rather
than "some simpler value". Because ``CaseSize`` ranks by distinct-value count and then by
value sum, and this is the smallest legal non-padding id, rewriting any token to it can only
move a case down the size order — never up, and never into a cycle.
"""


class Verdict(StrEnum):
    """Outcome of one reference/candidate comparison.

    The values are deliberately disjoint categories. A crash is never silently counted as a
    detected regression, and an unstable classification is never promoted to a failure.
    """

    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    UNSTABLE = "unstable"


class ExecutionMode(StrEnum):
    """How the requests of a case are executed.

    STATELESS_BATCH
        Every request is one row of a single padded batch, evaluated with one full-prefix
        forward pass. ``Request.prefix_length`` must equal the request length.
    CACHED_DECODE
        Exactly one request, batch size one. The first ``prefix_length`` tokens are
        prefilled in one call; each remaining token is consumed by a separate cached step.
    SESSION
        Between one and ``max_session_requests`` requests executed sequentially on the same
        adapter instance, each like CACHED_DECODE. Per-request cache state is cleared
        between requests; whether an implementation actually does so is exactly what the
        request-isolation fault family probes.
    """

    STATELESS_BATCH = "stateless_batch"
    CACHED_DECODE = "cached_decode"
    SESSION = "session"


class CheckpointKind(StrEnum):
    """Semantic address component naming *what* was captured, independent of call order."""

    EMBEDDING = "embedding"
    ATTN_OUT = "attn_out"
    MLP_OUT = "mlp_out"
    BLOCK_OUT = "block_out"
    FINAL_NORM = "final_norm"
    LOGITS = "logits"


class InvalidCaseError(ValueError):
    """Raised by ``Adapter.validate`` when a case violates the declared input contract."""


class AdapterExecutionError(RuntimeError):
    """Raised when an adapter fails for a reason that is not an input-contract violation."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_of(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Request:
    """One logical request: a sequence of valid tokens plus its padding and cache layout.

    Attributes
    ----------
    request_id:
        Stable identifier. It survives reduction: when the reducer deletes earlier requests
        from a session it must keep the target request's id unchanged, so that a failure
        signature recorded before reduction still addresses the same request afterwards.
    token_ids:
        The valid tokens, with no padding entries. ``PAD_TOKEN_ID`` is rejected here.
    prefix_length:
        Number of leading tokens consumed in the prefill call for cached execution. For
        STATELESS_BATCH it must equal ``len(token_ids)``.
    pad_left:
        Number of padding columns placed before the valid tokens when this request is
        materialized as a row of a padded batch. Ignored for cached execution, which is
        batch size one and unpadded.
    """

    request_id: str
    token_ids: tuple[int, ...]
    prefix_length: int
    pad_left: int = 0

    @property
    def n_valid(self) -> int:
        return len(self.token_ids)

    @property
    def n_decode_steps(self) -> int:
        """Tokens consumed one at a time after the prefill call."""
        return max(0, len(self.token_ids) - self.prefix_length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "token_ids": list(self.token_ids),
            "prefix_length": self.prefix_length,
            "pad_left": self.pad_left,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> Request:
        return Request(
            request_id=str(payload["request_id"]),
            token_ids=tuple(int(t) for t in payload["token_ids"]),
            prefix_length=int(payload["prefix_length"]),
            pad_left=int(payload.get("pad_left", 0)),
        )


@dataclass(frozen=True, slots=True)
class Case:
    """A fully specified, replayable comparison input.

    A case is pure data. It names the fixture configuration and weights it is valid
    against by hash, so replaying it in another process or another checkout either
    reproduces the same execution or fails loudly on a hash mismatch.
    """

    schema_version: int
    case_id: str
    model_config_id: str
    weights_sha256: str
    requests: tuple[Request, ...]
    execution_mode: ExecutionMode
    input_seed: int
    category: str = "uniform"
    provenance: tuple[str, ...] = ()

    @staticmethod
    def create(
        *,
        model_config_id: str,
        weights_sha256: str,
        requests: Sequence[Request],
        execution_mode: ExecutionMode,
        input_seed: int,
        category: str = "uniform",
        provenance: Sequence[str] = (),
    ) -> Case:
        """Build a case whose ``case_id`` is derived from its own canonical content."""
        draft = Case(
            schema_version=SCHEMA_VERSION,
            case_id="",
            model_config_id=model_config_id,
            weights_sha256=weights_sha256,
            requests=tuple(requests),
            execution_mode=execution_mode,
            input_seed=input_seed,
            category=category,
            provenance=tuple(provenance),
        )
        return replace(draft, case_id=f"case_{draft.content_hash()[:16]}")

    def content_hash(self) -> str:
        """Hash of everything that changes execution. Excludes ``case_id`` itself.

        ``provenance`` is excluded on purpose: it is bookkeeping about where a case came
        from, and two cases that execute identically must share a cache key.
        """
        return _sha256_of(
            {
                "schema_version": self.schema_version,
                "model_config_id": self.model_config_id,
                "weights_sha256": self.weights_sha256,
                "execution_mode": self.execution_mode.value,
                "input_seed": self.input_seed,
                "category": self.category,
                "requests": [r.to_dict() for r in self.requests],
            }
        )

    @property
    def total_valid_tokens(self) -> int:
        return sum(r.n_valid for r in self.requests)

    @property
    def total_padding_tokens(self) -> int:
        """Padding columns materialized for this case, zero outside stateless batching."""
        if self.execution_mode is not ExecutionMode.STATELESS_BATCH:
            return 0
        width = self.batch_width
        return sum(width - r.n_valid for r in self.requests)

    @property
    def batch_width(self) -> int:
        """Column count of the padded batch tensor for stateless execution."""
        if not self.requests:
            return 0
        return max(r.pad_left + r.n_valid for r in self.requests)

    def request_by_id(self, request_id: str) -> Request:
        for request in self.requests:
            if request.request_id == request_id:
                return request
        raise KeyError(f"no request {request_id!r} in {self.case_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "model_config_id": self.model_config_id,
            "weights_sha256": self.weights_sha256,
            "requests": [r.to_dict() for r in self.requests],
            "execution_mode": self.execution_mode.value,
            "input_seed": self.input_seed,
            "category": self.category,
            "provenance": list(self.provenance),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> Case:
        version = int(payload.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"case schema_version {version} is not supported by this build "
                f"(expected {SCHEMA_VERSION})"
            )
        return Case(
            schema_version=version,
            case_id=str(payload["case_id"]),
            model_config_id=str(payload["model_config_id"]),
            weights_sha256=str(payload["weights_sha256"]),
            requests=tuple(Request.from_dict(r) for r in payload["requests"]),
            execution_mode=ExecutionMode(payload["execution_mode"]),
            input_seed=int(payload["input_seed"]),
            category=str(payload.get("category", "uniform")),
            provenance=tuple(str(p) for p in payload.get("provenance", ())),
        )


@dataclass(frozen=True, slots=True)
class CheckpointAddress:
    """Stable semantic address of a captured activation.

    Raw hook invocation order is *not* an address: a full-prefix reference call fires each
    hook once for a whole sequence while a cached candidate fires it once per decode step.
    Aligning on ``(request_id, layer_name, token_position, kind)`` is what lets those two
    very different call schedules be compared at all.
    """

    request_id: str
    layer_name: str
    token_position: int
    kind: CheckpointKind

    def key(self) -> tuple[str, str, int, str]:
        return (self.request_id, self.layer_name, self.token_position, self.kind.value)

    def as_str(self) -> str:
        return f"{self.request_id}/{self.layer_name}/pos{self.token_position}/{self.kind.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_name": self.layer_name,
            "token_position": self.token_position,
            "kind": self.kind.value,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> CheckpointAddress:
        return CheckpointAddress(
            request_id=str(payload["request_id"]),
            layer_name=str(payload["layer_name"]),
            token_position=int(payload["token_position"]),
            kind=CheckpointKind(payload["kind"]),
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One captured activation vector plus its compact summary.

    ``values`` is retained only while the diagnostic capture budget allows it. When it is
    ``None`` the checkpoint still carries a summary, and alignment code must degrade to
    reporting "values unavailable" rather than silently comparing summaries as if they were
    element-wise evidence.
    """

    address: CheckpointAddress
    values: np.ndarray | None
    summary: dict[str, float]

    def to_dict(self, include_values: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "address": self.address.to_dict(),
            "summary": self.summary,
            "has_values": self.values is not None,
        }
        if include_values and self.values is not None:
            payload["values"] = [float(v) for v in np.asarray(self.values).ravel()]
        return payload


def summarize_tensor(values: np.ndarray) -> dict[str, float]:
    """Compact, comparison-free description of a tensor.

    Used for checkpoints whose values are dropped under the capture budget, and for
    reporting. Nonfinite entries are counted rather than propagated into the statistics.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]
    return {
        "count": float(flat.size),
        "n_finite": float(int(finite_mask.sum())),
        "n_nan": float(int(np.isnan(flat).sum())),
        "n_inf": float(int(np.isinf(flat).sum())),
        "mean": float(finite.mean()) if finite.size else 0.0,
        "absmax": float(np.abs(finite).max()) if finite.size else 0.0,
        "l2": float(np.sqrt((finite**2).sum())) if finite.size else 0.0,
    }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What one adapter produced for one case.

    ``outputs`` maps ``request_id`` to float32 logits of shape ``[n_valid, vocab]``. Only
    valid logical positions appear, in ascending position order.
    """

    adapter_id: str
    case_id: str
    outputs: Mapping[str, np.ndarray]
    checkpoints: tuple[Checkpoint, ...] = ()
    capture_enabled: bool = False
    wall_time_ns: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def checkpoint_index(self) -> dict[tuple[str, str, int, str], Checkpoint]:
        return {cp.address.key(): cp for cp in self.checkpoints}


@dataclass(frozen=True, slots=True)
class TolerancePolicy:
    """Explicit numerical policy. ``policy_id`` participates in every predicate cache key.

    ``violation = abs(candidate - reference) > atol + rtol * abs(reference)``

    ``zero_norm_eps`` defines the denominator rule for relative L2 error: when the reference
    tensor's L2 norm is below it, relative error is reported as the absolute L2 difference
    and flagged, rather than dividing by a near-zero norm to manufacture a large headline
    number.
    """

    atol: float = 1e-5
    rtol: float = 1e-4
    zero_norm_eps: float = 1e-12
    name: str = "cpu_fp32_default"

    @property
    def policy_id(self) -> str:
        return _sha256_of(
            {
                "atol": self.atol,
                "rtol": self.rtol,
                "zero_norm_eps": self.zero_norm_eps,
                "name": self.name,
            }
        )[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "atol": self.atol,
            "rtol": self.rtol,
            "zero_norm_eps": self.zero_norm_eps,
            "name": self.name,
            "policy_id": self.policy_id,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> TolerancePolicy:
        return TolerancePolicy(
            atol=float(payload["atol"]),
            rtol=float(payload["rtol"]),
            zero_norm_eps=float(payload.get("zero_norm_eps", 1e-12)),
            name=str(payload.get("name", "cpu_fp32_default")),
        )


@dataclass(frozen=True, slots=True)
class TensorDiff:
    """Element-wise comparison evidence for one aligned tensor pair."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    max_abs_err: float
    rel_l2_err: float
    rel_l2_denominator_degenerate: bool
    violating_fraction: float
    n_violations: int
    n_elements: int
    reference_nonfinite: int
    candidate_nonfinite: int

    @property
    def violated(self) -> bool:
        return self.n_violations > 0 or self.candidate_nonfinite > self.reference_nonfinite

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "max_abs_err": self.max_abs_err,
            "rel_l2_err": self.rel_l2_err,
            "rel_l2_denominator_degenerate": self.rel_l2_denominator_degenerate,
            "violating_fraction": self.violating_fraction,
            "n_violations": self.n_violations,
            "n_elements": self.n_elements,
            "reference_nonfinite": self.reference_nonfinite,
            "candidate_nonfinite": self.candidate_nonfinite,
            "violated": self.violated,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Verdict plus the evidence that produced it."""

    verdict: Verdict
    case_id: str
    policy: TolerancePolicy
    reference_adapter: str
    candidate_adapter: str
    diffs: tuple[TensorDiff, ...] = ()
    failing_request_ids: tuple[str, ...] = ()
    detail: str = ""
    reference_wall_time_ns: int = 0
    candidate_wall_time_ns: int = 0

    @property
    def max_abs_err(self) -> float:
        return max((d.max_abs_err for d in self.diffs), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "case_id": self.case_id,
            "policy": self.policy.to_dict(),
            "reference_adapter": self.reference_adapter,
            "candidate_adapter": self.candidate_adapter,
            "diffs": [d.to_dict() for d in self.diffs],
            "failing_request_ids": list(self.failing_request_ids),
            "detail": self.detail,
            "max_abs_err": self.max_abs_err,
            "reference_wall_time_ns": self.reference_wall_time_ns,
            "candidate_wall_time_ns": self.candidate_wall_time_ns,
        }


@dataclass(frozen=True, slots=True)
class FailureSignature:
    """What a reduction must preserve.

    A reduction is accepted only when the reduced case still produces this signature. The
    signature intentionally does *not* include error magnitudes, which legitimately shrink
    as an input shrinks. It cannot prove that two cases share a root cause; it establishes
    that the same failure class is still observed at the same target request, and (when
    ``checkpoint`` is set) still exposed at the same semantic checkpoint address.
    """

    verdict: Verdict
    target_request_id: str
    execution_mode: ExecutionMode
    checkpoint: CheckpointAddress | None = None

    def matches(self, other: FailureSignature) -> bool:
        if self.verdict is not other.verdict:
            return False
        if self.target_request_id != other.target_request_id:
            return False
        if self.execution_mode is not other.execution_mode:
            return False
        if self.checkpoint is None:
            return True
        if other.checkpoint is None:
            return False
        # Position indices shift as tokens are deleted, so the preserved identity is the
        # (request, layer, kind) address; the position is reported but not required to match.
        return (
            self.checkpoint.request_id == other.checkpoint.request_id
            and self.checkpoint.layer_name == other.checkpoint.layer_name
            and self.checkpoint.kind is other.checkpoint.kind
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "target_request_id": self.target_request_id,
            "execution_mode": self.execution_mode.value,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> FailureSignature:
        cp = payload.get("checkpoint")
        return FailureSignature(
            verdict=Verdict(payload["verdict"]),
            target_request_id=str(payload["target_request_id"]),
            execution_mode=ExecutionMode(payload["execution_mode"]),
            checkpoint=CheckpointAddress.from_dict(cp) if cp else None,
        )


@dataclass(frozen=True, slots=True)
class CaseSize:
    """Lexicographic size used to require strict improvement during reduction.

    Ordering: fewer requests, then fewer valid tokens, then fewer padding tokens, then
    lower token-value complexity (the number of distinct token values, then their sum).
    Requiring strict lexicographic decrease is what prevents the reducer from cycling
    between two same-size representations forever.
    """

    n_requests: int
    n_valid_tokens: int
    n_padding_tokens: int
    token_value_complexity: int

    @staticmethod
    def of(case: Case) -> CaseSize:
        distinct = {t for r in case.requests for t in r.token_ids}
        complexity = len(distinct) * 1_000_000 + sum(sorted(distinct))
        return CaseSize(
            n_requests=len(case.requests),
            n_valid_tokens=case.total_valid_tokens,
            n_padding_tokens=case.total_padding_tokens,
            token_value_complexity=complexity,
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.n_requests,
            self.n_valid_tokens,
            self.n_padding_tokens,
            self.token_value_complexity,
        )

    def __lt__(self, other: CaseSize) -> bool:
        return self.as_tuple() < other.as_tuple()

    def __le__(self, other: CaseSize) -> bool:
        return self.as_tuple() <= other.as_tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "n_valid_tokens": self.n_valid_tokens,
            "n_padding_tokens": self.n_padding_tokens,
            "token_value_complexity": self.token_value_complexity,
        }


@runtime_checkable
class Adapter(Protocol):
    """The only surface the debugging engine is allowed to use.

    Deliberately narrow: the generator, comparator, and reducer reach a model exclusively
    through these four members. That is what keeps the search policy unable to inspect
    mutant identities, source patches, or expected answers.
    """

    @property
    def adapter_id(self) -> str: ...

    def reset(self) -> None:
        """Return to clean state. Every replay begins with this."""
        ...

    def validate(self, case: Case) -> None:
        """Raise :class:`InvalidCaseError` if the case violates the input contract."""
        ...

    def run(self, case: Case, capture: bool = False) -> ExecutionResult: ...


__all__ = [
    "CANONICAL_TOKEN_ID",
    "PAD_TOKEN_ID",
    "SCHEMA_VERSION",
    "Adapter",
    "AdapterExecutionError",
    "Case",
    "CaseSize",
    "Checkpoint",
    "CheckpointAddress",
    "CheckpointKind",
    "ComparisonResult",
    "ExecutionMode",
    "ExecutionResult",
    "FailureSignature",
    "InvalidCaseError",
    "Request",
    "TensorDiff",
    "TolerancePolicy",
    "Verdict",
    "summarize_tensor",
]
