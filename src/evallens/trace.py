"""Semantic checkpoint alignment and earliest-observed-divergence localization.

What this module claims
-----------------------
It reports the **earliest observed divergence**: the first checkpoint, in a declared
traversal order, where the candidate leaves the tolerance band. That is evidence about where
divergence becomes *visible with the checkpoints this adapter exposes*. It is not proof of
root cause, and when adapters expose only layer outputs it cannot identify an operation
inside a layer.

Why every checkpoint is compared
--------------------------------
No binary search. Bisection requires the "diverged" predicate to be monotone along the
traversal, and it is not: floating-point discrepancies appear, shrink back inside tolerance,
and reappear. A real example lives in this project's own corpus — the `causal_mask.leak_last`
variant leaks the final key column, which at the *final* position is legitimately visible, so
that position agrees at every layer while every earlier position disagrees. A bisection over
that sequence would land wherever the midpoint happened to fall.

Comparing all aligned checkpoints costs one linear pass over a bounded capture. That is
cheap, and it is the only version that is correct.

The traversal order
-------------------
`(request index, network depth, token position)`.

Depth is the dominant axis because it is the causal one: a difference at block 0 must precede
any difference it causes at block 1. Within a depth, lower token positions come first, since
position `p`'s activations depend only on positions `<= p`.

This is a *total* order imposed on a partial one. Positions within a layer are computed
independently in a full-prefix pass and are not causally ordered relative to each other. The
order is declared, recorded, and stable; it is not a claim that the model computes in exactly
this sequence.

Alignment
---------
Checkpoints are matched by `(request_id, layer_name, token_position, kind)` — never by hook
invocation order. One full-prefix reference call records a whole sequence at once while a
cached candidate records one decode step at a time, so call order is not a correspondence.
Unmatched checkpoints on either side are reported rather than dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from evallens.compare import compare_arrays
from evallens.types import (
    Adapter,
    Case,
    Checkpoint,
    CheckpointAddress,
    CheckpointKind,
    ExecutionResult,
    TensorDiff,
    TolerancePolicy,
)

KIND_DEPTH_OFFSET: dict[CheckpointKind, int] = {
    CheckpointKind.EMBEDDING: 0,
    CheckpointKind.ATTN_OUT: 0,
    CheckpointKind.MLP_OUT: 1,
    CheckpointKind.BLOCK_OUT: 2,
    CheckpointKind.FINAL_NORM: 0,
    CheckpointKind.LOGITS: 1,
}
"""Rank of each checkpoint kind *within* its layer, following the dataflow of a block."""


def layer_rank(layer_name: str) -> int:
    """Position of a layer in the network, derived from its name.

    ``embed`` first, then ``block0 .. blockN`` in index order, then ``final``. Deriving this
    from the name rather than from a config means the traversal works for any adapter that
    follows the naming convention, including future ones.
    """
    if layer_name == "embed":
        return 0
    if layer_name.startswith("block"):
        return 1 + int(layer_name.removeprefix("block"))
    if layer_name == "final":
        return 1_000_000
    raise ValueError(f"unknown layer name {layer_name!r}")


def checkpoint_depth(address: CheckpointAddress) -> int:
    return layer_rank(address.layer_name) * 10 + KIND_DEPTH_OFFSET[address.kind]


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Which checkpoint addresses the two traced passes have in common."""

    matched: tuple[CheckpointAddress, ...]
    reference_only: tuple[CheckpointAddress, ...]
    candidate_only: tuple[CheckpointAddress, ...]
    reference_total: int
    candidate_total: int

    @property
    def aligned_fraction(self) -> float:
        larger = max(self.reference_total, self.candidate_total)
        return len(self.matched) / larger if larger else 0.0

    @property
    def fully_aligned(self) -> bool:
        return not self.reference_only and not self.candidate_only

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_matched": len(self.matched),
            "n_reference_only": len(self.reference_only),
            "n_candidate_only": len(self.candidate_only),
            "reference_total": self.reference_total,
            "candidate_total": self.candidate_total,
            "aligned_fraction": self.aligned_fraction,
            "fully_aligned": self.fully_aligned,
            "reference_only": [a.as_str() for a in self.reference_only[:16]],
            "candidate_only": [a.as_str() for a in self.candidate_only[:16]],
        }


@dataclass(frozen=True, slots=True)
class CheckpointComparison:
    """One aligned checkpoint pair, in traversal order."""

    address: CheckpointAddress
    order_index: int
    depth: int
    diff: TensorDiff | None
    values_available: bool

    @property
    def diverged(self) -> bool:
        return self.diff is not None and self.diff.violated

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict(),
            "address_str": self.address.as_str(),
            "order_index": self.order_index,
            "depth": self.depth,
            "values_available": self.values_available,
            "diverged": self.diverged,
            "diff": self.diff.to_dict() if self.diff else None,
        }


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Where divergence first becomes observable, and how it behaves after that."""

    available: bool
    reason: str
    alignment: AlignmentReport
    comparisons: tuple[CheckpointComparison, ...]
    earliest_observed: CheckpointAddress | None
    policy: TolerancePolicy
    capture_wall_time_ns: int = 0

    @property
    def divergent(self) -> tuple[CheckpointComparison, ...]:
        return tuple(c for c in self.comparisons if c.diverged)

    @property
    def reconverged(self) -> bool:
        """True when at least one aligned checkpoint after a divergence is back in tolerance.

        This is the direct evidence that the diverged predicate is not monotone along the
        traversal, which is why localization compares every checkpoint instead of bisecting.
        """
        seen_divergence = False
        for comparison in self.comparisons:
            if comparison.diverged:
                seen_divergence = True
            elif seen_divergence and comparison.diff is not None:
                return True
        return False

    @property
    def divergent_layers(self) -> tuple[str, ...]:
        seen: list[str] = []
        for comparison in self.divergent:
            if comparison.address.layer_name not in seen:
                seen.append(comparison.address.layer_name)
        return tuple(seen)

    def summary(self) -> str:
        if not self.available:
            return f"localization unavailable: {self.reason}"
        if self.earliest_observed is None:
            return (
                f"no aligned checkpoint diverges under {self.policy.name}; the failure is "
                "observable only at the output"
            )
        note = " (discrepancy reconverges later; not monotone)" if self.reconverged else ""
        return (
            f"earliest observed divergence at {self.earliest_observed.as_str()}; "
            f"{len(self.divergent)}/{len(self.comparisons)} aligned checkpoints diverge{note}"
        )

    def to_dict(self, max_comparisons: int = 256) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "summary": self.summary(),
            "alignment": self.alignment.to_dict(),
            "earliest_observed": (
                self.earliest_observed.to_dict() if self.earliest_observed else None
            ),
            "earliest_observed_str": (
                self.earliest_observed.as_str() if self.earliest_observed else None
            ),
            "n_compared": len(self.comparisons),
            "n_divergent": len(self.divergent),
            "divergent_layers": list(self.divergent_layers),
            "reconverged": self.reconverged,
            "policy": self.policy.to_dict(),
            "capture_wall_time_ns": self.capture_wall_time_ns,
            "comparisons": [c.to_dict() for c in self.comparisons[:max_comparisons]],
            "comparisons_truncated": len(self.comparisons) > max_comparisons,
            "interpretation": (
                "Earliest observed divergence is evidence about where a difference becomes "
                "visible at the checkpoints these adapters expose. It is not proof of root "
                "cause, and it cannot identify an operation inside a layer."
            ),
        }


def align_checkpoints(reference: ExecutionResult, candidate: ExecutionResult) -> AlignmentReport:
    """Match checkpoints by semantic address, reporting anything unmatched on either side."""
    reference_index = reference.checkpoint_index()
    candidate_index = candidate.checkpoint_index()
    reference_keys = set(reference_index)
    candidate_keys = set(candidate_index)

    def addresses(
        keys: set[tuple[str, str, int, str]],
        source: dict[tuple[str, str, int, str], Checkpoint],
    ) -> tuple[CheckpointAddress, ...]:
        return tuple(sorted((source[k].address for k in keys), key=_sort_key_without_request))

    return AlignmentReport(
        matched=addresses(reference_keys & candidate_keys, reference_index),
        reference_only=addresses(reference_keys - candidate_keys, reference_index),
        candidate_only=addresses(candidate_keys - reference_keys, candidate_index),
        reference_total=len(reference_index),
        candidate_total=len(candidate_index),
    )


def _sort_key_without_request(address: CheckpointAddress) -> tuple[int, int, str]:
    return (checkpoint_depth(address), address.token_position, address.request_id)


def _request_order(reference: ExecutionResult) -> dict[str, int]:
    """Request ordering taken from the reference's output mapping, which adapters build in
    case order. Falls back to alphabetical for any id that is not present."""
    return {request_id: index for index, request_id in enumerate(reference.outputs)}


def traversal_key(
    address: CheckpointAddress, request_order: dict[str, int]
) -> tuple[int, int, int]:
    """The declared total order: request, then network depth, then token position."""
    return (
        request_order.get(address.request_id, len(request_order)),
        checkpoint_depth(address),
        address.token_position,
    )


def localize(
    reference: ExecutionResult,
    candidate: ExecutionResult,
    policy: TolerancePolicy,
    *,
    capture_wall_time_ns: int = 0,
) -> LocalizationResult:
    """Compare every aligned checkpoint in traversal order and report the earliest divergence.

    Both results must come from a traced pass. When nothing aligns, the caller's output-level
    failure stands on its own and localization is reported as unavailable — never silently
    replaced by a guess.
    """
    alignment = align_checkpoints(reference, candidate)

    if not reference.capture_enabled or not candidate.capture_enabled:
        return LocalizationResult(
            available=False,
            reason=(
                "capture was not enabled on both adapters; localization requires a traced pass"
            ),
            alignment=alignment,
            comparisons=(),
            earliest_observed=None,
            policy=policy,
            capture_wall_time_ns=capture_wall_time_ns,
        )

    if not alignment.matched:
        return LocalizationResult(
            available=False,
            reason=(
                "no checkpoints align between these adapters; the output-level failure stands "
                "on its own"
            ),
            alignment=alignment,
            comparisons=(),
            earliest_observed=None,
            policy=policy,
            capture_wall_time_ns=capture_wall_time_ns,
        )

    reference_index = reference.checkpoint_index()
    candidate_index = candidate.checkpoint_index()
    order = _request_order(reference)
    ordered = sorted(alignment.matched, key=lambda a: traversal_key(a, order))

    comparisons: list[CheckpointComparison] = []
    for index, address in enumerate(ordered):
        reference_checkpoint: Checkpoint = reference_index[address.key()]
        candidate_checkpoint: Checkpoint = candidate_index[address.key()]
        available = (
            reference_checkpoint.values is not None and candidate_checkpoint.values is not None
        )
        diff: TensorDiff | None = None
        if available:
            assert reference_checkpoint.values is not None
            assert candidate_checkpoint.values is not None
            if reference_checkpoint.values.shape == candidate_checkpoint.values.shape:
                diff = compare_arrays(
                    address.as_str(),
                    reference_checkpoint.values,
                    candidate_checkpoint.values,
                    policy,
                )
            else:
                available = False
        comparisons.append(
            CheckpointComparison(
                address=address,
                order_index=index,
                depth=checkpoint_depth(address),
                diff=diff,
                values_available=available,
            )
        )

    compared = [c for c in comparisons if c.diff is not None]
    if not compared:
        return LocalizationResult(
            available=False,
            reason=(
                "aligned checkpoints exist but their values were dropped under the capture "
                "budget; only summaries remain, which are not element-wise evidence"
            ),
            alignment=alignment,
            comparisons=tuple(comparisons),
            earliest_observed=None,
            policy=policy,
            capture_wall_time_ns=capture_wall_time_ns,
        )

    earliest = next((c.address for c in comparisons if c.diverged), None)
    return LocalizationResult(
        available=True,
        reason="aligned checkpoints compared in declared traversal order",
        alignment=alignment,
        comparisons=tuple(comparisons),
        earliest_observed=earliest,
        policy=policy,
        capture_wall_time_ns=capture_wall_time_ns,
    )


def trace_and_localize(
    reference: Adapter,
    candidate: Adapter,
    case: Case,
    policy: TolerancePolicy,
) -> LocalizationResult:
    """Run a **separate** traced pass on both adapters and localize.

    Separate on purpose. Capture allocates, serializes, and slows execution, so folding it
    into the detection path would put that cost inside one side of a timing comparison and
    nowhere else. Detection decides *whether* something is wrong; this decides *where it
    first shows*, and it runs only after a stable failure has already been established.
    """
    started = time.perf_counter_ns()
    reference.reset()
    candidate.reset()
    reference_result = reference.run(case, capture=True)
    candidate_result = candidate.run(case, capture=True)
    elapsed = time.perf_counter_ns() - started
    return localize(reference_result, candidate_result, policy, capture_wall_time_ns=elapsed)


__all__ = [
    "KIND_DEPTH_OFFSET",
    "AlignmentReport",
    "CheckpointComparison",
    "LocalizationResult",
    "align_checkpoints",
    "checkpoint_depth",
    "layer_rank",
    "localize",
    "trace_and_localize",
    "traversal_key",
]
