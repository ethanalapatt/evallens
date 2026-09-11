"""Isolated, repeatable case execution and stability replay.

Two jobs:

1. Turn "run these two adapters on this case" into exactly one :class:`ComparisonResult`,
   with every failure mode landing in its own verdict category. An exception is ``ERROR``, a
   contract violation is ``INVALID``, an overrun is ``TIMEOUT`` or ``RESOURCE_LIMIT``. None
   of these are detections, and none of them are quietly swallowed.

2. Decide whether an observed failure is *stable*. A candidate failure is re-run from clean
   state three times by default; only a classification that holds every time enters
   reduction. A classification that changes is ``UNSTABLE`` and stays visible in the report
   rather than being retried until it produces a convenient answer.

On timeouts: execution happens in-process, so the per-case wall-clock limit is checked after
the call returns rather than interrupting a running kernel. That is reported honestly — a
``TIMEOUT`` here means "this case exceeded its budget", not "this case was killed at its
budget". The fresh-subprocess replay used for final verification does enforce a hard limit.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from evallens.compare import compare
from evallens.resources import (
    DEFAULT_RSS_LIMIT_BYTES,
    ResourceGuard,
    ResourceLimitExceeded,
)
from evallens.types import (
    Case,
    ComparisonResult,
    ExecutionResult,
    InvalidCaseError,
    TolerancePolicy,
    Verdict,
)

DEFAULT_STABILITY_REPLAYS = 3
DEFAULT_CASE_TIMEOUT_S = 30.0


class ResettableAdapter(Protocol):
    """An :class:`Adapter` — named separately only to document the reset requirement."""

    @property
    def adapter_id(self) -> str: ...

    def reset(self) -> None: ...

    def validate(self, case: Case) -> None: ...

    def run(self, case: Case, capture: bool = False) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class ReplayBudget:
    timeout_s: float = DEFAULT_CASE_TIMEOUT_S
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES


@dataclass(slots=True)
class RunCounters:
    """Separates *logical* predicate queries from the model work they actually cost."""

    model_runs: int = 0
    comparisons: int = 0
    errors: int = 0
    peak_rss_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_runs": self.model_runs,
            "comparisons": self.comparisons,
            "errors": self.errors,
            "peak_rss_bytes": self.peak_rss_bytes,
        }


def _empty(
    verdict: Verdict,
    case: Case,
    policy: TolerancePolicy,
    reference: ResettableAdapter,
    candidate: ResettableAdapter,
    detail: str,
) -> ComparisonResult:
    return ComparisonResult(
        verdict=verdict,
        case_id=case.case_id,
        policy=policy,
        reference_adapter=reference.adapter_id,
        candidate_adapter=candidate.adapter_id,
        detail=detail,
    )


def run_comparison(
    reference: ResettableAdapter,
    candidate: ResettableAdapter,
    case: Case,
    policy: TolerancePolicy,
    *,
    budget: ReplayBudget | None = None,
    counters: RunCounters | None = None,
    capture: bool = False,
) -> ComparisonResult:
    """Execute both adapters from clean state and classify the outcome.

    ``capture`` is off by default. Diagnostic capture belongs to a separate traced pass so
    that it never lands inside a detection timing measurement.
    """
    budget = budget or ReplayBudget()
    counters = counters or RunCounters()
    guard = ResourceGuard(budget.rss_limit_bytes, raise_on_exceed=True)
    started = time.perf_counter_ns()

    try:
        reference.reset()
        candidate.reset()
        reference.validate(case)
        candidate.validate(case)
    except InvalidCaseError as exc:
        return _empty(Verdict.INVALID, case, policy, reference, candidate, str(exc))
    except Exception as exc:
        counters.errors += 1
        return _empty(
            Verdict.ERROR, case, policy, reference, candidate, f"{type(exc).__name__}: {exc}"
        )

    try:
        guard.sample()
        reference_result = reference.run(case, capture=capture)
        counters.model_runs += 1
        guard.sample()
        candidate_result = candidate.run(case, capture=capture)
        counters.model_runs += 1
        guard.sample()
    except InvalidCaseError as exc:
        return _empty(Verdict.INVALID, case, policy, reference, candidate, str(exc))
    except ResourceLimitExceeded as exc:
        return _empty(Verdict.RESOURCE_LIMIT, case, policy, reference, candidate, str(exc))
    except Exception as exc:
        counters.errors += 1
        return _empty(
            Verdict.ERROR, case, policy, reference, candidate, f"{type(exc).__name__}: {exc}"
        )
    finally:
        counters.peak_rss_bytes = guard.peak_rss_bytes

    elapsed_s = (time.perf_counter_ns() - started) / 1e9
    if elapsed_s > budget.timeout_s:
        return _empty(
            Verdict.TIMEOUT,
            case,
            policy,
            reference,
            candidate,
            f"case took {elapsed_s:.2f}s, exceeding the {budget.timeout_s:.2f}s budget",
        )

    counters.comparisons += 1
    return compare(reference_result, candidate_result, policy)


@dataclass(slots=True)
class StabilityResult:
    """The outcome of re-running one case several times from clean state."""

    verdict: Verdict
    stable: bool
    observations: tuple[ComparisonResult, ...] = field(default_factory=tuple)
    replays: int = 0

    @property
    def representative(self) -> ComparisonResult:
        """A comparison carrying the evidence for the decided verdict."""
        for observation in self.observations:
            if observation.verdict is self.verdict:
                return observation
        return self.observations[0]

    @property
    def verdict_counts(self) -> dict[str, int]:
        return dict(Counter(o.verdict.value for o in self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "stable": self.stable,
            "replays": self.replays,
            "verdict_counts": self.verdict_counts,
            "representative": self.representative.to_dict() if self.observations else None,
        }


def stable_comparison(
    reference: ResettableAdapter,
    candidate: ResettableAdapter,
    case: Case,
    policy: TolerancePolicy,
    *,
    replays: int = DEFAULT_STABILITY_REPLAYS,
    budget: ReplayBudget | None = None,
    counters: RunCounters | None = None,
) -> StabilityResult:
    """Run ``case`` ``replays`` times from clean state and require an unchanging verdict.

    Cost is charged to ``counters``: stability replays are real model runs and they are not
    free. Hiding them would understate the cost of every detection in the benchmark.
    """
    if replays < 1:
        raise ValueError("replays must be at least 1")

    observations: list[ComparisonResult] = []
    for _ in range(replays):
        observation = run_comparison(
            reference, candidate, case, policy, budget=budget, counters=counters
        )
        observations.append(observation)
        # An input-contract violation is a property of the case, not of a particular run.
        if observation.verdict is Verdict.INVALID:
            return StabilityResult(Verdict.INVALID, True, tuple(observations), len(observations))

    verdicts = {observation.verdict for observation in observations}
    if len(verdicts) == 1:
        return StabilityResult(observations[0].verdict, True, tuple(observations), replays)
    return StabilityResult(Verdict.UNSTABLE, False, tuple(observations), replays)


__all__ = [
    "DEFAULT_CASE_TIMEOUT_S",
    "DEFAULT_STABILITY_REPLAYS",
    "ReplayBudget",
    "ResettableAdapter",
    "RunCounters",
    "StabilityResult",
    "run_comparison",
    "stable_comparison",
]
