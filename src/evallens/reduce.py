"""Failure-preserving input minimization.

Turning a 90-token, three-request session that fails into a two-token single request that
fails the same way is most of what makes a discovered regression actionable. This module does
that with a structure-aware `ddmin`, plus a separately implemented greedy single-deletion
baseline to compare it against.

What is reduced, in order
-------------------------
1. Earlier requests are removed from a session, always retaining the target request.
2. Token chunks are removed, keeping every request nonempty and the prefill boundary valid.
3. Padding is removed from stateless rows.
4. Token values are simplified toward :data:`~evallens.types.CANONICAL_TOKEN_ID`.

Every pass repeats until a full round accepts nothing, or the budget runs out.

What is never touched
---------------------
Weights, tolerance policy, adapters, and model architecture. Minimizing an *input* means
changing the input. Anything else would be changing the experiment to get a smaller answer.

Acceptance
----------
A transformation is accepted only when the result is **valid**, **strictly smaller** in the
lexicographic order `(requests, valid tokens, padding tokens, token complexity)`, in the
**same declared category**, and still reproduces a **stable** failure matching the recorded
signature. Strictness is what guarantees termination: each acceptance moves the case down a
well-founded order, so the loop cannot cycle between two equivalent representations.

Category preservation matters more than it looks. Shrinking a two-row batch to one row would
produce a smaller case that fails — but it would no longer be testing batching, and calling
that a reduction of the original failure would be wrong.

Blindness
---------
Nothing here may import `bench`, read a `Behavior`, or learn which candidate it is reducing
for. The reducer sees an `Adapter`, a `TolerancePolicy`, and a `FailureSignature` derived from
the failure it was handed. Validation goes through `adapter.validate`, so the reducer does not
even need the model config.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from evallens.env import numeric_identity
from evallens.generate import CaseCategory, classify_case
from evallens.replay import ReplayBudget, RunCounters, stable_comparison
from evallens.trace import trace_and_localize
from evallens.types import (
    CANONICAL_TOKEN_ID,
    Adapter,
    Case,
    CaseSize,
    ExecutionMode,
    FailureSignature,
    InvalidCaseError,
    Request,
    TolerancePolicy,
    Verdict,
)

DEFAULT_MAX_QUERIES = 256
DEFAULT_TIME_BUDGET_S = 60.0


class MinimalityStatus(StrEnum):
    """How strong a claim the result supports.

    ``ONE_MINIMAL`` means *1-minimal with respect to the declared deletion operations*: every
    remaining eligible single deletion was checked and none preserved the failure. It is not
    a claim that this is the globally smallest counterexample, which this search cannot
    establish and never asserts.
    """

    ONE_MINIMAL = "one_minimal_wrt_declared_operations"
    NOT_ESTABLISHED = "reduced_minimality_not_established"


@dataclass(slots=True)
class ReductionBudget:
    """Limits shared identically by every reducer, so a comparison is fair."""

    max_queries: int = DEFAULT_MAX_QUERIES
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    stability_replays: int = 3
    case_timeout_s: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_queries": self.max_queries,
            "time_budget_s": self.time_budget_s,
            "stability_replays": self.stability_replays,
            "case_timeout_s": self.case_timeout_s,
        }


@dataclass(slots=True)
class PredicateCounters:
    """Logical queries, real model runs, and cache hits are counted separately.

    They are very different quantities. A cached query costs nothing; a query that survives
    the cache costs ``stability_replays`` full comparisons, each of which runs both adapters.
    Reporting only one of the three would misrepresent the cost of every reduction.
    """

    logical_queries: int = 0
    cache_hits: int = 0
    executed_queries: int = 0
    model_runs: int = 0
    invalid_rejections: int = 0
    category_rejections: int = 0
    size_rejections: int = 0
    localization_queries: int = 0
    elapsed_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_queries": self.logical_queries,
            "cache_hits": self.cache_hits,
            "executed_queries": self.executed_queries,
            "model_runs": self.model_runs,
            "invalid_rejections": self.invalid_rejections,
            "category_rejections": self.category_rejections,
            "size_rejections": self.size_rejections,
            "localization_queries": self.localization_queries,
            "elapsed_s": round(self.elapsed_ns / 1e9, 4),
            "cache_hit_rate": (
                self.cache_hits / self.logical_queries if self.logical_queries else 0.0
            ),
        }


@dataclass(frozen=True, slots=True)
class PredicateOutcome:
    preserved: bool
    verdict: Verdict
    detail: str
    from_cache: bool = False


class BudgetExhaustedSignal(Exception):
    """Internal control flow: the budget ran out mid-search. The best case is preserved."""


class FailurePredicate:
    """Does this case still reproduce the recorded failure?

    Answers only that. It knows the two adapters, the policy, and the signature to preserve;
    it does not know what is wrong with the candidate, and it has no way to find out.

    The cache is keyed by the case's canonical content hash together with every identity that
    could change the answer: both adapter ids, the weights hash, the policy id, the signature,
    and the numeric environment. Omitting any of those would let a stale verdict silently
    corrupt a whole reduction.
    """

    def __init__(
        self,
        reference: Adapter,
        candidate: Adapter,
        policy: TolerancePolicy,
        signature: FailureSignature,
        *,
        budget: ReductionBudget | None = None,
        counters: PredicateCounters | None = None,
        environment_id: str | None = None,
    ) -> None:
        self.reference = reference
        self.candidate = candidate
        self.policy = policy
        self.signature = signature
        self.budget = budget or ReductionBudget()
        self.counters = counters or PredicateCounters()
        self.environment_id = environment_id or numeric_identity()
        self._cache: dict[str, PredicateOutcome] = {}
        self._started_ns = time.perf_counter_ns()
        self._run_counters = RunCounters()

    # -- identity ---------------------------------------------------------------------------

    @property
    def identity(self) -> dict[str, str]:
        return {
            "reference_adapter": self.reference.adapter_id,
            "candidate_adapter": self.candidate.adapter_id,
            "policy_id": self.policy.policy_id,
            "environment_id": self.environment_id,
            "signature": str(self.signature.to_dict()),
        }

    def cache_key(self, case: Case) -> str:
        parts = [case.content_hash(), *(f"{k}={v}" for k, v in sorted(self.identity.items()))]
        return "|".join(parts)

    # -- budget -----------------------------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return (time.perf_counter_ns() - self._started_ns) / 1e9

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.counters.logical_queries >= self.budget.max_queries
            or self.elapsed_s >= self.budget.time_budget_s
        )

    def check_budget(self) -> None:
        if self.budget_exhausted:
            raise BudgetExhaustedSignal(
                f"reduction budget exhausted after {self.counters.logical_queries} queries "
                f"and {self.elapsed_s:.2f}s"
            )

    # -- the predicate ------------------------------------------------------------------------

    def __call__(self, case: Case) -> PredicateOutcome:
        self.check_budget()
        self.counters.logical_queries += 1

        key = self.cache_key(case)
        cached = self._cache.get(key)
        if cached is not None:
            self.counters.cache_hits += 1
            return PredicateOutcome(cached.preserved, cached.verdict, cached.detail, True)

        outcome = self._evaluate(case)
        self._cache[key] = outcome
        self.counters.elapsed_ns = time.perf_counter_ns() - self._started_ns
        return outcome

    def _evaluate(self, case: Case) -> PredicateOutcome:
        try:
            self.candidate.validate(case)
        except InvalidCaseError as exc:
            self.counters.invalid_rejections += 1
            return PredicateOutcome(False, Verdict.INVALID, str(exc))

        self.counters.executed_queries += 1
        before = self._run_counters.model_runs
        outcome = stable_comparison(
            self.reference,
            self.candidate,
            case,
            self.policy,
            replays=self.budget.stability_replays,
            budget=ReplayBudget(timeout_s=self.budget.case_timeout_s),
            counters=self._run_counters,
        )
        self.counters.model_runs += self._run_counters.model_runs - before

        if not outcome.stable or outcome.verdict is not Verdict.FAIL:
            return PredicateOutcome(
                False,
                outcome.verdict,
                outcome.representative.detail if outcome.observations else "",
            )

        representative = outcome.representative
        if self.signature.target_request_id not in representative.failing_request_ids:
            return PredicateOutcome(
                False,
                outcome.verdict,
                f"fails, but not at the target request {self.signature.target_request_id!r}",
            )

        observed = FailureSignature(
            verdict=Verdict.FAIL,
            target_request_id=self.signature.target_request_id,
            execution_mode=case.execution_mode,
            checkpoint=None,
        )
        if self.signature.checkpoint is not None:
            observed = self._with_checkpoint(case, observed)

        if not self.signature.matches(observed):
            return PredicateOutcome(
                False, outcome.verdict, "fails, but with a different failure signature"
            )
        return PredicateOutcome(True, Verdict.FAIL, representative.detail)

    def _with_checkpoint(self, case: Case, observed: FailureSignature) -> FailureSignature:
        """Run the extra traced pass only when checkpoint identity must be preserved."""
        self.counters.localization_queries += 1
        localization = trace_and_localize(self.reference, self.candidate, case, self.policy)
        if not localization.available:
            return observed
        return FailureSignature(
            observed.verdict,
            observed.target_request_id,
            observed.execution_mode,
            localization.earliest_observed,
        )


# --- transformations --------------------------------------------------------------------------


def _rebuild_request(
    request: Request, tokens: Sequence[int], pad_left: int | None = None
) -> Request:
    """Rebuild a request from reduced tokens, reconstructing its prefill boundary.

    Shrinking a request renumbers its positions, so the boundary has to be derived from the
    reduced case rather than copied. The rule preserves the *kind* of boundary: a
    prefill-only request stays prefill-only, and a request with decode steps keeps at least
    one, so the reduction cannot silently change what is being exercised.
    """
    count = len(tokens)
    if request.prefix_length >= request.n_valid:
        prefix = count
    else:
        prefix = min(request.prefix_length, count - 1) if count >= 2 else count
    return Request(
        request_id=request.request_id,
        token_ids=tuple(tokens),
        prefix_length=max(1, prefix),
        pad_left=request.pad_left if pad_left is None else pad_left,
    )


def _rebuild_case(case: Case, requests: Sequence[Request], note: str) -> Case:
    return Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=list(requests),
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
        provenance=(*case.provenance, note),
    )


@dataclass(frozen=True, slots=True)
class Transform:
    """One reducible dimension of a case, expressed as a list of removable elements."""

    name: str
    elements: tuple[Any, ...]
    build: Callable[[tuple[int, ...]], Case | None]

    def __len__(self) -> int:
        return len(self.elements)


def request_transform(case: Case, target_request_id: str) -> Transform:
    """Removable elements: the non-target requests of a session."""
    removable = [
        index
        for index, request in enumerate(case.requests)
        if request.request_id != target_request_id
    ]

    def build(kept: tuple[int, ...]) -> Case | None:
        # Surviving requests stay in their original order, and the target always survives —
        # keeping its id stable is what lets a signature recorded before reduction still
        # address the same request afterwards.
        survivors = {removable[i] for i in kept}
        requests = [
            request
            for index, request in enumerate(case.requests)
            if request.request_id == target_request_id or index in survivors
        ]
        return _rebuild_case(case, requests, f"removed {len(removable) - len(kept)} request(s)")

    return Transform("remove_requests", tuple(removable), build)


def token_transform(case: Case, request_id: str) -> Transform:
    """Removable elements: the token positions of one request."""
    target = case.request_by_id(request_id)
    positions = tuple(range(target.n_valid))

    def build(kept: tuple[int, ...]) -> Case | None:
        tokens = [target.token_ids[p] for p in sorted(kept)]
        if not tokens:
            return None
        requests = [
            _rebuild_request(target, tokens) if r.request_id == request_id else r
            for r in case.requests
        ]
        return _rebuild_case(
            case, requests, f"{request_id}: {target.n_valid} -> {len(tokens)} tokens"
        )

    return Transform(f"remove_tokens[{request_id}]", positions, build)


def padding_transform(case: Case) -> Transform:
    """Removable elements: the rows that currently carry left padding."""
    padded = tuple(index for index, request in enumerate(case.requests) if request.pad_left > 0)

    def build(kept: tuple[int, ...]) -> Case | None:
        survivors = {padded[i] for i in kept}
        requests = [
            request
            if index in survivors or index not in padded
            else Request(request.request_id, request.token_ids, request.prefix_length, 0)
            for index, request in enumerate(case.requests)
        ]
        return _rebuild_case(case, requests, "reduced padding")

    return Transform("reduce_padding", padded, build)


def token_value_transform(case: Case) -> Transform:
    """Removable elements: the distinct non-canonical token *values* in the case.

    Grouping by value rather than by slot is what makes this operation monotone. Rewriting a
    single slot can *raise* ``token_value_complexity``: canonicalizing one token of
    ``[2, 2, 2]`` gives ``[1, 2, 2]``, which has two distinct values where the original had
    one. Those candidates are always rejected by the strict-decrease rule, so a slot-wise
    search spends most of its query budget proposing changes that cannot be accepted.

    Rewriting *every* occurrence of one value always moves down the order: the distinct-value
    count never increases, and the value sum strictly decreases because the canonical id is
    the smallest legal token.
    """
    values = tuple(
        sorted(
            {
                token
                for request in case.requests
                for token in request.token_ids
                if token != CANONICAL_TOKEN_ID
            }
        )
    )

    def build(kept: tuple[int, ...]) -> Case | None:
        survivors = {values[i] for i in kept}
        requests = [
            Request(
                request.request_id,
                tuple(
                    token
                    if token in survivors or token == CANONICAL_TOKEN_ID
                    else CANONICAL_TOKEN_ID
                    for token in request.token_ids
                ),
                request.prefix_length,
                request.pad_left,
            )
            for request in case.requests
        ]
        return _rebuild_case(case, requests, f"simplified {len(values) - len(kept)} token value(s)")

    return Transform("simplify_tokens", values, build)


def transforms_for(case: Case, target_request_id: str) -> list[Transform]:
    """The declared operations, in the order they are applied."""
    ordered: list[Transform] = []
    if case.execution_mode is ExecutionMode.SESSION and len(case.requests) > 1:
        ordered.append(request_transform(case, target_request_id))
    for request in case.requests:
        if request.n_valid > 1:
            ordered.append(token_transform(case, request.request_id))
    if any(request.pad_left > 0 for request in case.requests):
        ordered.append(padding_transform(case))
    if any(token != CANONICAL_TOKEN_ID for request in case.requests for token in request.token_ids):
        ordered.append(token_value_transform(case))
    return [transform for transform in ordered if len(transform) > 0]


# --- acceptance ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    accepted: bool
    case: Case | None
    reason: str


def evaluate_candidate(
    original: Case,
    candidate: Case | None,
    predicate: FailurePredicate,
    original_category: CaseCategory,
) -> AcceptanceResult:
    """Apply the four acceptance conditions, cheapest first.

    Order matters for cost: structure checks are free, the predicate costs several model
    runs. Rejecting on size or category before ever running the model is most of why the
    query budget goes as far as it does.
    """
    if candidate is None:
        return AcceptanceResult(False, None, "transformation produced no valid case")
    if not CaseSize.of(candidate) < CaseSize.of(original):
        predicate.counters.size_rejections += 1
        return AcceptanceResult(False, None, "not strictly smaller")
    if classify_case(candidate) is not original_category:
        predicate.counters.category_rejections += 1
        return AcceptanceResult(
            False, None, f"would change category to {classify_case(candidate).value}"
        )
    outcome = predicate(candidate)
    if not outcome.preserved:
        return AcceptanceResult(False, None, outcome.detail or outcome.verdict.value)
    return AcceptanceResult(True, candidate, "preserved")


# --- the reducers ------------------------------------------------------------------------------


@dataclass(slots=True)
class ReductionStep:
    operation: str
    strategy: str
    before: CaseSize
    after: CaseSize
    accepted: bool
    note: str
    queries_at_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "strategy": self.strategy,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "accepted": self.accepted,
            "note": self.note,
            "queries_at_step": self.queries_at_step,
        }


@dataclass(slots=True)
class ReductionResult:
    """The reduced case plus everything needed to judge how it was obtained."""

    strategy: str
    original: Case
    reduced: Case
    signature: FailureSignature
    steps: list[ReductionStep]
    counters: PredicateCounters
    budget: ReductionBudget
    budget_exhausted: bool
    minimality: MinimalityStatus
    minimality_note: str
    wall_time_ns: int

    @property
    def original_size(self) -> CaseSize:
        return CaseSize.of(self.original)

    @property
    def reduced_size(self) -> CaseSize:
        return CaseSize.of(self.reduced)

    @property
    def token_reduction_ratio(self) -> float:
        """Original valid tokens divided by reduced valid tokens. 1.0 means no reduction."""
        reduced = self.reduced.total_valid_tokens
        return self.original.total_valid_tokens / reduced if reduced else float("inf")

    @property
    def accepted_steps(self) -> list[ReductionStep]:
        return [step for step in self.steps if step.accepted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "original_case": self.original.to_dict(),
            "reduced_case": self.reduced.to_dict(),
            "signature": self.signature.to_dict(),
            "original_size": self.original_size.to_dict(),
            "reduced_size": self.reduced_size.to_dict(),
            "token_reduction_ratio": self.token_reduction_ratio,
            "request_reduction": (self.original_size.n_requests - self.reduced_size.n_requests),
            "padding_reduction": (
                self.original_size.n_padding_tokens - self.reduced_size.n_padding_tokens
            ),
            "n_steps": len(self.steps),
            "n_accepted_steps": len(self.accepted_steps),
            "steps": [step.to_dict() for step in self.steps],
            "counters": self.counters.to_dict(),
            "budget": self.budget.to_dict(),
            "budget_exhausted": self.budget_exhausted,
            "minimality": self.minimality.value,
            "minimality_note": self.minimality_note,
            "wall_time_s": round(self.wall_time_ns / 1e9, 4),
        }


def _ddmin_elements(
    transform: Transform,
    original: Case,
    predicate: FailurePredicate,
    category: CaseCategory,
    steps: list[ReductionStep],
) -> Case | None:
    """Structure-aware ddmin over one transform's elements.

    The classic algorithm: try removing coarse chunks first, then their complements, then
    halve the granularity. Written out directly rather than pulled from a library, because
    the interesting part is the acceptance rule around it — validity, strict size decrease,
    category preservation, and signature match — and that is what a library version would
    hide.
    """
    kept = tuple(range(len(transform)))
    best: Case | None = None
    granularity = 2

    while len(kept) >= 2:
        chunk_size = max(1, len(kept) // granularity)
        chunks = [kept[i : i + chunk_size] for i in range(0, len(kept), chunk_size)]

        progressed = False
        # Phase 1: can a single chunk carry the failure on its own?
        for chunk in chunks:
            base = best or original
            result = evaluate_candidate(base, transform.build(chunk), predicate, category)
            steps.append(
                ReductionStep(
                    transform.name,
                    "ddmin/subset",
                    CaseSize.of(base),
                    CaseSize.of(result.case) if result.case else CaseSize.of(base),
                    result.accepted,
                    result.reason,
                    predicate.counters.logical_queries,
                )
            )
            if result.accepted and result.case is not None:
                best, kept, granularity, progressed = result.case, chunk, 2, True
                break

        if progressed:
            continue

        # Phase 2: can the failure survive without each chunk?
        for chunk in chunks:
            complement = tuple(e for e in kept if e not in set(chunk))
            if not complement:
                continue
            base = best or original
            result = evaluate_candidate(base, transform.build(complement), predicate, category)
            steps.append(
                ReductionStep(
                    transform.name,
                    "ddmin/complement",
                    CaseSize.of(base),
                    CaseSize.of(result.case) if result.case else CaseSize.of(base),
                    result.accepted,
                    result.reason,
                    predicate.counters.logical_queries,
                )
            )
            if result.accepted and result.case is not None:
                best = result.case
                kept = complement
                granularity = max(granularity - 1, 2)
                progressed = True
                break

        if progressed:
            continue
        if granularity >= len(kept):
            break
        granularity = min(granularity * 2, len(kept))

    return best


def _greedy_elements(
    transform: Transform,
    original: Case,
    predicate: FailurePredicate,
    category: CaseCategory,
    steps: list[ReductionStep],
) -> Case | None:
    """Greedy single-deletion baseline: try dropping each element, one at a time.

    Implemented separately from ddmin on purpose. Sharing machinery between a method and the
    baseline it is measured against is how a baseline quietly inherits the method's
    advantages, and then the comparison measures nothing.
    """
    kept = list(range(len(transform)))
    best: Case | None = None

    for element in list(kept):
        candidate_kept = tuple(e for e in kept if e != element)
        if not candidate_kept:
            continue
        base = best or original
        result = evaluate_candidate(base, transform.build(candidate_kept), predicate, category)
        steps.append(
            ReductionStep(
                transform.name,
                "greedy/single",
                CaseSize.of(base),
                CaseSize.of(result.case) if result.case else CaseSize.of(base),
                result.accepted,
                result.reason,
                predicate.counters.logical_queries,
            )
        )
        if result.accepted and result.case is not None:
            best = result.case
            kept = list(candidate_kept)

    return best


STRATEGIES: dict[str, Callable[..., Case | None]] = {
    "ddmin": _ddmin_elements,
    "greedy": _greedy_elements,
}


def reduce_case(
    case: Case,
    predicate: FailurePredicate,
    *,
    strategy: str = "ddmin",
    budget: ReductionBudget | None = None,
    max_rounds: int = 4,
) -> ReductionResult:
    """Minimize ``case`` while preserving the predicate's failure signature.

    Always returns the best valid failing case found, including when the budget runs out
    mid-search. A timeout must never discard progress or, worse, return something that does
    not actually fail.
    """
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; available: {sorted(STRATEGIES)}")
    search = STRATEGIES[strategy]
    budget = budget or predicate.budget
    started = time.perf_counter_ns()

    category = classify_case(case)
    current = case
    steps: list[ReductionStep] = []
    exhausted = False

    try:
        for _ in range(max_rounds):
            round_start_size = CaseSize.of(current)
            for transform in transforms_for(current, predicate.signature.target_request_id):
                reduced = search(transform, current, predicate, category, steps)
                if reduced is not None and CaseSize.of(reduced) < CaseSize.of(current):
                    current = reduced
            if not CaseSize.of(current) < round_start_size:
                break
    except BudgetExhaustedSignal:
        exhausted = True

    minimality, note = MinimalityStatus.NOT_ESTABLISHED, "minimality was not checked"
    if not exhausted:
        minimality, note = certify_minimality(current, predicate, category)

    return ReductionResult(
        strategy=strategy,
        original=case,
        reduced=current,
        signature=predicate.signature,
        steps=steps,
        counters=predicate.counters,
        budget=budget,
        budget_exhausted=exhausted or predicate.budget_exhausted,
        minimality=minimality,
        minimality_note=note,
        wall_time_ns=time.perf_counter_ns() - started,
    )


def certify_minimality(
    case: Case, predicate: FailurePredicate, category: CaseCategory
) -> tuple[MinimalityStatus, str]:
    """Check every remaining eligible single deletion.

    Only when all of them fail to preserve the failure may the result be called *1-minimal
    with respect to the declared deletion operations*. If the budget runs out partway, the
    honest answer is that minimality was not established — not a weaker claim dressed up as
    a stronger one.
    """
    checked = 0
    try:
        for transform in transforms_for(case, predicate.signature.target_request_id):
            for element in range(len(transform)):
                kept = tuple(e for e in range(len(transform)) if e != element)
                if not kept:
                    continue
                checked += 1
                result = evaluate_candidate(case, transform.build(kept), predicate, category)
                if result.accepted:
                    return (
                        MinimalityStatus.NOT_ESTABLISHED,
                        f"a further single deletion in {transform.name} still preserves the "
                        "failure, so this case is not 1-minimal",
                    )
    except BudgetExhaustedSignal:
        return (
            MinimalityStatus.NOT_ESTABLISHED,
            f"budget ran out after checking {checked} single deletions",
        )
    if checked == 0:
        return (
            MinimalityStatus.ONE_MINIMAL,
            "no eligible single deletion remains; the case is at the floor of the declared "
            "operations",
        )
    return (
        MinimalityStatus.ONE_MINIMAL,
        f"all {checked} remaining single deletions were checked and none preserved the "
        "failure. This is 1-minimal with respect to the declared deletion operations, not a "
        "globally smallest counterexample.",
    )


def signature_from_failure(
    case: Case,
    failing_request_ids: Sequence[str],
    *,
    checkpoint: Any = None,
) -> FailureSignature:
    """Derive the signature to preserve from an observed failure.

    The target is the first failing request in case order, so the choice is deterministic and
    does not depend on set iteration.
    """
    failing = set(failing_request_ids)
    for request in case.requests:
        if request.request_id in failing:
            return FailureSignature(
                Verdict.FAIL, request.request_id, case.execution_mode, checkpoint
            )
    raise ValueError(f"none of {sorted(failing)} is a request of {case.case_id}")


@dataclass(slots=True)
class ReducerComparison:
    """Paired ddmin-vs-greedy results from one identical starting case."""

    case_id: str
    ddmin: ReductionResult
    greedy: ReductionResult
    _unused: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ddmin": self.ddmin.to_dict(),
            "greedy": self.greedy.to_dict(),
            "ddmin_smaller": self.ddmin.reduced_size < self.greedy.reduced_size,
            "greedy_smaller": self.greedy.reduced_size < self.ddmin.reduced_size,
            "tied": self.ddmin.reduced_size.as_tuple() == self.greedy.reduced_size.as_tuple(),
        }


__all__ = [
    "DEFAULT_MAX_QUERIES",
    "DEFAULT_TIME_BUDGET_S",
    "STRATEGIES",
    "AcceptanceResult",
    "BudgetExhaustedSignal",
    "FailurePredicate",
    "MinimalityStatus",
    "PredicateCounters",
    "PredicateOutcome",
    "ReducerComparison",
    "ReductionBudget",
    "ReductionResult",
    "ReductionStep",
    "Transform",
    "certify_minimality",
    "evaluate_candidate",
    "padding_transform",
    "reduce_case",
    "request_transform",
    "signature_from_failure",
    "token_transform",
    "token_value_transform",
    "transforms_for",
]
