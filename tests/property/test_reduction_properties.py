"""Reduction properties over many starting cases.

The integration tests use three fixed scenarios. These check the invariants that must hold
for *any* failing case the search is handed: it terminates, it never returns something that
does not fail, and it never returns something invalid.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evallens.adapters.encoding import validate_case
from evallens.adapters.native import CandidateAdapter, ReferenceAdapter
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import UNIT_FIXTURE, make_weights, weights_sha256
from evallens.generate import classify_case
from evallens.reduce import (
    FailurePredicate,
    PredicateCounters,
    ReductionBudget,
    reduce_case,
    signature_from_failure,
    transforms_for,
)
from evallens.replay import stable_comparison
from evallens.types import Case, CaseSize, ExecutionMode, Request, TolerancePolicy, Verdict

_WEIGHTS = make_weights(UNIT_FIXTURE)
_SHA = weights_sha256(_WEIGHTS)
POLICY = TolerancePolicy()

# One fault per mode, chosen so the generated starting case reliably fails in that mode.
MODE_FAULTS = {
    ExecutionMode.STATELESS_BATCH: Behavior(causal="off_by_one"),
    ExecutionMode.CACHED_DECODE: Behavior(cache_index="write_overwrite_last"),
    ExecutionMode.SESSION: Behavior(reset="none"),
}

SETTINGS = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

TOKENS = st.lists(st.integers(min_value=2, max_value=96), min_size=3, max_size=20)


def _reduce(case: Case, behavior: Behavior, strategy: str, budget: ReductionBudget | None = None):
    reference = ReferenceAdapter(UNIT_FIXTURE, _WEIGHTS)
    candidate = CandidateAdapter(UNIT_FIXTURE, _WEIGHTS, behavior)
    outcome = stable_comparison(reference, candidate, case, POLICY)
    if outcome.verdict is not Verdict.FAIL or not outcome.stable:
        return None
    signature = signature_from_failure(case, outcome.representative.failing_request_ids)
    predicate = FailurePredicate(
        reference,
        candidate,
        POLICY,
        signature,
        budget=budget or ReductionBudget(max_queries=96, time_budget_s=20.0),
        counters=PredicateCounters(),
    )
    return reference, candidate, reduce_case(case, predicate, strategy=strategy, budget=budget)


@given(TOKENS, st.sampled_from(["ddmin", "greedy"]))
@SETTINGS
def test_a_reduced_cached_case_is_always_valid_and_still_failing(
    tokens: list[int], strategy: str
) -> None:
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=max(1, len(tokens) // 3))],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=0,
    )
    outcome = _reduce(case, MODE_FAULTS[ExecutionMode.CACHED_DECODE], strategy)
    if outcome is None:
        return
    reference, candidate, result = outcome

    validate_case(result.reduced, UNIT_FIXTURE)
    assert result.reduced_size <= result.original_size
    assert classify_case(result.reduced) is classify_case(case)
    final = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert final.verdict is Verdict.FAIL
    assert final.stable is True


@given(TOKENS, st.sampled_from(["ddmin", "greedy"]))
@SETTINGS
def test_a_reduced_stateless_case_is_always_valid_and_still_failing(
    tokens: list[int], strategy: str
) -> None:
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=len(tokens))],
        execution_mode=ExecutionMode.STATELESS_BATCH,
        input_seed=0,
    )
    outcome = _reduce(case, MODE_FAULTS[ExecutionMode.STATELESS_BATCH], strategy)
    if outcome is None:
        return
    reference, candidate, result = outcome

    validate_case(result.reduced, UNIT_FIXTURE)
    final = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert final.verdict is Verdict.FAIL


@given(TOKENS)
@SETTINGS
def test_reduction_always_terminates_inside_its_budget(tokens: list[int]) -> None:
    """Termination is structural: every acceptance moves down a well-founded order."""
    budget = ReductionBudget(max_queries=64, time_budget_s=20.0)
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=1)],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=0,
    )
    outcome = _reduce(case, MODE_FAULTS[ExecutionMode.CACHED_DECODE], "ddmin", budget)
    if outcome is None:
        return
    _, _, result = outcome
    assert result.counters.logical_queries <= budget.max_queries
    assert result.wall_time_ns / 1e9 <= budget.time_budget_s * 3


@given(TOKENS)
@SETTINGS
def test_accepted_steps_form_a_strictly_decreasing_chain(tokens: list[int]) -> None:
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=max(1, len(tokens) // 2))],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=0,
    )
    outcome = _reduce(case, MODE_FAULTS[ExecutionMode.CACHED_DECODE], "ddmin")
    if outcome is None:
        return
    _, _, result = outcome
    for step in result.accepted_steps:
        assert step.after.as_tuple() < step.before.as_tuple()


@given(TOKENS)
@SETTINGS
def test_reduction_is_deterministic_for_a_given_start(tokens: list[int]) -> None:
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=2 if len(tokens) > 2 else 1)],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=0,
    )
    first = _reduce(case, MODE_FAULTS[ExecutionMode.CACHED_DECODE], "ddmin")
    second = _reduce(case, MODE_FAULTS[ExecutionMode.CACHED_DECODE], "ddmin")
    if first is None or second is None:
        return
    assert first[2].reduced.to_dict() == second[2].reduced.to_dict()


@given(TOKENS)
@SETTINGS
def test_every_transform_produces_valid_or_rejected_cases_never_silently_broken(
    tokens: list[int],
) -> None:
    """Transform outputs must be either valid or explicitly ``None``, never malformed."""
    case = Case.create(
        model_config_id=UNIT_FIXTURE.config_id,
        weights_sha256=_SHA,
        requests=[Request("r0", tuple(tokens), prefix_length=max(1, len(tokens) // 2))],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=0,
    )
    for transform in transforms_for(case, "r0"):
        for keep in (tuple(range(len(transform))), tuple(range(len(transform)))[:1]):
            built = transform.build(keep)
            if built is None:
                continue
            validate_case(built, UNIT_FIXTURE)
            assert CaseSize.of(built) <= CaseSize.of(case)
            assert all(r.n_valid >= 1 for r in built.requests)
            assert all(1 <= r.prefix_length <= r.n_valid for r in built.requests)
