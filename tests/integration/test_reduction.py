"""Reduction against the real fixture: validity, preservation, budgets, and minimality."""

from __future__ import annotations

import numpy as np
import pytest
from bench.mutants import MUTANTS, build_mutant_adapter

from evallens.adapters.encoding import validate_case
from evallens.adapters.native import CandidateAdapter, NativeAdapterSpec, ReferenceAdapter
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.generate import classify_case
from evallens.reduce import (
    FailurePredicate,
    MinimalityStatus,
    PredicateCounters,
    ReductionBudget,
    certify_minimality,
    evaluate_candidate,
    reduce_case,
    signature_from_failure,
    transforms_for,
)
from evallens.replay import replay_in_subprocess, stable_comparison
from evallens.types import (
    CANONICAL_TOKEN_ID,
    Case,
    CaseSize,
    ExecutionMode,
    FailureSignature,
    Request,
    TolerancePolicy,
    Verdict,
)

POLICY = TolerancePolicy()
STRATEGIES = ["ddmin", "greedy"]


def _tokens(rng: np.random.Generator, config: ModelConfig, n: int) -> tuple[int, ...]:
    return tuple(int(t) for t in rng.integers(2, config.vocab_size, size=n))


def _case(config: ModelConfig, weights: WeightDict, requests, mode, seed: int = 1) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=weights_sha256(weights),
        requests=requests,
        execution_mode=mode,
        input_seed=seed,
    )


@pytest.fixture
def scenarios(config: ModelConfig, weights: WeightDict):
    """Three failing starting cases, one per execution mode, each with a matching fault."""
    rng = np.random.default_rng(3)
    return {
        "cached": (
            _case(
                config,
                weights,
                [Request("r0", _tokens(rng, config, 40), 5)],
                ExecutionMode.CACHED_DECODE,
            ),
            Behavior(cache_index="write_overwrite_last"),
        ),
        "session": (
            _case(
                config,
                weights,
                [
                    Request("r0", _tokens(rng, config, 20), 4),
                    Request("r1", _tokens(rng, config, 25), 3),
                    Request("r2", _tokens(rng, config, 18), 2),
                ],
                ExecutionMode.SESSION,
            ),
            Behavior(reset="none"),
        ),
        "padded": (
            _case(
                config,
                weights,
                [
                    Request("r0", _tokens(rng, config, 20), 20, pad_left=4),
                    Request("r1", _tokens(rng, config, 30), 30),
                ],
                ExecutionMode.STATELESS_BATCH,
            ),
            Behavior(pad_mask="ignore"),
        ),
    }


def _setup(config, weights, case, behavior, budget=None):
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, behavior)
    outcome = stable_comparison(reference, candidate, case, POLICY)
    assert outcome.verdict is Verdict.FAIL and outcome.stable, "the fixture case must fail first"
    signature = signature_from_failure(case, outcome.representative.failing_request_ids)
    predicate = FailurePredicate(
        reference,
        candidate,
        POLICY,
        signature,
        budget=budget or ReductionBudget(),
        counters=PredicateCounters(),
    )
    return reference, candidate, predicate, signature


# --- the core guarantees -------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_the_reduced_case_is_valid_and_still_fails(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str, name: str
) -> None:
    case, behavior = scenarios[name]
    reference, candidate, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy=strategy)

    validate_case(result.reduced, config)
    outcome = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert outcome.verdict is Verdict.FAIL
    assert outcome.stable is True


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_reduction_strictly_shrinks_the_case(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str, name: str
) -> None:
    case, behavior = scenarios[name]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy=strategy)

    assert result.reduced_size < result.original_size
    assert result.token_reduction_ratio > 1.0
    assert result.reduced.total_valid_tokens < case.total_valid_tokens


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_every_accepted_step_strictly_decreases_size(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str, name: str
) -> None:
    """The guarantee that makes termination structural rather than hopeful."""
    case, behavior = scenarios[name]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy=strategy)
    for step in result.accepted_steps:
        assert step.after.as_tuple() < step.before.as_tuple(), step.operation


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_the_declared_category_is_preserved(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str, name: str
) -> None:
    """A smaller case in a different category is not a reduction of this failure."""
    case, behavior = scenarios[name]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy=strategy)
    assert classify_case(result.reduced) is classify_case(case)
    assert result.reduced.execution_mode is case.execution_mode


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_target_request_survives_session_reduction(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str
) -> None:
    """Deleting earlier requests must never delete the one the failure is about."""
    case, behavior = scenarios["session"]
    _, _, predicate, signature = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy=strategy)

    ids = {r.request_id for r in result.reduced.requests}
    assert signature.target_request_id in ids
    assert len(result.reduced.requests) <= len(case.requests)
    # A between-request reset fault needs a preceding request to leave residue, so the
    # reducer cannot legitimately shrink this to one request.
    assert len(result.reduced.requests) >= 2


def test_reduction_reaches_a_genuinely_small_case(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")
    assert result.reduced.total_valid_tokens <= 4
    assert result.token_reduction_ratio >= 10.0


def test_token_values_are_simplified_toward_the_canonical_token(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")
    tokens = [t for r in result.reduced.requests for t in r.token_ids]
    assert CANONICAL_TOKEN_ID in tokens


# --- acceptance rules ------------------------------------------------------------------------


def test_an_invalid_candidate_is_never_accepted(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    # Smaller and in the same category, so it reaches the validity check rather than being
    # rejected earlier by the cheaper structural rules.
    invalid = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=[Request("r0", (2, config.vocab_size + 10), 1)],
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
    )
    assert classify_case(invalid) is classify_case(case)
    assert CaseSize.of(invalid) < CaseSize.of(case)

    result = evaluate_candidate(case, invalid, predicate, classify_case(case))
    assert not result.accepted
    assert result.reason.startswith("token")
    assert predicate.counters.invalid_rejections == 1


def test_a_non_smaller_candidate_is_rejected_without_running_the_model(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """Cheap checks first: this is most of why the query budget goes as far as it does."""
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    before = predicate.counters.executed_queries

    result = evaluate_candidate(case, case, predicate, classify_case(case))
    assert not result.accepted
    assert "not strictly smaller" in result.reason
    assert predicate.counters.size_rejections == 1
    assert predicate.counters.executed_queries == before


def test_a_category_changing_candidate_is_rejected(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["padded"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    # Strictly smaller, but a single unpadded row is no longer a padded batch. A smaller case
    # in a different category is not a reduction of *this* failure.
    unpadded = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=[Request("r0", case.requests[0].token_ids[:3], 3, 0)],
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
    )
    assert CaseSize.of(unpadded) < CaseSize.of(case)
    assert classify_case(unpadded) is not classify_case(case)

    result = evaluate_candidate(case, unpadded, predicate, classify_case(case))
    assert not result.accepted
    assert "change category" in result.reason
    assert predicate.counters.category_rejections == 1


def test_a_failure_at_a_different_request_does_not_preserve_the_signature(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["session"]
    reference, candidate, _, _ = _setup(config, weights, case, behavior)
    wrong = FailureSignature(Verdict.FAIL, "r0", case.execution_mode)
    predicate = FailurePredicate(reference, candidate, POLICY, wrong, counters=PredicateCounters())

    outcome = predicate(case)
    assert not outcome.preserved
    assert "not at the target request" in outcome.detail


def test_a_passing_case_does_not_preserve_the_failure(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    reference, _, _, signature = _setup(config, weights, case, behavior)
    good = CandidateAdapter(config, weights)
    predicate = FailurePredicate(reference, good, POLICY, signature, counters=PredicateCounters())
    outcome = predicate(case)
    assert not outcome.preserved
    assert outcome.verdict is Verdict.PASS


# --- the predicate cache ----------------------------------------------------------------------


def test_repeating_a_query_hits_the_cache_and_runs_no_model(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)

    first = predicate(case)
    runs_after_first = predicate.counters.model_runs
    second = predicate(case)

    assert first.preserved == second.preserved
    assert second.from_cache is True
    assert predicate.counters.cache_hits == 1
    assert predicate.counters.logical_queries == 2
    assert predicate.counters.executed_queries == 1
    assert predicate.counters.model_runs == runs_after_first


def test_the_cache_key_covers_every_behavioral_dependency(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """A stale verdict from a different policy or adapter would corrupt a whole reduction."""
    case, behavior = scenarios["cached"]
    reference, candidate, predicate, signature = _setup(config, weights, case, behavior)
    baseline = predicate.cache_key(case)

    other_policy = FailurePredicate(
        reference, candidate, TolerancePolicy(atol=1.0, name="loose"), signature
    )
    assert other_policy.cache_key(case) != baseline

    other_candidate = FailurePredicate(
        reference, CandidateAdapter(config, weights, Behavior(norm="large_eps")), POLICY, signature
    )
    assert other_candidate.cache_key(case) != baseline

    other_signature = FailurePredicate(
        reference, candidate, POLICY, FailureSignature(Verdict.FAIL, "zz", case.execution_mode)
    )
    assert other_signature.cache_key(case) != baseline

    other_environment = FailurePredicate(
        reference, candidate, POLICY, signature, environment_id="different"
    )
    assert other_environment.cache_key(case) != baseline


def test_a_case_differing_only_in_provenance_shares_a_cache_entry(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """Provenance is bookkeeping; two cases that execute identically must share a verdict."""
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    annotated = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=case.requests,
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        provenance=("reduced from somewhere",),
    )
    assert predicate.cache_key(annotated) == predicate.cache_key(case)


def test_counters_separate_logical_queries_from_real_work(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")
    counters = result.counters

    assert counters.logical_queries > 0
    assert counters.executed_queries + counters.cache_hits <= counters.logical_queries
    # Each executed query pays for stability replays, each running both adapters.
    assert counters.model_runs >= counters.executed_queries * 2
    assert counters.to_dict()["cache_hit_rate"] == pytest.approx(
        counters.cache_hits / counters.logical_queries
    )


# --- budgets -----------------------------------------------------------------------------------


def test_a_tiny_query_budget_still_returns_a_valid_failing_case(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """Timeout must preserve the best valid failing case, never discard progress."""
    case, behavior = scenarios["cached"]
    reference, candidate, predicate, _ = _setup(
        config, weights, case, behavior, budget=ReductionBudget(max_queries=3)
    )
    result = reduce_case(case, predicate, strategy="ddmin")

    assert result.budget_exhausted is True
    assert result.minimality is MinimalityStatus.NOT_ESTABLISHED
    validate_case(result.reduced, config)
    outcome = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert outcome.verdict is Verdict.FAIL
    assert result.reduced_size <= result.original_size


def test_a_tiny_time_budget_still_returns_a_valid_failing_case(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["session"]
    reference, candidate, predicate, _ = _setup(
        config, weights, case, behavior, budget=ReductionBudget(time_budget_s=0.001)
    )
    result = reduce_case(case, predicate, strategy="ddmin")

    assert result.budget_exhausted is True
    outcome = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert outcome.verdict is Verdict.FAIL


def test_the_query_budget_is_actually_enforced(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["session"]
    _, _, predicate, _ = _setup(
        config, weights, case, behavior, budget=ReductionBudget(max_queries=7)
    )
    result = reduce_case(case, predicate, strategy="greedy")
    assert result.counters.logical_queries <= 7


def test_both_strategies_receive_identical_budgets(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["cached"]
    budget = ReductionBudget(max_queries=64, time_budget_s=30.0)
    results = {}
    for strategy in STRATEGIES:
        _, _, predicate, _ = _setup(config, weights, case, behavior, budget=budget)
        results[strategy] = reduce_case(case, predicate, strategy=strategy, budget=budget)
    assert results["ddmin"].budget.to_dict() == results["greedy"].budget.to_dict()


# --- minimality -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_a_one_minimal_claim_means_no_single_deletion_survives(
    config: ModelConfig, weights: WeightDict, scenarios, name: str
) -> None:
    """Verify the claim independently rather than trusting the label."""
    case, behavior = scenarios[name]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")
    if result.minimality is not MinimalityStatus.ONE_MINIMAL:
        pytest.skip("this scenario did not certify minimality under the budget")

    category = classify_case(result.reduced)
    for transform in transforms_for(result.reduced, result.signature.target_request_id):
        for element in range(len(transform)):
            kept = tuple(e for e in range(len(transform)) if e != element)
            if not kept:
                continue
            check = evaluate_candidate(result.reduced, transform.build(kept), predicate, category)
            assert not check.accepted, f"{transform.name} element {element} was still reducible"


def test_the_minimality_claim_is_scoped_to_the_declared_operations(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """The label and its note must never read as "globally smallest counterexample"."""
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")
    payload = result.to_dict()

    assert payload["minimality"] in {
        "one_minimal_wrt_declared_operations",
        "reduced_minimality_not_established",
    }
    note = payload["minimality_note"]
    if "globally smallest" in note:
        assert "not a globally smallest" in note.replace("\n", " ")


def test_certification_reports_not_established_when_the_budget_runs_out(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["session"]
    _, _, predicate, _ = _setup(
        config, weights, case, behavior, budget=ReductionBudget(max_queries=2)
    )
    predicate.counters.logical_queries = 2  # force the budget to be spent
    status, note = certify_minimality(case, predicate, classify_case(case))
    assert status is MinimalityStatus.NOT_ESTABLISHED
    assert "budget ran out" in note


# --- determinism and paired comparison -------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_reduction_is_deterministic(
    config: ModelConfig, weights: WeightDict, scenarios, strategy: str
) -> None:
    case, behavior = scenarios["cached"]
    reduced = []
    for _ in range(2):
        _, _, predicate, _ = _setup(config, weights, case, behavior)
        reduced.append(reduce_case(case, predicate, strategy=strategy).reduced)
    assert reduced[0].to_dict() == reduced[1].to_dict()


@pytest.mark.parametrize("name", ["cached", "session", "padded"])
def test_ddmin_and_greedy_start_from_the_identical_case(
    config: ModelConfig, weights: WeightDict, scenarios, name: str
) -> None:
    """Paired comparison: same start, same budget, independently implemented searches."""
    case, behavior = scenarios[name]
    results = {}
    for strategy in STRATEGIES:
        _, _, predicate, _ = _setup(config, weights, case, behavior)
        results[strategy] = reduce_case(case, predicate, strategy=strategy)

    assert results["ddmin"].original.case_id == results["greedy"].original.case_id == case.case_id
    assert all(r.reduced_size < r.original_size for r in results.values())


# --- post-reduction verification ---------------------------------------------------------------------


def test_the_reduced_case_reproduces_in_a_fresh_subprocess(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """A hard-timeout, zero-residue re-run of the final answer."""
    case, behavior = scenarios["cached"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    result = reduce_case(case, predicate, strategy="ddmin")

    replay = replay_in_subprocess(
        result.reduced,
        NativeAdapterSpec("reference", config),
        NativeAdapterSpec("candidate", config, behavior),
        POLICY,
    )
    assert replay.harness_ok is True
    assert replay.reproduced_failure is True
    assert replay.verdict is Verdict.FAIL


def test_a_clean_case_does_not_reproduce_a_failure_in_a_subprocess(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    """The negative control for the subprocess path."""
    case, _ = scenarios["cached"]
    replay = replay_in_subprocess(
        case,
        NativeAdapterSpec("reference", config),
        NativeAdapterSpec("candidate", config),
        POLICY,
    )
    assert replay.harness_ok is True
    assert replay.reproduced_failure is False
    assert replay.verdict is Verdict.PASS


def test_a_harness_error_is_not_a_reproduced_failure(config: ModelConfig) -> None:
    """An import error or a crash must never be mistaken for a successful reproduction."""
    broken = Case.create(
        model_config_id="nonexistent-config",
        weights_sha256="0" * 64,
        requests=[Request("r0", (1, 2), 2)],
        execution_mode=ExecutionMode.STATELESS_BATCH,
        input_seed=0,
    )
    replay = replay_in_subprocess(
        broken,
        NativeAdapterSpec("reference", config),
        NativeAdapterSpec("candidate", config, Behavior(causal="off_by_one")),
        POLICY,
    )
    assert replay.reproduced_failure is False
    assert replay.verdict is not Verdict.FAIL


# --- the whole corpus ----------------------------------------------------------------------------------


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.mutant_id)
def test_every_qualified_variant_reduces_to_a_still_failing_case(
    mutant, config: ModelConfig, weights: WeightDict
) -> None:
    """End-to-end over the full corpus, starting from each variant's own trigger."""
    case = mutant.trigger.build(config, weights)
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)

    outcome = stable_comparison(reference, candidate, case, POLICY)
    signature = signature_from_failure(case, outcome.representative.failing_request_ids)
    predicate = FailurePredicate(
        reference, candidate, POLICY, signature, counters=PredicateCounters()
    )
    result = reduce_case(case, predicate, strategy="ddmin")

    validate_case(result.reduced, config)
    assert classify_case(result.reduced) is classify_case(case)
    assert result.reduced_size <= result.original_size
    final = stable_comparison(reference, candidate, result.reduced, POLICY)
    assert final.verdict is Verdict.FAIL, mutant.mutant_id
    assert final.stable is True


def test_serialization_carries_the_full_reduction_history(
    config: ModelConfig, weights: WeightDict, scenarios
) -> None:
    case, behavior = scenarios["session"]
    _, _, predicate, _ = _setup(config, weights, case, behavior)
    payload = reduce_case(case, predicate, strategy="ddmin").to_dict()

    assert payload["strategy"] == "ddmin"
    assert payload["original_case"]["case_id"] == case.case_id
    assert payload["token_reduction_ratio"] > 1.0
    assert payload["n_steps"] >= payload["n_accepted_steps"] > 0
    assert payload["counters"]["logical_queries"] > 0
    assert payload["budget"]["max_queries"] > 0
    assert isinstance(payload["steps"][0]["before"], dict)
    assert CaseSize(**payload["reduced_size"]) == CaseSize.of(
        Case.from_dict(payload["reduced_case"])
    )
