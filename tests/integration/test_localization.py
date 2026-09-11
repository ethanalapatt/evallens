"""Checkpoint alignment and earliest-observed-divergence localization."""

from __future__ import annotations

import numpy as np
import pytest
from bench.mutants import MUTANTS, MutantSpec, build_mutant_adapter

from evallens.adapters.native import CandidateAdapter, CaptureBudget, ReferenceAdapter
from evallens.compare import compare
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.trace import (
    align_checkpoints,
    checkpoint_depth,
    layer_rank,
    localize,
    trace_and_localize,
    traversal_key,
)
from evallens.types import (
    Case,
    CheckpointAddress,
    CheckpointKind,
    ExecutionMode,
    ExecutionResult,
    Request,
    TolerancePolicy,
    Verdict,
)

POLICY = TolerancePolicy()


def _case(config: ModelConfig, weights: WeightDict, requests, mode) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=weights_sha256(weights),
        requests=requests,
        execution_mode=mode,
        input_seed=0,
    )


@pytest.fixture
def stateless_case(config: ModelConfig, weights: WeightDict) -> Case:
    return _case(
        config, weights, [Request("r0", (11, 12, 13, 14, 15), 5)], ExecutionMode.STATELESS_BATCH
    )


# --- the traversal order ------------------------------------------------------------------


def test_layer_rank_orders_the_network() -> None:
    assert layer_rank("embed") < layer_rank("block0") < layer_rank("block1") < layer_rank("final")
    assert layer_rank("block2") < layer_rank("block10")


def test_unknown_layer_names_are_rejected_rather_than_sorted_arbitrarily() -> None:
    with pytest.raises(ValueError, match="unknown layer name"):
        layer_rank("mystery")


def test_depth_follows_block_dataflow() -> None:
    def depth(layer: str, kind: CheckpointKind) -> int:
        return checkpoint_depth(CheckpointAddress("r", layer, 0, kind))

    assert (
        depth("block0", CheckpointKind.ATTN_OUT)
        < depth("block0", CheckpointKind.MLP_OUT)
        < depth("block0", CheckpointKind.BLOCK_OUT)
        < depth("block1", CheckpointKind.ATTN_OUT)
    )
    assert depth("embed", CheckpointKind.EMBEDDING) < depth("block0", CheckpointKind.ATTN_OUT)
    assert depth("block1", CheckpointKind.BLOCK_OUT) < depth("final", CheckpointKind.FINAL_NORM)
    assert depth("final", CheckpointKind.FINAL_NORM) < depth("final", CheckpointKind.LOGITS)


def test_traversal_is_request_then_depth_then_position() -> None:
    order = {"a": 0, "b": 1}

    def key(request: str, layer: str, position: int, kind: CheckpointKind):
        return traversal_key(CheckpointAddress(request, layer, position, kind), order)

    # Request dominates everything.
    assert key("a", "final", 99, CheckpointKind.LOGITS) < key(
        "b", "embed", 0, CheckpointKind.EMBEDDING
    )
    # Depth dominates position.
    assert key("a", "block0", 99, CheckpointKind.ATTN_OUT) < key(
        "a", "block1", 0, CheckpointKind.ATTN_OUT
    )
    # Position breaks ties within a depth.
    assert key("a", "block0", 1, CheckpointKind.ATTN_OUT) < key(
        "a", "block0", 2, CheckpointKind.ATTN_OUT
    )


# --- alignment ---------------------------------------------------------------------------


def test_full_prefix_and_cached_passes_align_completely(
    config: ModelConfig, weights: WeightDict
) -> None:
    """Different call schedules, identical checkpoint addresses."""
    case = _case(config, weights, [Request("r0", (5, 6, 7, 8), 2)], ExecutionMode.CACHED_DECODE)
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights)
    reference_result = reference.run(case, capture=True)
    candidate_result = candidate.run(case, capture=True)

    assert reference_result.meta["model_calls"] == 1
    assert candidate_result.meta["model_calls"] == 3

    alignment = align_checkpoints(reference_result, candidate_result)
    assert alignment.fully_aligned
    assert alignment.aligned_fraction == 1.0
    assert alignment.reference_only == ()
    assert alignment.candidate_only == ()
    assert len(alignment.matched) == 4 * (1 + 2 * 3 + 2)


def test_alignment_reports_unmatched_checkpoints_rather_than_dropping_them(
    config: ModelConfig, weights: WeightDict
) -> None:
    case = _case(config, weights, [Request("r0", (5, 6, 7, 8), 2)], ExecutionMode.CACHED_DECODE)
    reference_result = ReferenceAdapter(config, weights).run(case, capture=True)
    truncated = CandidateAdapter(
        config, weights, capture_budget=CaptureBudget(max_checkpoints=5)
    ).run(case, capture=True)

    alignment = align_checkpoints(reference_result, truncated)
    assert not alignment.fully_aligned
    assert len(alignment.reference_only) > 0
    assert alignment.candidate_only == ()
    assert 0 < alignment.aligned_fraction < 1.0
    assert alignment.to_dict()["n_reference_only"] == len(alignment.reference_only)


def test_alignment_is_by_address_not_by_call_order(
    config: ModelConfig, weights: WeightDict
) -> None:
    """Alignment must survive the candidate recording in a completely different sequence."""
    case = _case(config, weights, [Request("r0", (5, 6, 7), 1)], ExecutionMode.CACHED_DECODE)
    reference_result = ReferenceAdapter(config, weights).run(case, capture=True)
    candidate_result = CandidateAdapter(config, weights).run(case, capture=True)

    reversed_candidate = ExecutionResult(
        adapter_id=candidate_result.adapter_id,
        case_id=candidate_result.case_id,
        outputs=candidate_result.outputs,
        checkpoints=tuple(reversed(candidate_result.checkpoints)),
        capture_enabled=True,
    )
    assert align_checkpoints(reference_result, reversed_candidate).fully_aligned
    result = localize(reference_result, reversed_candidate, POLICY)
    assert result.available
    assert result.earliest_observed is None  # identical values, just reordered records


# --- the crafted reconvergence case ---------------------------------------------------------


def test_a_crafted_intermediate_divergence_reconverges_before_the_output(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """The case that forbids binary search.

    The probe genuinely perturbs the residual stream while the embedding checkpoint is
    recorded and genuinely removes it before anything consumes it. So an intermediate
    diverges, everything after it agrees, and the output is within tolerance. A bisection for
    "the first mismatch" would be searching a non-monotone predicate.
    """
    reference = ReferenceAdapter(config, weights)
    probe = CandidateAdapter(config, weights, Behavior(reconverge_probe=0.05))

    output = compare(reference.run(stateless_case), probe.run(stateless_case), POLICY)
    assert output.verdict is Verdict.PASS, "the probe must reconverge at the output"

    result = trace_and_localize(reference, probe, stateless_case, POLICY)
    assert result.available
    assert result.reconverged is True
    assert result.earliest_observed is not None
    assert result.earliest_observed.kind is CheckpointKind.EMBEDDING
    assert result.divergent_layers == ("embed",)
    assert len(result.divergent) == 5  # one per token position
    assert "reconverges later" in result.summary()


def test_the_reconvergence_probe_is_not_an_injected_fault() -> None:
    """It is a diagnostic. It must never be counted as a detection or a fault variant."""
    probe = Behavior(reconverge_probe=0.05)
    assert not probe.is_reference
    assert not probe.is_injected_fault
    assert all(mutant.behavior.is_injected_fault for mutant in MUTANTS)
    assert all(mutant.behavior.reconverge_probe == 0.0 for mutant in MUTANTS)


def test_a_real_fault_also_exhibits_non_monotone_divergence(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """Non-monotonicity is not an artifact of the probe; the corpus shows it too.

    `leak_last` makes the final key column always visible. At the *final* position that
    column is legitimately visible, so that position agrees at every layer while every
    earlier position disagrees.
    """
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="leak_last"))
    result = trace_and_localize(reference, candidate, stateless_case, POLICY)

    assert result.available
    assert result.reconverged is True

    last_position = stateless_case.requests[0].n_valid - 1
    attn = [
        c
        for c in result.comparisons
        if c.address.layer_name == "block0" and c.address.kind is CheckpointKind.ATTN_OUT
    ]
    assert [c.diverged for c in attn][:last_position] == [True] * last_position
    assert attn[last_position].diverged is False


# --- localizing the real corpus ---------------------------------------------------------------


@pytest.mark.parametrize(
    "mutant",
    [m for m in MUTANTS if m.trigger.execution_mode is ExecutionMode.STATELESS_BATCH],
    ids=lambda m: m.mutant_id,
)
def test_stateless_faults_localize_to_an_exposed_checkpoint(
    mutant: MutantSpec, config: ModelConfig, weights: WeightDict
) -> None:
    case = mutant.trigger.build(config, weights)
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)

    result = trace_and_localize(reference, candidate, case, POLICY)
    assert result.available, result.reason
    assert result.alignment.fully_aligned
    assert result.earliest_observed is not None, mutant.mutant_id
    # Every stateless fault in this corpus acts at or after the first attention, never in the
    # embedding, which the candidate computes identically.
    assert result.earliest_observed.layer_name == "block0"
    assert result.earliest_observed.kind is CheckpointKind.ATTN_OUT


@pytest.mark.parametrize(
    "mutant",
    [m for m in MUTANTS if m.trigger.execution_mode is not ExecutionMode.STATELESS_BATCH],
    ids=lambda m: m.mutant_id,
)
def test_cached_and_session_faults_localize(
    mutant: MutantSpec, config: ModelConfig, weights: WeightDict
) -> None:
    case = mutant.trigger.build(config, weights)
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)

    result = trace_and_localize(reference, candidate, case, POLICY)
    assert result.available, result.reason
    assert result.earliest_observed is not None, mutant.mutant_id
    assert result.divergent, mutant.mutant_id
    assert result.earliest_observed == result.divergent[0].address


def test_attention_scaling_cannot_show_at_the_first_position(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """A softmax over a single key is 1.0 whatever the scale, so position 0 must agree.

    This is the sharpest check that localization reports where divergence is genuinely
    observable rather than where the fault was injected.
    """
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(attn_scale="no_sqrt"))
    result = trace_and_localize(reference, candidate, stateless_case, POLICY)

    assert result.earliest_observed is not None
    assert result.earliest_observed.layer_name == "block0"
    assert result.earliest_observed.kind is CheckpointKind.ATTN_OUT
    assert result.earliest_observed.token_position == 1

    first = next(
        c
        for c in result.comparisons
        if c.address.layer_name == "block0"
        and c.address.kind is CheckpointKind.ATTN_OUT
        and c.address.token_position == 0
    )
    assert not first.diverged


def test_a_session_fault_localizes_to_the_affected_request(
    config: ModelConfig, weights: WeightDict
) -> None:
    """A missing between-request reset cannot affect the first request."""
    case = _case(
        config,
        weights,
        [Request("q0", (11, 12, 13, 14), 2), Request("q1", (15, 16, 17), 1)],
        ExecutionMode.SESSION,
    )
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(reset="none"))
    result = trace_and_localize(reference, candidate, case, POLICY)

    assert result.available
    assert result.earliest_observed is not None
    assert result.earliest_observed.request_id == "q1"
    assert all(c.address.request_id == "q1" for c in result.divergent)


# --- unavailable localization -------------------------------------------------------------------


def test_localization_is_unavailable_without_a_traced_pass(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="off_by_one"))
    result = localize(reference.run(stateless_case), candidate.run(stateless_case), POLICY)
    assert not result.available
    assert "capture was not enabled" in result.reason
    assert result.earliest_observed is None
    assert "localization unavailable" in result.summary()


def test_localization_is_unavailable_when_nothing_aligns(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """The output-level failure must stand on its own rather than being replaced by a guess."""
    reference_result = ReferenceAdapter(config, weights).run(stateless_case, capture=True)
    disjoint = ExecutionResult(
        adapter_id="other",
        case_id=stateless_case.case_id,
        outputs=reference_result.outputs,
        checkpoints=(),
        capture_enabled=True,
    )
    result = localize(reference_result, disjoint, POLICY)
    assert not result.available
    assert "no checkpoints align" in result.reason
    assert result.to_dict()["earliest_observed"] is None


def test_localization_is_unavailable_when_values_were_dropped(
    config: ModelConfig, weights: WeightDict
) -> None:
    """Summaries are not element-wise evidence and must not be compared as if they were."""
    case = _case(config, weights, [Request("r0", (1, 2, 3), 3)], ExecutionMode.STATELESS_BATCH)
    budget = CaptureBudget(max_values_per_checkpoint=1)
    reference = ReferenceAdapter(config, weights, capture_budget=budget)
    candidate = CandidateAdapter(
        config, weights, Behavior(causal="off_by_one"), capture_budget=budget
    )
    result = trace_and_localize(reference, candidate, case, POLICY)

    assert not result.available
    assert "values were dropped" in result.reason
    assert result.comparisons
    assert all(not c.values_available for c in result.comparisons)


def test_identical_implementations_localize_with_no_divergence(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights)
    result = trace_and_localize(reference, candidate, stateless_case, POLICY)

    assert result.available
    assert result.earliest_observed is None
    assert result.divergent == ()
    assert result.reconverged is False
    assert "observable only at the output" in result.summary()


# --- capture hygiene ------------------------------------------------------------------------------


def test_the_traced_pass_does_not_leak_between_runs(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="off_by_one"))
    first = trace_and_localize(reference, candidate, stateless_case, POLICY)
    second = trace_and_localize(reference, candidate, stateless_case, POLICY)

    assert len(first.comparisons) == len(second.comparisons)
    assert first.earliest_observed == second.earliest_observed
    assert [c.diverged for c in first.comparisons] == [c.diverged for c in second.comparisons]


def test_the_traced_pass_is_separate_from_the_detection_path(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """Detection must not pay capture's cost, and capture must not change the outputs."""
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="off_by_one"))

    untraced = candidate.run(stateless_case, capture=False)
    traced = candidate.run(stateless_case, capture=True)
    np.testing.assert_array_equal(untraced.outputs["r0"], traced.outputs["r0"])
    assert untraced.checkpoints == ()
    assert len(traced.checkpoints) > 0

    result = trace_and_localize(reference, candidate, stateless_case, POLICY)
    assert result.capture_wall_time_ns > 0


def test_localization_serializes_with_its_interpretation_caveat(
    config: ModelConfig, weights: WeightDict, stateless_case: Case
) -> None:
    """The caveat travels with the data so a viewer cannot present this as a root cause."""
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="off_by_one"))
    payload = trace_and_localize(reference, candidate, stateless_case, POLICY).to_dict()

    assert payload["available"] is True
    assert payload["earliest_observed_str"].startswith("r0/block0/")
    assert payload["n_divergent"] > 0
    assert "not proof of root cause" in payload["interpretation"]
    assert payload["policy"]["policy_id"] == POLICY.policy_id


def test_serialization_truncates_long_comparison_lists(
    config: ModelConfig, weights: WeightDict
) -> None:
    case = _case(
        config, weights, [Request("r0", tuple(range(1, 25)), 24)], ExecutionMode.STATELESS_BATCH
    )
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, Behavior(causal="off_by_one"))
    result = trace_and_localize(reference, candidate, case, POLICY)
    payload = result.to_dict(max_comparisons=10)

    assert len(payload["comparisons"]) == 10
    assert payload["comparisons_truncated"] is True
    assert payload["n_compared"] == len(result.comparisons) > 10
