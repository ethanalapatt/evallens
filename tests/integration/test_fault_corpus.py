"""Qualification of the injected-fault corpus and the known-good controls.

Every fault here is deliberate. These tests establish that each declared variant actually
does what it claims, and that every control a correct implementation should pass does pass.
"""

from __future__ import annotations

import pytest
from bench.controls import BENIGN_PERTURBATION, CONTROL_BY_ID, CONTROLS
from bench.mutants import (
    FAMILIES,
    MUTANT_BY_ID,
    MUTANTS,
    MutantSpec,
    qualify_mutant,
)
from bench.seeds import SEED_SETS, assert_disjoint

from evallens.adapters.encoding import validate_case
from evallens.adapters.native import ReferenceAdapter
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict
from evallens.replay import run_comparison, stable_comparison
from evallens.types import TolerancePolicy, Verdict

POLICY = TolerancePolicy()


# --- corpus shape ---------------------------------------------------------------------------


def test_eight_families_with_two_variants_each() -> None:
    assert len(FAMILIES) == 8
    assert len(MUTANTS) == 16
    assert len({m.mutant_id for m in MUTANTS}) == 16
    by_family: dict[str, list[MutantSpec]] = {}
    for mutant in MUTANTS:
        by_family.setdefault(mutant.family_key, []).append(mutant)
    assert set(by_family) == {family.key for family in FAMILIES}
    for key, variants in by_family.items():
        assert len(variants) == 2, key
        assert len({v.variant for v in variants}) == 2, key


def test_every_mutant_is_labeled_as_an_injected_fault() -> None:
    """The label is not decoration: reports and the viewer read it."""
    for mutant in MUTANTS:
        assert mutant.injected is True
        assert mutant.to_dict()["injected_fault"] is True


def test_every_mutant_has_a_non_default_behavior() -> None:
    """A mutant whose behavior is the reference would be a control masquerading as a fault."""
    for mutant in MUTANTS:
        assert not mutant.behavior.is_reference, mutant.mutant_id


def test_mutant_behaviors_are_distinct() -> None:
    behaviors = [mutant.behavior for mutant in MUTANTS]
    assert len(set(behaviors)) == len(behaviors)


def test_every_mutant_has_a_rationale_for_its_trigger() -> None:
    for mutant in MUTANTS:
        assert len(mutant.trigger.rationale) > 40, mutant.mutant_id
        assert len(mutant.description) > 20, mutant.mutant_id


# --- qualification ----------------------------------------------------------------------------


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.mutant_id)
def test_every_declared_variant_qualifies_on_its_own_trigger(
    mutant: MutantSpec, config: ModelConfig, weights: WeightDict
) -> None:
    """Each variant must produce a *stable* FAIL on an independently written trigger.

    A crash would be ERROR and would not qualify: a mutant that only crashes exercises the
    exception path, not the detector.
    """
    result = qualify_mutant(mutant, config, weights, POLICY)
    assert result.qualified, f"{mutant.mutant_id}: {result.verdict.value} — {result.detail}"
    assert result.verdict is Verdict.FAIL
    assert result.stable is True
    assert result.max_abs_err > POLICY.atol


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.mutant_id)
def test_every_trigger_case_is_itself_valid(
    mutant: MutantSpec, config: ModelConfig, weights: WeightDict
) -> None:
    """A trigger that failed validation would score as INVALID, not as a detected fault."""
    validate_case(mutant.trigger.build(config, weights), config)


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.mutant_id)
def test_the_reference_never_fails_a_trigger_case(
    mutant: MutantSpec, config: ModelConfig, weights: WeightDict
) -> None:
    """The trigger must be a valid input, not an input the reference also chokes on."""
    case = mutant.trigger.build(config, weights)
    reference = ReferenceAdapter(config, weights)
    outputs = reference.run(case).outputs
    assert set(outputs) == {r.request_id for r in case.requests}


def test_a_mutants_fault_is_silent_rather_than_a_crash(
    config: ModelConfig, weights: WeightDict
) -> None:
    """Shape-valid silent faults are the interesting case; crashes are tracked separately."""
    for mutant in MUTANTS:
        case = mutant.trigger.build(config, weights)
        reference = ReferenceAdapter(config, weights)
        from bench.mutants import build_mutant_adapter

        candidate = build_mutant_adapter(mutant, config, weights)
        result = run_comparison(reference, candidate, case, POLICY)
        assert result.verdict is not Verdict.ERROR, f"{mutant.mutant_id} crashed: {result.detail}"


# --- controls -----------------------------------------------------------------------------------


@pytest.mark.parametrize("control", CONTROLS, ids=lambda c: c.control_id)
def test_every_control_is_a_stable_pass(control, config: ModelConfig, weights: WeightDict) -> None:
    """A stable FAIL on any control is a false positive, not a detection."""
    case = control.build_case(config, weights)
    reference, candidate = control.build_adapters(config, weights)
    outcome = stable_comparison(reference, candidate, case, POLICY)
    assert outcome.verdict is Verdict.PASS, f"{control.control_id}: {outcome.representative.detail}"
    assert outcome.stable is True


def test_controls_cover_more_than_exact_equality(config: ModelConfig, weights: WeightDict) -> None:
    """At least one control must pass on the tolerance band rather than on equality."""
    benign = CONTROL_BY_ID["benign_subtolerance_perturbation"]
    case = benign.build_case(config, weights)
    reference, candidate = benign.build_adapters(config, weights)
    result = run_comparison(reference, candidate, case, POLICY)
    assert result.verdict is Verdict.PASS
    assert result.max_abs_err > 0.0, "the benign perturbation must be real, not a no-op"
    assert result.max_abs_err < POLICY.atol


def test_the_benign_perturbation_is_actually_below_tolerance(
    config: ModelConfig, weights: WeightDict
) -> None:
    """Guard against the perturbation silently drifting into detectable territory."""
    assert POLICY.rtol > BENIGN_PERTURBATION


def test_every_control_is_labeled_as_not_injected() -> None:
    for control in CONTROLS:
        assert control.to_dict()["injected_fault"] is False
        assert control.behavior.is_reference or control.behavior.perturb_scale > 0


def test_controls_cover_the_declared_comparison_kinds() -> None:
    ids = {control.control_id for control in CONTROLS}
    assert {
        "identical_stateless",
        "cached_vs_full_prefix",
        "padded_alignment",
        "batch_permutation",
        "fresh_request_isolation",
        "benign_subtolerance_perturbation",
    } <= ids


@pytest.mark.parametrize("control", CONTROLS, ids=lambda c: c.control_id)
def test_every_control_case_is_valid(control, config: ModelConfig, weights: WeightDict) -> None:
    validate_case(control.build_case(config, weights), config)


# --- seeds ---------------------------------------------------------------------------------------


def test_seed_sets_are_disjoint_and_nonempty() -> None:
    assert_disjoint()
    assert set(SEED_SETS) == {"calibration", "development", "evaluation"}
    for name, seeds in SEED_SETS.items():
        assert len(seeds) >= 16, name


def test_calibration_and_evaluation_share_no_seed() -> None:
    """The separation that makes 'frozen before held-out evaluation' mean anything."""
    assert not set(SEED_SETS["calibration"]) & set(SEED_SETS["evaluation"])


# --- the switchboard default ----------------------------------------------------------------------


def test_the_default_behavior_is_the_reference() -> None:
    """If Behavior() were ever faulty, every control in the project would be meaningless."""
    assert Behavior().is_reference
    assert Behavior().describe() == "reference (no injected fault)"
    for value in Behavior().to_dict().values():
        assert value in ("correct", 0.0)


def test_each_declared_knob_changes_behavior_identity() -> None:
    for mutant in MUTANTS:
        assert mutant.behavior != Behavior()
        assert "injected:" in mutant.behavior.describe()


def test_mutant_lookup_is_complete() -> None:
    assert set(MUTANT_BY_ID) == {mutant.mutant_id for mutant in MUTANTS}
