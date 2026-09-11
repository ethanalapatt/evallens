"""Generator validity, determinism, category coverage, and diversity.

Both generators are held to the same standard on the same declared space. A generator that
emitted invalid inputs would score "detections" that are really crashes, and a generator that
could not reach a category could never trigger the fault families that live there.
"""

from __future__ import annotations

import pytest

from evallens.adapters.encoding import validate_case
from evallens.fixtures.config import UNIT_FIXTURE, ModelConfig, WeightDict, weights_sha256
from evallens.generate import (
    ALL_CATEGORIES,
    GENERATORS,
    BoundaryAwareGenerator,
    CaseCategory,
    CaseGenerator,
    GeneratorCapability,
    UniformValidGenerator,
    build_generator,
    classify_case,
)
from evallens.types import PAD_TOKEN_ID, Case, ExecutionMode

GENERATOR_CLASSES = [UniformValidGenerator, BoundaryAwareGenerator]
IDS = [cls.name for cls in GENERATOR_CLASSES]


@pytest.fixture(scope="module")
def capability() -> GeneratorCapability:
    return GeneratorCapability.for_fixture(UNIT_FIXTURE, "f" * 64, max_tokens=32)


def _make(cls: type[CaseGenerator], capability: GeneratorCapability) -> CaseGenerator:
    return cls(capability)


# --- validity ------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_every_generated_case_is_valid(
    cls: type[CaseGenerator], capability: GeneratorCapability, config: ModelConfig
) -> None:
    for case in _make(cls, capability).generate(120, seed=7):
        validate_case(case, config)


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_generated_tokens_are_never_the_padding_id(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    for case in _make(cls, capability).generate(120, seed=8):
        for request in case.requests:
            assert PAD_TOKEN_ID not in request.token_ids
            assert all(1 <= t < capability.vocab_size for t in request.token_ids)


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_generated_cases_respect_the_declared_limits(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    for case in _make(cls, capability).generate(120, seed=9):
        assert case.batch_width <= UNIT_FIXTURE.max_position
        for request in case.requests:
            assert 1 <= request.n_valid <= capability.max_tokens_per_request
            assert 1 <= request.prefix_length <= request.n_valid
            assert 0 <= request.pad_left <= capability.max_pad_left
        if case.execution_mode is ExecutionMode.STATELESS_BATCH:
            assert len(case.requests) <= capability.max_batch_rows
        elif case.execution_mode is ExecutionMode.CACHED_DECODE:
            assert len(case.requests) == 1
        else:
            assert len(case.requests) <= capability.max_session_requests


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_the_declared_category_matches_the_actual_structure(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    """A case labeled `session` that is really a single cached request would corrupt scoring."""
    for case in _make(cls, capability).generate(120, seed=10):
        assert case.category == classify_case(case).value


# --- determinism ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_generation_is_deterministic_in_count_and_seed(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    """Determinism is what makes a recorded benchmark trial replayable."""
    first = _make(cls, capability).generate(40, seed=11)
    second = _make(cls, capability).generate(40, seed=11)
    assert [c.case_id for c in first] == [c.case_id for c in second]
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_different_seeds_produce_different_cases(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    first = {c.case_id for c in _make(cls, capability).generate(40, seed=12)}
    second = {c.case_id for c in _make(cls, capability).generate(40, seed=13)}
    assert len(first & second) < len(first) // 2


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_a_prefix_of_a_longer_run_is_the_same_run(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    """Generating 10 then 40 must agree on the first 10, so budgets are comparable."""
    short = _make(cls, capability).generate(10, seed=14)
    long = _make(cls, capability).generate(40, seed=14)
    assert [c.case_id for c in short] == [c.case_id for c in long[:10]]


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_case_ids_are_content_hashes_of_the_generated_case(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    for case in _make(cls, capability).generate(30, seed=15):
        assert case.case_id == f"case_{case.content_hash()[:16]}"


# --- coverage ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_every_declared_category_is_emitted(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    seen = {classify_case(c) for c in _make(cls, capability).generate(60, seed=16)}
    assert seen == set(ALL_CATEGORIES)


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_categories_are_emitted_in_equal_proportion(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    """Both generators get the same budget spread, so detection differences reflect choices."""
    cases = _make(cls, capability).generate(60, seed=17)
    counts = dict.fromkeys(ALL_CATEGORIES, 0)
    for case in cases:
        counts[classify_case(case)] += 1
    assert set(counts.values()) == {60 // len(ALL_CATEGORIES)}


def test_both_generators_produce_the_same_category_sequence(
    capability: GeneratorCapability,
) -> None:
    uniform = [classify_case(c) for c in UniformValidGenerator(capability).generate(30, seed=18)]
    boundary = [classify_case(c) for c in BoundaryAwareGenerator(capability).generate(30, seed=18)]
    assert uniform == boundary


@pytest.mark.parametrize("cls", GENERATOR_CLASSES, ids=IDS)
def test_generated_cases_are_meaningfully_diverse(
    cls: type[CaseGenerator], capability: GeneratorCapability
) -> None:
    """Distinct structures, not sixty copies of one shape."""
    cases = _make(cls, capability).generate(60, seed=19)
    assert len({c.case_id for c in cases}) >= 55
    assert len({c.total_valid_tokens for c in cases}) >= 8
    assert len({len(c.requests) for c in cases}) >= 3


# --- the boundary generator's distinguishing behavior -----------------------------------------


def test_the_boundary_generator_reaches_both_length_extremes(
    capability: GeneratorCapability,
) -> None:
    lengths = {
        request.n_valid
        for case in BoundaryAwareGenerator(capability).generate(60, seed=20)
        for request in case.requests
    }
    assert 1 in lengths
    assert capability.max_tokens_per_request in lengths


def test_the_boundary_generator_emits_repeated_token_patterns(
    capability: GeneratorCapability,
) -> None:
    """Random tokens almost never repeat; a wrong cache slot can hide behind that."""
    cases = BoundaryAwareGenerator(capability).generate(60, seed=21)
    uniform_repeats = sum(
        1
        for case in UniformValidGenerator(capability).generate(60, seed=21)
        for request in case.requests
        if request.n_valid > 2 and len(set(request.token_ids)) == 1
    )
    boundary_repeats = sum(
        1
        for case in cases
        for request in case.requests
        if request.n_valid > 2 and len(set(request.token_ids)) == 1
    )
    assert boundary_repeats > uniform_repeats


def test_the_boundary_generator_emits_prefill_boundaries(
    capability: GeneratorCapability,
) -> None:
    """Prefill at 1 and at n-1 are where cache-boundary bugs live."""
    boundaries = {
        (request.prefix_length, request.n_valid)
        for case in BoundaryAwareGenerator(capability).generate(60, seed=22)
        if case.execution_mode is not ExecutionMode.STATELESS_BATCH
        for request in case.requests
    }
    assert any(prefix == 1 for prefix, _ in boundaries)
    assert any(prefix == length - 1 for prefix, length in boundaries if length > 1)


def test_the_boundary_generator_emits_unequal_padded_lengths(
    capability: GeneratorCapability,
) -> None:
    ragged = [
        case
        for case in BoundaryAwareGenerator(capability).generate(60, seed=23)
        if len(case.requests) > 1 and len({r.n_valid for r in case.requests}) > 1
    ]
    assert ragged


# --- capability and registry -------------------------------------------------------------------


def test_capability_is_derived_from_the_fixture(weights: WeightDict) -> None:
    capability = GeneratorCapability.for_fixture(UNIT_FIXTURE, weights_sha256(weights))
    assert capability.model_config_id == UNIT_FIXTURE.config_id
    assert capability.vocab_size == UNIT_FIXTURE.vocab_size
    assert capability.max_tokens_per_request <= UNIT_FIXTURE.max_position
    assert capability.categories == ALL_CATEGORIES


def test_capability_clamps_the_token_limit_to_the_context_window() -> None:
    capability = GeneratorCapability.for_fixture(UNIT_FIXTURE, "a" * 64, max_tokens=10_000)
    assert capability.max_tokens_per_request == UNIT_FIXTURE.max_position


def test_padded_token_limit_leaves_room_for_padding() -> None:
    capability = GeneratorCapability.for_fixture(UNIT_FIXTURE, "a" * 64, max_tokens=32)
    assert capability.padded_token_limit() == 32 - capability.max_pad_left


def test_the_registry_exposes_both_generators(capability: GeneratorCapability) -> None:
    assert set(GENERATORS) == {"uniform-valid", "boundary-aware"}
    assert isinstance(build_generator("uniform-valid", capability), UniformValidGenerator)
    assert isinstance(build_generator("boundary-aware", capability), BoundaryAwareGenerator)
    with pytest.raises(KeyError, match="unknown generator"):
        build_generator("nope", capability)


def test_generators_record_their_identity_in_provenance(
    capability: GeneratorCapability,
) -> None:
    case = UniformValidGenerator(capability).generate(1, seed=24)[0]
    assert any("uniform-valid@" in entry for entry in case.provenance)
    assert any("seed=24" in entry for entry in case.provenance)


def test_provenance_does_not_change_a_case_identity(capability: GeneratorCapability) -> None:
    """Two generators that happen to emit the same case must share a predicate-cache key."""
    case = UniformValidGenerator(capability).generate(1, seed=25)[0]
    rebuilt = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=case.requests,
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
        provenance=("somewhere else",),
    )
    assert rebuilt.case_id == case.case_id


def test_zero_count_is_rejected(capability: GeneratorCapability) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        UniformValidGenerator(capability).generate(0, seed=1)


# --- classification -------------------------------------------------------------------------------


def test_classification_priority_is_fixed() -> None:
    from evallens.types import Request

    def case(requests, mode) -> Case:
        return Case.create(
            model_config_id="c",
            weights_sha256="a" * 64,
            requests=requests,
            execution_mode=mode,
            input_seed=0,
        )

    single = case([Request("r", (1, 2), 2)], ExecutionMode.STATELESS_BATCH)
    assert classify_case(single) is CaseCategory.STATELESS_SINGLE

    batch = case([Request("a", (1,), 1), Request("b", (2,), 1)], ExecutionMode.STATELESS_BATCH)
    assert classify_case(batch) is CaseCategory.STATELESS_BATCH

    # Padding dominates row count: a padded batch exercises the padding machinery.
    padded = case(
        [Request("a", (1,), 1, pad_left=2), Request("b", (2,), 1)], ExecutionMode.STATELESS_BATCH
    )
    assert classify_case(padded) is CaseCategory.STATELESS_PADDED

    prefill = case([Request("r", (1, 2, 3), 3)], ExecutionMode.CACHED_DECODE)
    assert classify_case(prefill) is CaseCategory.CACHED_PREFILL_ONLY

    decode = case([Request("r", (1, 2, 3), 1)], ExecutionMode.CACHED_DECODE)
    assert classify_case(decode) is CaseCategory.CACHED_DECODE

    session = case([Request("a", (1,), 1), Request("b", (2,), 1)], ExecutionMode.SESSION)
    assert classify_case(session) is CaseCategory.SESSION
