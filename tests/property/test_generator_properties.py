"""Generator properties over many seeds and budgets.

The unit tests check a handful of seeds. These check the claim that matters for the
benchmark: *whatever* seed and budget a trial draws, both generators produce valid,
repeatable, correctly labeled cases that the real adapters can execute.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evallens.adapters.encoding import validate_case
from evallens.adapters.native import CandidateAdapter, ReferenceAdapter
from evallens.fixtures.config import UNIT_FIXTURE, make_weights, weights_sha256
from evallens.generate import (
    ALL_CATEGORIES,
    BoundaryAwareGenerator,
    GeneratorCapability,
    UniformValidGenerator,
    classify_case,
)
from evallens.replay import run_comparison
from evallens.types import PAD_TOKEN_ID, TolerancePolicy, Verdict

_WEIGHTS = make_weights(UNIT_FIXTURE)
_SHA = weights_sha256(_WEIGHTS)
CAPABILITY = GeneratorCapability.for_fixture(UNIT_FIXTURE, _SHA, max_tokens=24)
POLICY = TolerancePolicy()

GENERATOR_NAMES = st.sampled_from(["uniform-valid", "boundary-aware"])
SEEDS = st.integers(min_value=0, max_value=2**31 - 1)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _generator(name: str):
    cls = UniformValidGenerator if name == "uniform-valid" else BoundaryAwareGenerator
    return cls(CAPABILITY)


@given(GENERATOR_NAMES, SEEDS, st.integers(min_value=1, max_value=24))
@SETTINGS
def test_generated_cases_are_always_valid(name: str, seed: int, count: int) -> None:
    for case in _generator(name).generate(count, seed):
        validate_case(case, UNIT_FIXTURE, weights_sha256=_SHA)


@given(GENERATOR_NAMES, SEEDS, st.integers(min_value=1, max_value=24))
@SETTINGS
def test_generation_is_always_repeatable(name: str, seed: int, count: int) -> None:
    first = _generator(name).generate(count, seed)
    second = _generator(name).generate(count, seed)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


@given(GENERATOR_NAMES, SEEDS, st.integers(min_value=1, max_value=24))
@SETTINGS
def test_declared_categories_always_match_the_structure(name: str, seed: int, count: int) -> None:
    for case in _generator(name).generate(count, seed):
        assert case.category == classify_case(case).value


@given(GENERATOR_NAMES, SEEDS)
@SETTINGS
def test_a_full_category_cycle_always_covers_every_category(name: str, seed: int) -> None:
    cases = _generator(name).generate(len(ALL_CATEGORIES), seed)
    assert {classify_case(c) for c in cases} == set(ALL_CATEGORIES)


@given(GENERATOR_NAMES, SEEDS, st.integers(min_value=1, max_value=24))
@SETTINGS
def test_tokens_are_always_inside_the_declared_vocabulary(name: str, seed: int, count: int) -> None:
    for case in _generator(name).generate(count, seed):
        for request in case.requests:
            assert request.token_ids
            assert PAD_TOKEN_ID not in request.token_ids
            assert all(0 < t < CAPABILITY.vocab_size for t in request.token_ids)


@given(GENERATOR_NAMES, SEEDS)
@settings(max_examples=12, deadline=None)
def test_generated_cases_actually_execute_and_agree(name: str, seed: int) -> None:
    """The known-good candidate must pass on anything either generator can produce.

    If this ever failed, a "detection" in the benchmark could be the generator's fault
    rather than the candidate's.
    """
    reference = ReferenceAdapter(UNIT_FIXTURE, _WEIGHTS)
    candidate = CandidateAdapter(UNIT_FIXTURE, _WEIGHTS)
    for case in _generator(name).generate(len(ALL_CATEGORIES), seed):
        result = run_comparison(reference, candidate, case, POLICY)
        assert result.verdict is Verdict.PASS, f"{case.category}: {result.detail}"


@given(SEEDS)
@settings(max_examples=20, deadline=None)
def test_both_generators_receive_and_respect_the_same_budget(seed: int) -> None:
    """Equal budgets are what make a generator comparison meaningful."""
    uniform = UniformValidGenerator(CAPABILITY).generate(12, seed)
    boundary = BoundaryAwareGenerator(CAPABILITY).generate(12, seed)
    assert len(uniform) == len(boundary) == 12
    assert [classify_case(c) for c in uniform] == [classify_case(c) for c in boundary]


@given(SEEDS)
@settings(max_examples=20, deadline=None)
def test_the_two_generators_do_not_produce_identical_cases(seed: int) -> None:
    """They share the category schedule but must differ in their choices within it."""
    uniform = {c.case_id for c in UniformValidGenerator(CAPABILITY).generate(18, seed)}
    boundary = {c.case_id for c in BoundaryAwareGenerator(CAPABILITY).generate(18, seed)}
    assert uniform != boundary


@given(GENERATOR_NAMES, SEEDS)
@settings(max_examples=20, deadline=None)
def test_generated_token_counts_stay_within_the_memory_relevant_bounds(
    name: str, seed: int
) -> None:
    """Bounded work per case is what keeps a whole benchmark inside the RSS budget."""
    for case in _generator(name).generate(12, seed):
        assert case.total_valid_tokens <= (
            CAPABILITY.max_tokens_per_request * CAPABILITY.max_batch_rows
        )
        assert case.batch_width <= UNIT_FIXTURE.max_position
        assert np.isfinite(case.total_padding_tokens)
