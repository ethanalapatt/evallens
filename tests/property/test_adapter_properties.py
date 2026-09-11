"""Properties of the adapters over the whole declared valid case space.

The parametrized integration tests check specific shapes. These check the *claim*: for any
valid case, the known-good candidate agrees with the reference. A control that only holds on
hand-picked lengths would be worthless as a baseline for detection.

Example counts are kept modest because every example runs the real model.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings

from evallens.adapters.native import CandidateAdapter, ReferenceAdapter
from evallens.compare import compare
from evallens.fixtures.config import UNIT_FIXTURE, make_weights, weights_sha256
from evallens.replay import run_comparison, stable_comparison
from evallens.types import Case, ExecutionMode, TolerancePolicy, Verdict

from .strategies import valid_cases

POLICY = TolerancePolicy()
_WEIGHTS = make_weights(UNIT_FIXTURE)
_SHA = weights_sha256(_WEIGHTS)
_REFERENCE = ReferenceAdapter(UNIT_FIXTURE, _WEIGHTS)
_CANDIDATE = CandidateAdapter(UNIT_FIXTURE, _WEIGHTS)

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _retarget(case: Case) -> Case:
    """Rebuild a strategy-generated case against the real weights hash."""
    return Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=_SHA,
        requests=case.requests,
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
    )


@given(valid_cases())
@SETTINGS
def test_the_known_good_candidate_always_agrees_with_the_reference(case: Case) -> None:
    """The M2 control, stated as a property rather than a handful of examples."""
    case = _retarget(case)
    result = compare(_REFERENCE.run(case), _CANDIDATE.run(case), POLICY)
    assert result.verdict is Verdict.PASS, f"{case.execution_mode.value}: {result.detail}"


@given(valid_cases())
@SETTINGS
def test_outputs_have_one_row_per_valid_token_and_are_finite(case: Case) -> None:
    case = _retarget(case)
    for result in (_REFERENCE.run(case), _CANDIDATE.run(case)):
        assert set(result.outputs) == {r.request_id for r in case.requests}
        for request in case.requests:
            array = result.outputs[request.request_id]
            assert array.shape == (request.n_valid, UNIT_FIXTURE.vocab_size)
            assert array.dtype == np.float32
            assert np.isfinite(array).all()


@given(valid_cases())
@SETTINGS
def test_execution_is_deterministic_across_repeats(case: Case) -> None:
    """Determinism is what makes an UNSTABLE verdict meaningful when it does appear."""
    case = _retarget(case)
    first = _CANDIDATE.run(case)
    second = _CANDIDATE.run(case)
    for request_id, values in first.outputs.items():
        np.testing.assert_array_equal(values, second.outputs[request_id])


@given(valid_cases())
@SETTINGS
def test_a_valid_case_never_produces_error_or_invalid(case: Case) -> None:
    case = _retarget(case)
    result = run_comparison(_REFERENCE, _CANDIDATE, case, POLICY)
    assert result.verdict in {Verdict.PASS, Verdict.FAIL}, result.detail


@given(valid_cases())
@SETTINGS
def test_known_good_comparisons_are_stable_passes(case: Case) -> None:
    case = _retarget(case)
    result = stable_comparison(_REFERENCE, _CANDIDATE, case, POLICY)
    assert result.verdict is Verdict.PASS
    assert result.stable is True


@given(valid_cases())
@SETTINGS
def test_a_case_carrying_the_wrong_weights_hash_is_always_invalid(case: Case) -> None:
    mismatched = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256="0" * 64,
        requests=case.requests,
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
    )
    assert run_comparison(_REFERENCE, _CANDIDATE, mismatched, POLICY).verdict is Verdict.INVALID


@given(valid_cases())
@SETTINGS
def test_cached_and_stateless_paths_agree_on_single_requests(case: Case) -> None:
    """One request, run as a full-prefix batch row and as incremental decode, must agree."""
    case = _retarget(case)
    request = case.requests[0]
    if request.pad_left:
        return

    from evallens.types import Request

    stateless = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=_SHA,
        requests=[Request(request.request_id, request.token_ids, request.n_valid)],
        execution_mode=ExecutionMode.STATELESS_BATCH,
        input_seed=case.input_seed,
    )
    cached = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=_SHA,
        requests=[request],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=case.input_seed,
    )
    batched = _REFERENCE.run(stateless).outputs[request.request_id]
    incremental = _CANDIDATE.run(cached).outputs[request.request_id]
    assert np.abs(batched - incremental).max() < POLICY.atol
