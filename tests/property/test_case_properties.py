"""Properties of the case schema: validity, identity, serialization, and size ordering."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings

from evallens.adapters.encoding import (
    encode_cached_steps,
    encode_stateless_batch,
    validate_case,
)
from evallens.fixtures.config import UNIT_FIXTURE
from evallens.types import CANONICAL_TOKEN_ID, PAD_TOKEN_ID, Case, CaseSize, ExecutionMode, Request

from .strategies import cached_request, valid_cases

SETTINGS = settings(max_examples=150, deadline=None)


@given(valid_cases())
@SETTINGS
def test_generated_cases_are_valid(case: Case) -> None:
    validate_case(case, UNIT_FIXTURE, weights_sha256=case.weights_sha256)


@given(valid_cases())
@SETTINGS
def test_case_serialization_round_trips(case: Case) -> None:
    restored = Case.from_dict(case.to_dict())
    assert restored == case
    assert restored.content_hash() == case.content_hash()


@given(valid_cases())
@SETTINGS
def test_case_id_is_a_function_of_content_alone(case: Case) -> None:
    rebuilt = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=case.requests,
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
        provenance=("different", "provenance"),
    )
    assert rebuilt.case_id == case.case_id


@given(valid_cases())
@SETTINGS
def test_stateless_encoding_preserves_every_valid_token(case: Case) -> None:
    if case.execution_mode is not ExecutionMode.STATELESS_BATCH:
        return
    encoding = encode_stateless_batch(case)
    for layout in encoding.layouts:
        request = case.request_by_id(layout.request_id)
        row = encoding.token_ids[layout.row, layout.col_start : layout.col_stop]
        np.testing.assert_array_equal(row, np.asarray(request.token_ids))
        assert encoding.key_valid[layout.row, layout.col_start : layout.col_stop].all()
    assert int(encoding.key_valid.sum()) == case.total_valid_tokens
    assert (encoding.token_ids[~encoding.key_valid] == PAD_TOKEN_ID).all()


@given(valid_cases())
@SETTINGS
def test_stateless_positions_are_contiguous_from_zero(case: Case) -> None:
    if case.execution_mode is not ExecutionMode.STATELESS_BATCH:
        return
    encoding = encode_stateless_batch(case)
    for layout in encoding.layouts:
        positions = encoding.position_ids[layout.row, layout.col_start : layout.col_stop]
        np.testing.assert_array_equal(positions, np.arange(layout.n_valid))


@given(cached_request())
@SETTINGS
def test_cached_steps_partition_the_request(request: Request) -> None:
    steps = encode_cached_steps(request)
    covered = [position for _, _, logical in steps for position in logical]
    assert covered == list(range(request.n_valid))
    assert len(steps) == 1 + request.n_decode_steps
    fed = [int(t) for tokens, _, _ in steps for t in tokens.ravel()]
    assert fed == list(request.token_ids)


@given(valid_cases())
@SETTINGS
def test_size_is_nonnegative_and_consistent(case: Case) -> None:
    size = CaseSize.of(case)
    assert size.n_requests == len(case.requests) >= 1
    assert size.n_valid_tokens == case.total_valid_tokens >= size.n_requests
    assert size.n_padding_tokens >= 0
    assert size.token_value_complexity > 0


@given(valid_cases())
@SETTINGS
def test_deleting_a_token_strictly_decreases_size(case: Case) -> None:
    """The guarantee that keeps the reducer from cycling forever."""
    target = max(case.requests, key=lambda r: r.n_valid)
    if target.n_valid < 2:
        return
    shortened = Request(
        target.request_id,
        target.token_ids[:-1],
        prefix_length=min(target.prefix_length, target.n_valid - 1),
        pad_left=target.pad_left,
    )
    smaller = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=[shortened if r.request_id == target.request_id else r for r in case.requests],
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
    )
    assert CaseSize.of(smaller) < CaseSize.of(case)


@given(valid_cases())
@SETTINGS
def test_removing_a_request_strictly_decreases_size(case: Case) -> None:
    if len(case.requests) < 2:
        return
    smaller = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=case.requests[1:],
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
    )
    assert CaseSize.of(smaller) < CaseSize.of(case)


@given(valid_cases())
@SETTINGS
def test_canonicalizing_token_values_never_increases_size(case: Case) -> None:
    """Simplifying toward the fixed canonical token can only move a case down the order."""
    value = CANONICAL_TOKEN_ID
    canonical = Case.create(
        model_config_id=case.model_config_id,
        weights_sha256=case.weights_sha256,
        requests=[
            Request(r.request_id, (value,) * r.n_valid, r.prefix_length, r.pad_left)
            for r in case.requests
        ],
        execution_mode=case.execution_mode,
        input_seed=case.input_seed,
        category=case.category,
    )
    assert CaseSize.of(canonical) <= CaseSize.of(case)
