"""Input validation and the canonical case-to-tensor encoding."""

from __future__ import annotations

import numpy as np
import pytest

from evallens.adapters.encoding import (
    MAX_BATCH_ROWS,
    MAX_SESSION_REQUESTS,
    encode_cached_steps,
    encode_stateless_batch,
    encode_unpadded_single,
    validate_case,
)
from evallens.fixtures.config import ModelConfig
from evallens.types import PAD_TOKEN_ID, Case, ExecutionMode, InvalidCaseError, Request


def _case(config: ModelConfig, requests, mode=ExecutionMode.STATELESS_BATCH, **kw) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=kw.pop("weights_sha256", "a" * 64),
        requests=requests,
        execution_mode=mode,
        input_seed=kw.pop("input_seed", 0),
        **kw,
    )


def _r(name: str, tokens, prefix=None, pad_left=0) -> Request:
    length = len(tokens) if prefix is None else prefix
    return Request(name, tuple(tokens), prefix_length=length, pad_left=pad_left)


# --- validation ------------------------------------------------------------------------


def test_valid_cases_pass(config: ModelConfig) -> None:
    validate_case(_case(config, [_r("a", [1, 2, 3])]), config)
    validate_case(
        _case(config, [_r("a", [1, 2, 3], prefix=1)], ExecutionMode.CACHED_DECODE), config
    )
    validate_case(_case(config, [_r("a", [1, 2]), _r("b", [3])], ExecutionMode.SESSION), config)


def test_weights_hash_mismatch_is_invalid(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2])])
    with pytest.raises(InvalidCaseError, match="weights"):
        validate_case(case, config, weights_sha256="b" * 64)


def test_model_config_mismatch_is_invalid(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2])])
    with pytest.raises(InvalidCaseError, match="model config"):
        validate_case(case, ModelConfig(n_layers=4, name="other"))


def test_empty_request_list_is_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="at least one request"):
        validate_case(_case(config, []), config)


def test_all_padding_request_is_invalid(config: ModelConfig) -> None:
    """The single most important rejection: an empty row must never be silently accepted."""
    case = _case(config, [Request("a", (), prefix_length=1, pad_left=4)])
    with pytest.raises(InvalidCaseError, match="no valid tokens"):
        validate_case(case, config)


def test_padding_token_id_inside_token_ids_is_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="reserved padding id"):
        validate_case(_case(config, [_r("a", [1, PAD_TOKEN_ID, 3])]), config)


@pytest.mark.parametrize("token", [-1, 97, 1000])
def test_out_of_vocabulary_token_is_invalid(config: ModelConfig, token: int) -> None:
    with pytest.raises(InvalidCaseError, match=r"outside vocabulary|reserved padding"):
        validate_case(_case(config, [_r("a", [1, token])]), config)


def test_non_integer_token_is_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="not an int"):
        validate_case(_case(config, [Request("a", (1, 2.5), prefix_length=2)]), config)  # type: ignore[arg-type]


def test_boolean_token_is_invalid(config: ModelConfig) -> None:
    """``bool`` is an ``int`` subclass; accepting it would let True become token 1."""
    with pytest.raises(InvalidCaseError, match="not an int"):
        validate_case(_case(config, [Request("a", (1, True), prefix_length=2)]), config)  # type: ignore[arg-type]


def test_request_longer_than_context_is_invalid(config: ModelConfig) -> None:
    tokens = [1] * (config.max_position + 1)
    with pytest.raises(InvalidCaseError, match="context limit"):
        validate_case(_case(config, [_r("a", tokens)]), config)


def test_padded_width_over_context_is_invalid(config: ModelConfig) -> None:
    tokens = [1] * config.max_position
    case = _case(config, [_r("a", tokens, pad_left=1)])
    with pytest.raises(InvalidCaseError, match="padded batch width"):
        validate_case(case, config)


@pytest.mark.parametrize("prefix", [0, -1, 4])
def test_prefix_length_outside_range_is_invalid(config: ModelConfig, prefix: int) -> None:
    case = _case(config, [_r("a", [1, 2, 3], prefix=prefix)], ExecutionMode.CACHED_DECODE)
    with pytest.raises(InvalidCaseError, match="prefix_length"):
        validate_case(case, config)


def test_stateless_prefix_must_cover_the_whole_request(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2, 3], prefix=2)])
    with pytest.raises(InvalidCaseError, match="prefix_length must equal"):
        validate_case(case, config)


def test_cached_execution_rejects_padding(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2], prefix=1, pad_left=2)], ExecutionMode.CACHED_DECODE)
    with pytest.raises(InvalidCaseError, match="pad_left must be 0"):
        validate_case(case, config)


def test_negative_padding_is_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="negative"):
        validate_case(_case(config, [_r("a", [1, 2], pad_left=-1)]), config)


def test_cached_decode_requires_exactly_one_request(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1]), _r("b", [2])], ExecutionMode.CACHED_DECODE)
    with pytest.raises(InvalidCaseError, match="batch size one"):
        validate_case(case, config)


def test_duplicate_request_ids_are_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="duplicate request_id"):
        validate_case(_case(config, [_r("a", [1]), _r("a", [2])]), config)


def test_empty_request_id_is_invalid(config: ModelConfig) -> None:
    with pytest.raises(InvalidCaseError, match="nonempty string"):
        validate_case(_case(config, [_r("", [1])]), config)


def test_batch_and_session_limits_are_enforced(config: ModelConfig) -> None:
    rows = [_r(f"r{i}", [1, 2]) for i in range(MAX_BATCH_ROWS + 1)]
    with pytest.raises(InvalidCaseError, match="limit is"):
        validate_case(_case(config, rows), config)

    requests = [_r(f"r{i}", [1, 2]) for i in range(MAX_SESSION_REQUESTS + 1)]
    with pytest.raises(InvalidCaseError, match="limit is"):
        validate_case(_case(config, requests, ExecutionMode.SESSION), config)


# --- encoding --------------------------------------------------------------------------


def test_stateless_encoding_layout(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [5, 6, 7], pad_left=2), _r("b", [8, 9])])
    encoding = encode_stateless_batch(case)

    assert encoding.batch_size == 2
    assert encoding.width == 5
    np.testing.assert_array_equal(encoding.token_ids[0], [0, 0, 5, 6, 7])
    np.testing.assert_array_equal(encoding.token_ids[1], [8, 9, 0, 0, 0])
    np.testing.assert_array_equal(encoding.key_valid[0], [False, False, True, True, True])
    np.testing.assert_array_equal(encoding.key_valid[1], [True, True, False, False, False])


def test_position_ids_are_assigned_by_valid_index_not_column(config: ModelConfig) -> None:
    """The convention that makes padding invariance testable."""
    case = _case(config, [_r("a", [5, 6, 7], pad_left=2)])
    encoding = encode_stateless_batch(case)
    np.testing.assert_array_equal(encoding.position_ids[0, 2:5], [0, 1, 2])


def test_padding_columns_never_carry_a_valid_flag(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1], pad_left=3), _r("b", [2, 3, 4, 5])])
    encoding = encode_stateless_batch(case)
    assert encoding.key_valid.sum() == 5
    assert (encoding.token_ids[~encoding.key_valid] == PAD_TOKEN_ID).all()


def test_layouts_locate_each_request(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2], pad_left=2), _r("b", [3, 4, 5, 6])])
    layouts = {layout.request_id: layout for layout in encode_stateless_batch(case).layouts}
    assert (layouts["a"].row, layouts["a"].col_start, layouts["a"].col_stop) == (0, 2, 4)
    assert (layouts["b"].row, layouts["b"].col_start, layouts["b"].col_stop) == (1, 0, 4)


def test_stateless_encoder_rejects_a_cached_case(config: ModelConfig) -> None:
    case = _case(config, [_r("a", [1, 2], prefix=1)], ExecutionMode.CACHED_DECODE)
    with pytest.raises(InvalidCaseError, match="requires STATELESS_BATCH"):
        encode_stateless_batch(case)


def test_cached_steps_split_prefill_and_decode() -> None:
    request = _r("a", [10, 11, 12, 13], prefix=2)
    steps = encode_cached_steps(request)

    assert len(steps) == 3
    prefill_tokens, prefill_positions, prefill_logical = steps[0]
    np.testing.assert_array_equal(prefill_tokens, [[10, 11]])
    np.testing.assert_array_equal(prefill_positions, [[0, 1]])
    assert prefill_logical == [0, 1]

    for index, position in enumerate((2, 3)):
        tokens, positions, logical = steps[index + 1]
        np.testing.assert_array_equal(tokens, [[request.token_ids[position]]])
        np.testing.assert_array_equal(positions, [[position]])
        assert logical == [position]


def test_cached_steps_are_teacher_forced() -> None:
    """Every step feeds the case's own canonical token, never a model prediction."""
    request = _r("a", [4, 5, 6], prefix=1)
    fed = [int(tokens[0, 0]) for tokens, _, _ in encode_cached_steps(request)[1:]]
    assert fed == [5, 6]


def test_cached_steps_with_full_prefill_have_no_decode_steps() -> None:
    assert len(encode_cached_steps(_r("a", [1, 2, 3], prefix=3))) == 1


def test_cached_steps_cover_every_position_exactly_once() -> None:
    request = _r("a", list(range(1, 9)), prefix=3)
    covered = [p for _, _, logical in encode_cached_steps(request) for p in logical]
    assert covered == list(range(8))


def test_unpadded_single_encoding() -> None:
    encoding = encode_unpadded_single(_r("a", [7, 8, 9], prefix=1))
    np.testing.assert_array_equal(encoding.token_ids, [[7, 8, 9]])
    np.testing.assert_array_equal(encoding.position_ids, [[0, 1, 2]])
    assert encoding.key_valid.all()
    assert encoding.layouts[0].col_start == 0
