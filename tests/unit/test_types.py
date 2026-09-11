"""Case identity, serialization, size ordering, tolerance policy, and failure signatures."""

from __future__ import annotations

import numpy as np
import pytest

from evallens.types import (
    SCHEMA_VERSION,
    Case,
    CaseSize,
    CheckpointAddress,
    CheckpointKind,
    ExecutionMode,
    FailureSignature,
    Request,
    TolerancePolicy,
    Verdict,
    summarize_tensor,
)


def _case(**overrides) -> Case:
    defaults = {
        "model_config_id": "cfg",
        "weights_sha256": "a" * 64,
        "requests": [Request("r0", (3, 4, 5), prefix_length=2)],
        "execution_mode": ExecutionMode.CACHED_DECODE,
        "input_seed": 7,
    }
    defaults.update(overrides)
    return Case.create(**defaults)  # type: ignore[arg-type]


def test_case_id_is_derived_from_content() -> None:
    assert _case().case_id == _case().case_id
    assert _case().case_id != _case(input_seed=8).case_id


def test_case_id_is_stable_across_serialization_round_trip() -> None:
    original = _case()
    restored = Case.from_dict(original.to_dict())
    assert restored == original
    assert restored.content_hash() == original.content_hash()


def test_provenance_does_not_change_the_content_hash() -> None:
    """Two cases that execute identically must share a predicate-cache key."""
    bare = _case()
    annotated = _case(provenance=["reduced from case_x", "step 12"])
    assert bare.content_hash() == annotated.content_hash()
    assert bare.case_id == annotated.case_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_config_id", "other"),
        ("weights_sha256", "b" * 64),
        ("execution_mode", ExecutionMode.STATELESS_BATCH),
        ("input_seed", 99),
        ("category", "boundary"),
    ],
)
def test_every_behavioral_field_changes_the_content_hash(field: str, value: object) -> None:
    if field == "execution_mode":
        other = _case(execution_mode=value, requests=[Request("r0", (3, 4, 5), prefix_length=3)])
        assert other.content_hash() != _case().content_hash()
        return
    assert _case(**{field: value}).content_hash() != _case().content_hash()


def test_request_token_changes_the_content_hash() -> None:
    other = _case(requests=[Request("r0", (3, 4, 6), prefix_length=2)])
    assert other.content_hash() != _case().content_hash()


def test_from_dict_rejects_an_unknown_schema_version() -> None:
    payload = _case().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        Case.from_dict(payload)


def test_request_decode_step_accounting() -> None:
    request = Request("r", (1, 2, 3, 4, 5), prefix_length=2)
    assert request.n_valid == 5
    assert request.n_decode_steps == 3
    assert Request("r", (1, 2), prefix_length=2).n_decode_steps == 0


def test_batch_width_and_padding_accounting() -> None:
    case = Case.create(
        model_config_id="cfg",
        weights_sha256="a" * 64,
        requests=[
            Request("a", (1, 2, 3), prefix_length=3, pad_left=2),
            Request("b", (4, 5), prefix_length=2),
        ],
        execution_mode=ExecutionMode.STATELESS_BATCH,
        input_seed=1,
    )
    assert case.batch_width == 5
    assert case.total_valid_tokens == 5
    assert case.total_padding_tokens == (5 - 3) + (5 - 2)


def test_padding_is_zero_outside_stateless_execution() -> None:
    assert _case().total_padding_tokens == 0


def test_request_lookup_raises_for_an_unknown_id() -> None:
    with pytest.raises(KeyError):
        _case().request_by_id("nope")


def test_case_size_orders_lexicographically() -> None:
    small = CaseSize(1, 5, 0, 0)
    assert small < CaseSize(1, 6, 0, 0)
    assert small < CaseSize(2, 1, 0, 0)
    assert CaseSize(1, 5, 0, 0) < CaseSize(1, 5, 1, 0)
    assert CaseSize(1, 5, 0, 0) < CaseSize(1, 5, 0, 1)
    assert not small < CaseSize(1, 5, 0, 0)


def test_case_size_reflects_token_value_simplification() -> None:
    complex_case = _case(requests=[Request("r0", (11, 22, 33), prefix_length=2)])
    canonical = _case(requests=[Request("r0", (1, 1, 1), prefix_length=2)])
    assert CaseSize.of(canonical) < CaseSize.of(complex_case)


def test_tolerance_policy_id_depends_on_every_field() -> None:
    base = TolerancePolicy()
    assert base.policy_id == TolerancePolicy().policy_id
    assert base.policy_id != TolerancePolicy(atol=1e-6).policy_id
    assert base.policy_id != TolerancePolicy(rtol=1e-5).policy_id
    assert base.policy_id != TolerancePolicy(zero_norm_eps=1e-10).policy_id
    assert base.policy_id != TolerancePolicy(name="other").policy_id


def test_tolerance_policy_round_trip() -> None:
    policy = TolerancePolicy(atol=3e-5, rtol=2e-4, name="calibrated")
    assert TolerancePolicy.from_dict(policy.to_dict()) == policy


def test_failure_signature_matches_only_on_class_request_and_mode() -> None:
    base = FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE)
    assert base.matches(FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE))
    assert not base.matches(FailureSignature(Verdict.ERROR, "r0", ExecutionMode.CACHED_DECODE))
    assert not base.matches(FailureSignature(Verdict.FAIL, "r1", ExecutionMode.CACHED_DECODE))
    assert not base.matches(FailureSignature(Verdict.FAIL, "r0", ExecutionMode.SESSION))


def test_failure_signature_with_a_checkpoint_ignores_position_but_not_identity() -> None:
    """Deleting tokens renumbers positions, so position is reported, not required to match."""
    address = CheckpointAddress("r0", "block1", 7, CheckpointKind.ATTN_OUT)
    signature = FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE, address)

    moved = CheckpointAddress("r0", "block1", 2, CheckpointKind.ATTN_OUT)
    assert signature.matches(
        FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE, moved)
    )

    other_layer = CheckpointAddress("r0", "block0", 7, CheckpointKind.ATTN_OUT)
    assert not signature.matches(
        FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE, other_layer)
    )

    other_kind = CheckpointAddress("r0", "block1", 7, CheckpointKind.MLP_OUT)
    assert not signature.matches(
        FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE, other_kind)
    )

    assert not signature.matches(
        FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE, None)
    )


def test_signature_without_a_checkpoint_accepts_a_localized_one() -> None:
    """An output-level signature is the weaker requirement and must stay satisfiable."""
    loose = FailureSignature(Verdict.FAIL, "r0", ExecutionMode.CACHED_DECODE)
    localized = FailureSignature(
        Verdict.FAIL,
        "r0",
        ExecutionMode.CACHED_DECODE,
        CheckpointAddress("r0", "block0", 1, CheckpointKind.LOGITS),
    )
    assert loose.matches(localized)


def test_failure_signature_round_trip() -> None:
    signature = FailureSignature(
        Verdict.FAIL,
        "r0",
        ExecutionMode.SESSION,
        CheckpointAddress("r0", "block1", 3, CheckpointKind.BLOCK_OUT),
    )
    assert FailureSignature.from_dict(signature.to_dict()) == signature


def test_checkpoint_address_key_and_string_are_stable() -> None:
    address = CheckpointAddress("r0", "block1", 4, CheckpointKind.MLP_OUT)
    assert address.key() == ("r0", "block1", 4, "mlp_out")
    assert address.as_str() == "r0/block1/pos4/mlp_out"
    assert CheckpointAddress.from_dict(address.to_dict()) == address


def test_summarize_tensor_counts_nonfinite_without_poisoning_statistics() -> None:
    values = np.array([1.0, -3.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    summary = summarize_tensor(values)
    assert summary["count"] == 5
    assert summary["n_finite"] == 2
    assert summary["n_nan"] == 1
    assert summary["n_inf"] == 2
    assert summary["absmax"] == 3.0
    assert summary["mean"] == pytest.approx(-1.0)


def test_summarize_tensor_handles_an_all_nonfinite_tensor() -> None:
    summary = summarize_tensor(np.array([np.nan, np.inf]))
    assert summary["n_finite"] == 0
    assert summary["mean"] == 0.0
    assert summary["l2"] == 0.0
