"""The numerical policy: tolerance band, nonfinite handling, and output-contract checks."""

from __future__ import annotations

import numpy as np
import pytest

from evallens.compare import compare, compare_arrays, within_policy
from evallens.types import ExecutionResult, TolerancePolicy, Verdict

POLICY = TolerancePolicy(atol=1e-5, rtol=1e-4)


def _result(
    adapter: str, outputs: dict[str, np.ndarray], case_id: str = "case_x"
) -> ExecutionResult:
    return ExecutionResult(adapter_id=adapter, case_id=case_id, outputs=outputs)


def _pair(reference: np.ndarray, candidate: np.ndarray):
    return _result("ref", {"r0": reference}), _result("cand", {"r0": candidate})


# --- the tolerance band ------------------------------------------------------------------


def test_identical_tensors_pass_with_zero_error() -> None:
    values = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    diff = compare_arrays("k", values, values.copy(), POLICY)
    assert not diff.violated
    assert diff.max_abs_err == 0.0
    assert diff.rel_l2_err == 0.0
    assert diff.n_violations == 0


def test_difference_just_inside_the_band_passes() -> None:
    reference = np.array([1.0], dtype=np.float32)
    inside = reference + np.float32(0.9 * (POLICY.atol + POLICY.rtol * 1.0))
    assert within_policy(reference, inside, POLICY)


def test_difference_just_outside_the_band_fails() -> None:
    reference = np.array([1.0], dtype=np.float32)
    outside = reference + np.float32(1.1 * (POLICY.atol + POLICY.rtol * 1.0))
    assert not within_policy(reference, outside, POLICY)


def test_the_band_scales_with_reference_magnitude() -> None:
    """rtol must actually do something: the same absolute error passes at a larger value."""
    delta = 5e-4
    assert not within_policy(np.array([1.0]), np.array([1.0 + delta]), POLICY)
    assert within_policy(np.array([1000.0]), np.array([1000.0 + delta]), POLICY)


def test_atol_covers_values_near_zero() -> None:
    assert within_policy(np.array([0.0]), np.array([5e-6]), POLICY)
    assert not within_policy(np.array([0.0]), np.array([5e-5]), POLICY)


def test_violation_counts_and_fraction_are_exact() -> None:
    reference = np.zeros(10, dtype=np.float32)
    candidate = reference.copy()
    candidate[:3] = 1.0
    diff = compare_arrays("k", reference, candidate, POLICY)
    assert diff.n_violations == 3
    assert diff.n_elements == 10
    assert diff.violating_fraction == pytest.approx(0.3)
    assert diff.max_abs_err == pytest.approx(1.0)


def test_shape_and_dtype_are_reported() -> None:
    reference = np.zeros((2, 3), dtype=np.float32)
    diff = compare_arrays("k", reference, reference.copy(), POLICY)
    assert diff.shape == (2, 3)
    assert diff.dtype == "float32"


def test_mismatched_shapes_raise_at_the_array_level() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        compare_arrays("k", np.zeros(3), np.zeros(4), POLICY)


# --- relative L2 and the zero-norm rule ---------------------------------------------------


def test_relative_l2_uses_the_reference_norm() -> None:
    reference = np.array([3.0, 4.0])  # L2 norm 5
    candidate = np.array([3.0, 4.5])
    diff = compare_arrays("k", reference, candidate, POLICY)
    assert diff.rel_l2_err == pytest.approx(0.5 / 5.0)
    assert not diff.rel_l2_denominator_degenerate


def test_near_zero_reference_norm_reports_absolute_l2_and_flags_it() -> None:
    """Never divide by a tiny norm to manufacture an impressive relative error."""
    reference = np.zeros(4)
    candidate = np.array([0.0, 0.0, 0.0, 2.0])
    diff = compare_arrays("k", reference, candidate, POLICY)
    assert diff.rel_l2_denominator_degenerate
    assert diff.rel_l2_err == pytest.approx(2.0)
    assert np.isfinite(diff.rel_l2_err)


# --- nonfinite handling -------------------------------------------------------------------


def test_candidate_nan_against_a_finite_reference_is_a_violation() -> None:
    diff = compare_arrays("k", np.array([1.0, 2.0]), np.array([1.0, np.nan]), POLICY)
    assert diff.violated
    assert diff.n_violations == 1
    assert diff.candidate_nonfinite == 1
    assert diff.reference_nonfinite == 0
    assert np.isfinite(diff.max_abs_err)


def test_candidate_inf_against_a_finite_reference_is_a_violation() -> None:
    diff = compare_arrays("k", np.array([1.0]), np.array([np.inf]), POLICY)
    assert diff.violated
    assert diff.candidate_nonfinite == 1


def test_matching_nans_are_agreement_not_a_violation() -> None:
    """The reference genuinely produced NaN there and the candidate matched it."""
    values = np.array([1.0, np.nan])
    diff = compare_arrays("k", values, values.copy(), POLICY)
    assert not diff.violated
    assert diff.reference_nonfinite == diff.candidate_nonfinite == 1


def test_differing_nonfinite_kinds_are_a_violation() -> None:
    diff = compare_arrays("k", np.array([np.inf]), np.array([np.nan]), POLICY)
    assert diff.violated
    diff2 = compare_arrays("k", np.array([np.inf]), np.array([-np.inf]), POLICY)
    assert diff2.violated


def test_a_candidate_that_loses_a_nan_is_also_a_violation() -> None:
    diff = compare_arrays("k", np.array([np.nan]), np.array([0.0]), POLICY)
    assert diff.violated


def test_max_abs_err_ignores_nonfinite_positions() -> None:
    """A single NaN must not poison the reported magnitude of the finite disagreement."""
    reference = np.array([1.0, 2.0, 3.0])
    candidate = np.array([1.5, np.nan, 3.0])
    diff = compare_arrays("k", reference, candidate, POLICY)
    assert diff.max_abs_err == pytest.approx(0.5)


# --- verdict-level comparison --------------------------------------------------------------


def test_matching_outputs_are_pass() -> None:
    values = np.zeros((3, 5), dtype=np.float32)
    result = compare(*_pair(values, values.copy()), POLICY)
    assert result.verdict is Verdict.PASS
    assert result.failing_request_ids == ()
    assert "within tolerance" in result.detail


def test_violating_outputs_are_fail_with_evidence_in_the_detail() -> None:
    reference = np.zeros((2, 4), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = 1.0
    result = compare(*_pair(reference, candidate), POLICY)
    assert result.verdict is Verdict.FAIL
    assert result.failing_request_ids == ("r0",)
    assert "max|Δ|" in result.detail
    assert result.max_abs_err == pytest.approx(1.0)


def test_nonfinite_candidate_output_is_called_out_explicitly() -> None:
    reference = np.zeros(4, dtype=np.float32)
    candidate = reference.copy()
    candidate[2] = np.nan
    result = compare(*_pair(reference, candidate), POLICY)
    assert result.verdict is Verdict.FAIL
    assert "nonfinite candidate output" in result.detail


def test_a_missing_request_is_an_output_contract_failure() -> None:
    reference = _result("ref", {"a": np.zeros(3), "b": np.zeros(3)})
    candidate = _result("cand", {"a": np.zeros(3)})
    result = compare(reference, candidate, POLICY)
    assert result.verdict is Verdict.FAIL
    assert "output contract violated" in result.detail
    assert result.failing_request_ids == ("b",)


def test_an_unexpected_request_is_an_output_contract_failure() -> None:
    reference = _result("ref", {"a": np.zeros(3)})
    candidate = _result("cand", {"a": np.zeros(3), "z": np.zeros(3)})
    result = compare(reference, candidate, POLICY)
    assert result.verdict is Verdict.FAIL
    assert result.failing_request_ids == ("z",)


def test_a_shape_disagreement_is_an_output_contract_failure() -> None:
    """Producing the wrong number of positions is a regression regardless of the values."""
    result = compare(*_pair(np.zeros((4, 2)), np.zeros((3, 2))), POLICY)
    assert result.verdict is Verdict.FAIL
    assert "shape" in result.detail
    assert result.failing_request_ids == ("r0",)


def test_comparing_results_from_different_cases_raises() -> None:
    reference = _result("ref", {"a": np.zeros(3)}, case_id="case_a")
    candidate = _result("cand", {"a": np.zeros(3)}, case_id="case_b")
    with pytest.raises(ValueError, match="different cases"):
        compare(reference, candidate, POLICY)


def test_only_the_violating_request_is_named() -> None:
    reference = _result("ref", {"a": np.zeros(3), "b": np.zeros(3)})
    candidate = _result("cand", {"a": np.zeros(3), "b": np.ones(3)})
    result = compare(reference, candidate, POLICY)
    assert result.failing_request_ids == ("b",)
    assert len(result.diffs) == 2


def test_the_policy_travels_with_the_result() -> None:
    strict = TolerancePolicy(atol=0.0, rtol=0.0, name="strict")
    reference = np.zeros(3, dtype=np.float32)
    candidate = reference + np.float32(1e-9)
    assert compare(*_pair(reference, candidate), strict).verdict is Verdict.FAIL
    assert compare(*_pair(reference, candidate), POLICY).verdict is Verdict.PASS

    result = compare(*_pair(reference, candidate), strict)
    assert result.policy.policy_id == strict.policy_id
    assert result.to_dict()["policy"]["name"] == "strict"


def test_result_serialization_carries_the_diffs() -> None:
    reference = np.zeros(4, dtype=np.float32)
    candidate = reference.copy()
    candidate[0] = 1.0
    payload = compare(*_pair(reference, candidate), POLICY).to_dict()
    assert payload["verdict"] == "fail"
    assert payload["diffs"][0]["n_violations"] == 1
    assert payload["diffs"][0]["violated"] is True
