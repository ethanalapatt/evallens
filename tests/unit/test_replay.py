"""Verdict categorization and stability replay.

The point of these tests is that each failure mode lands in its own bucket. A crash must not
be able to masquerade as a detected regression, and a flaky result must not be promoted to a
stable failure just because one of its runs happened to fail.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from evallens.replay import (
    ReplayBudget,
    RunCounters,
    StabilityResult,
    run_comparison,
    stable_comparison,
)
from evallens.types import (
    Case,
    ExecutionMode,
    ExecutionResult,
    InvalidCaseError,
    Request,
    TolerancePolicy,
    Verdict,
)

POLICY = TolerancePolicy()

CASE = Case.create(
    model_config_id="cfg",
    weights_sha256="a" * 64,
    requests=[Request("r0", (1, 2, 3), prefix_length=2)],
    execution_mode=ExecutionMode.CACHED_DECODE,
    input_seed=1,
)


class FakeAdapter:
    """A scripted adapter. Labeled synthetic: it exists to exercise verdict routing only."""

    def __init__(
        self,
        adapter_id: str,
        outputs,
        *,
        validate_error: Exception | None = None,
        run_error: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._adapter_id = adapter_id
        self._outputs = outputs if callable(outputs) else itertools.repeat(outputs).__next__
        self.validate_error = validate_error
        self.run_error = run_error
        self.delay_s = delay_s
        self.resets = 0
        self.runs = 0

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def reset(self) -> None:
        self.resets += 1

    def validate(self, case: Case) -> None:
        if self.validate_error is not None:
            raise self.validate_error

    def run(self, case: Case, capture: bool = False) -> ExecutionResult:
        self.runs += 1
        if self.run_error is not None:
            raise self.run_error
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        return ExecutionResult(self._adapter_id, case.case_id, {"r0": self._outputs()})


def _constant(value: float) -> np.ndarray:
    return np.full((3, 4), value, dtype=np.float32)


# --- verdict routing -------------------------------------------------------------------------


def test_matching_outputs_are_pass() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    assert run_comparison(reference, candidate, CASE, POLICY).verdict is Verdict.PASS


def test_differing_outputs_are_fail() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(2.0))
    assert run_comparison(reference, candidate, CASE, POLICY).verdict is Verdict.FAIL


def test_an_invalid_case_is_invalid_not_a_failure() -> None:
    reference = FakeAdapter("ref", _constant(1.0), validate_error=InvalidCaseError("bad tokens"))
    candidate = FakeAdapter("cand", _constant(1.0))
    result = run_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.INVALID
    assert "bad tokens" in result.detail
    assert result.diffs == ()


def test_an_invalid_case_raised_during_run_is_still_invalid() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0), run_error=InvalidCaseError("late rejection"))
    assert run_comparison(reference, candidate, CASE, POLICY).verdict is Verdict.INVALID


def test_a_crash_is_an_error_not_a_detection() -> None:
    """The most important routing rule in the project."""
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0), run_error=RuntimeError("kernel exploded"))
    counters = RunCounters()
    result = run_comparison(reference, candidate, CASE, POLICY, counters=counters)
    assert result.verdict is Verdict.ERROR
    assert "RuntimeError: kernel exploded" in result.detail
    assert counters.errors == 1
    assert counters.comparisons == 0


def test_a_validation_crash_is_an_error() -> None:
    reference = FakeAdapter("ref", _constant(1.0), validate_error=TypeError("broken contract"))
    candidate = FakeAdapter("cand", _constant(1.0))
    result = run_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.ERROR
    assert "TypeError" in result.detail


def test_exceeding_the_time_budget_is_a_timeout() -> None:
    reference = FakeAdapter("ref", _constant(1.0), delay_s=0.02)
    candidate = FakeAdapter("cand", _constant(1.0))
    result = run_comparison(
        reference, candidate, CASE, POLICY, budget=ReplayBudget(timeout_s=0.001)
    )
    assert result.verdict is Verdict.TIMEOUT
    assert "exceeding" in result.detail


def test_exceeding_the_rss_ceiling_is_a_resource_limit() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    result = run_comparison(
        reference, candidate, CASE, POLICY, budget=ReplayBudget(rss_limit_bytes=1)
    )
    assert result.verdict is Verdict.RESOURCE_LIMIT
    assert "ceiling" in result.detail


def test_both_adapters_are_reset_before_every_run() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    for _ in range(3):
        run_comparison(reference, candidate, CASE, POLICY)
    assert reference.resets == candidate.resets == 3


def test_counters_separate_model_runs_from_comparisons() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    counters = RunCounters()
    run_comparison(reference, candidate, CASE, POLICY, counters=counters)
    assert counters.model_runs == 2  # one per adapter
    assert counters.comparisons == 1
    assert counters.peak_rss_bytes is not None
    assert counters.to_dict()["model_runs"] == 2


def test_capture_is_off_unless_requested() -> None:
    seen: list[bool] = []

    class Recording(FakeAdapter):
        def run(self, case: Case, capture: bool = False) -> ExecutionResult:
            seen.append(capture)
            return super().run(case, capture)

    run_comparison(
        Recording("ref", _constant(1.0)), Recording("cand", _constant(1.0)), CASE, POLICY
    )
    assert seen == [False, False]


# --- stability ------------------------------------------------------------------------------


def test_a_consistently_failing_case_is_a_stable_failure() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(5.0))
    result = stable_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.FAIL
    assert result.stable is True
    assert result.replays == 3
    assert result.verdict_counts == {"fail": 3}
    assert result.representative.verdict is Verdict.FAIL


def test_a_consistently_passing_case_is_a_stable_pass() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    result = stable_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.PASS
    assert result.stable is True


def test_a_flapping_case_is_unstable_and_never_promoted_to_a_failure() -> None:
    values = iter([_constant(1.0), _constant(9.0), _constant(1.0)])
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", lambda: next(values))
    result = stable_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.UNSTABLE
    assert result.stable is False
    assert result.verdict_counts == {"pass": 2, "fail": 1}


def test_stability_replays_are_charged_to_the_counters() -> None:
    """Hiding replay cost would understate the price of every detection in the benchmark."""
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    counters = RunCounters()
    stable_comparison(reference, candidate, CASE, POLICY, replays=5, counters=counters)
    assert counters.model_runs == 10
    assert counters.comparisons == 5


def test_an_invalid_case_short_circuits_stability_replays() -> None:
    """Input validity is a property of the case, so re-running it proves nothing."""
    reference = FakeAdapter("ref", _constant(1.0), validate_error=InvalidCaseError("nope"))
    candidate = FakeAdapter("cand", _constant(1.0))
    result = stable_comparison(reference, candidate, CASE, POLICY)
    assert result.verdict is Verdict.INVALID
    assert result.stable is True
    assert result.replays == 1


def test_replays_must_be_at_least_one() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(1.0))
    with pytest.raises(ValueError, match="at least 1"):
        stable_comparison(reference, candidate, CASE, POLICY, replays=0)


def test_stability_result_serializes_its_evidence() -> None:
    reference = FakeAdapter("ref", _constant(1.0))
    candidate = FakeAdapter("cand", _constant(3.0))
    payload = stable_comparison(reference, candidate, CASE, POLICY).to_dict()
    assert payload["verdict"] == "fail"
    assert payload["stable"] is True
    assert payload["verdict_counts"] == {"fail": 3}
    assert payload["representative"]["verdict"] == "fail"


def test_representative_prefers_an_observation_matching_the_verdict() -> None:
    observations = stable_comparison(
        FakeAdapter("ref", _constant(1.0)),
        FakeAdapter("cand", lambda: next(iter([_constant(1.0)]))),
        CASE,
        POLICY,
        replays=1,
    )
    assert isinstance(observations, StabilityResult)
    assert observations.representative.verdict is observations.verdict
