"""Resource budgets and environment capture."""

from __future__ import annotations

import time

import pytest

from evallens.env import EnvironmentManifest, capture_environment
from evallens.resources import (
    BudgetExhausted,
    ResourceGuard,
    ResourceLimitExceeded,
    TimeBudget,
    current_rss_bytes,
    guarded,
    set_deterministic_threads,
)


def test_rss_is_readable_and_plausible() -> None:
    rss = current_rss_bytes()
    assert rss is not None
    assert 1 << 20 < rss < 64 * 1024**3


def test_guard_tracks_a_peak_and_reports_within_budget() -> None:
    with guarded() as guard:
        guard.sample()
        guard.sample()
    report = guard.report()
    assert report.exceeded is False
    assert report.samples >= 4
    assert report.peak_rss_bytes is not None
    assert report.elapsed_ns > 0
    assert report.to_dict()["peak_rss_mib"] == pytest.approx(
        report.peak_rss_bytes / 1024**2, abs=0.1
    )


def test_guard_raises_when_the_ceiling_is_crossed() -> None:
    """An impossible ceiling proves the check is real rather than never evaluated."""
    guard = ResourceGuard(limit_bytes=1)
    with pytest.raises(ResourceLimitExceeded, match="exceeded the configured ceiling"):
        guard.sample()
    assert guard.exceeded is True


def test_guard_can_record_an_overrun_without_raising() -> None:
    guard = ResourceGuard(limit_bytes=1, raise_on_exceed=False)
    guard.sample()
    assert guard.exceeded is True
    assert guard.report().exceeded is True


def test_time_budget_accounting() -> None:
    budget = TimeBudget(limit_s=10.0)
    assert not budget.exhausted
    assert 0.0 < budget.remaining_s <= 10.0
    budget.check()

    expired = TimeBudget(limit_s=0.001)
    time.sleep(0.005)
    assert expired.exhausted
    assert expired.remaining_s < 0
    with pytest.raises(BudgetExhausted, match="wall-clock budget"):
        expired.check()


def test_time_budget_reset() -> None:
    budget = TimeBudget(limit_s=0.001)
    time.sleep(0.005)
    assert budget.exhausted
    budget.reset()
    assert budget.elapsed_s < 0.001 or budget.elapsed_s >= 0


def test_deterministic_threads_are_applied() -> None:
    report = set_deterministic_threads(2)
    assert report["requested_threads"] == 2
    assert report["torch_num_threads"] == 2


def test_environment_manifest_is_populated_and_hashable() -> None:
    manifest = capture_environment()
    assert isinstance(manifest, EnvironmentManifest)
    assert manifest.torch_version is not None
    assert manifest.numpy_version is not None
    assert manifest.cpu_count and manifest.cpu_count > 0
    assert len(manifest.manifest_id) == 16
    assert manifest.to_dict()["manifest_id"] == manifest.manifest_id


def test_manifest_reports_missing_fields_instead_of_inventing_them() -> None:
    partial = EnvironmentManifest(
        python_version="3.13.7",
        python_executable="/x",
        torch_version=None,
        numpy_version="2.0",
        platform="p",
        machine="arm64",
        os_release="15.6",
        chip=None,
        physical_memory_bytes=1,
        cpu_count=8,
        torch_num_threads=4,
        torch_num_interop_threads=1,
        default_dtype="torch.float32",
        mps_available=True,
        mps_built=True,
        cuda_available=False,
        thermal_pressure=None,
        git_commit="abc",
        git_branch="main",
        git_dirty=False,
    )
    assert set(partial.missing_fields()) == {"torch_version", "chip", "thermal_pressure"}


def test_publishable_requires_a_clean_known_commit() -> None:
    def manifest(commit: str | None, dirty: bool | None) -> EnvironmentManifest:
        return EnvironmentManifest(
            "3.13",
            "/x",
            "2",
            "2",
            "p",
            "arm64",
            "15",
            "M3",
            1,
            8,
            4,
            1,
            "f32",
            True,
            True,
            False,
            None,
            commit,
            "main",
            dirty,
        )

    assert manifest("abc", False).git.publishable
    assert not manifest("abc", True).git.publishable
    assert not manifest(None, False).git.publishable
    assert not manifest("abc", None).git.publishable
