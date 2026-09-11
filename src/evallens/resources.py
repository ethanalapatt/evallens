"""Resource budgets: deterministic thread settings, RSS sampling, and time limits.

The design budget is 4 GiB of combined worker RSS with short sequences and one benchmark
worker. That is a *bound we enforce*, not a measurement we claim: :class:`ResourceGuard`
samples real RSS and marks a run ``RESOURCE_LIMIT`` when it crosses the configured ceiling,
rather than letting the machine swap and quietly corrupt every timing in the study.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_THREADS = 4


class ResourceLimitExceeded(RuntimeError):
    """Raised when a guarded region exceeds its RSS ceiling."""


class BudgetExhausted(RuntimeError):
    """Raised when a guarded region exceeds its wall-clock ceiling."""


def set_deterministic_threads(threads: int = DEFAULT_THREADS) -> dict[str, Any]:
    """Pin thread counts so timings and float reductions are comparable across runs.

    Reduction order in a multithreaded matmul can change the last bits of a result. Pinning
    threads does not make CPU FP32 bitwise-reproducible in general, but it removes the
    largest source of run-to-run jitter and it keeps the reference and candidate on equal
    footing, which is what the comparison actually requires.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, str(threads))
    applied: dict[str, Any] = {"requested_threads": threads}
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Already initialized in this process; not fatal and worth recording honestly.
            applied["interop_already_set"] = True
        torch.set_grad_enabled(False)
        applied["torch_num_threads"] = int(torch.get_num_threads())
        applied["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception as exc:  # pragma: no cover
        applied["error"] = repr(exc)
    return applied


def current_rss_bytes() -> int | None:
    """Resident set size of this process, or ``None`` when it cannot be read."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
            return int(usage) if usage > 1 << 24 else int(usage) * 1024
        except Exception:
            return None


@dataclass
class ResourceReport:
    peak_rss_bytes: int | None
    limit_bytes: int
    elapsed_ns: int
    exceeded: bool
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": (
                None if self.peak_rss_bytes is None else round(self.peak_rss_bytes / 1024**2, 1)
            ),
            "limit_bytes": self.limit_bytes,
            "limit_mib": round(self.limit_bytes / 1024**2, 1),
            "elapsed_ns": self.elapsed_ns,
            "elapsed_s": round(self.elapsed_ns / 1e9, 4),
            "exceeded": self.exceeded,
            "samples": self.samples,
        }


class ResourceGuard:
    """Samples RSS at explicit checkpoints and enforces a ceiling."""

    def __init__(self, limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES, *, raise_on_exceed: bool = True):
        self.limit_bytes = limit_bytes
        self.raise_on_exceed = raise_on_exceed
        self.peak_rss_bytes: int | None = None
        self.samples = 0
        self.exceeded = False
        self._start_ns = time.perf_counter_ns()

    def sample(self) -> int | None:
        rss = current_rss_bytes()
        self.samples += 1
        if rss is None:
            return None
        if self.peak_rss_bytes is None or rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if rss > self.limit_bytes:
            self.exceeded = True
            if self.raise_on_exceed:
                raise ResourceLimitExceeded(
                    f"RSS {rss / 1024**2:.1f} MiB exceeded the configured ceiling "
                    f"{self.limit_bytes / 1024**2:.1f} MiB"
                )
        return rss

    def report(self) -> ResourceReport:
        return ResourceReport(
            peak_rss_bytes=self.peak_rss_bytes,
            limit_bytes=self.limit_bytes,
            elapsed_ns=time.perf_counter_ns() - self._start_ns,
            exceeded=self.exceeded,
            samples=self.samples,
        )


@dataclass
class TimeBudget:
    """Wall-clock budget measured with ``perf_counter_ns``."""

    limit_s: float
    _start_ns: int = 0

    def __post_init__(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def reset(self) -> None:
        self._start_ns = time.perf_counter_ns()

    @property
    def elapsed_s(self) -> float:
        return (time.perf_counter_ns() - self._start_ns) / 1e9

    @property
    def remaining_s(self) -> float:
        return self.limit_s - self.elapsed_s

    @property
    def exhausted(self) -> bool:
        return self.remaining_s <= 0.0

    def check(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(f"exceeded {self.limit_s:.3f}s wall-clock budget")


@contextmanager
def guarded(
    limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES, *, raise_on_exceed: bool = True
) -> Iterator[ResourceGuard]:
    guard = ResourceGuard(limit_bytes, raise_on_exceed=raise_on_exceed)
    guard.sample()
    try:
        yield guard
    finally:
        guard.sample()


__all__ = [
    "DEFAULT_RSS_LIMIT_BYTES",
    "DEFAULT_THREADS",
    "BudgetExhausted",
    "ResourceGuard",
    "ResourceLimitExceeded",
    "ResourceReport",
    "TimeBudget",
    "current_rss_bytes",
    "guarded",
    "set_deterministic_threads",
]
