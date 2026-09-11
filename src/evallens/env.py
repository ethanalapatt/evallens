"""Environment manifest capture.

Every run records the machine it actually ran on. None of these values are guessed: each
one is read from the interpreter, the installed packages, or the OS at call time, and a
field that cannot be determined is recorded as ``None`` rather than filled with a plausible
default. Benchmark reports refuse to publish when the manifest is incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str | None
    branch: str | None
    dirty: bool | None

    @property
    def publishable(self) -> bool:
        """Headline runs require a known commit and a clean working tree."""
        return self.commit is not None and self.dirty is False


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    python_version: str
    python_executable: str
    torch_version: str | None
    numpy_version: str | None
    platform: str
    machine: str
    os_release: str | None
    chip: str | None
    physical_memory_bytes: int | None
    cpu_count: int | None
    torch_num_threads: int | None
    torch_num_interop_threads: int | None
    default_dtype: str | None
    mps_available: bool | None
    mps_built: bool | None
    cuda_available: bool | None
    thermal_pressure: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def git(self) -> GitState:
        return GitState(self.git_commit, self.git_branch, self.git_dirty)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_id"] = self.manifest_id
        return payload

    def missing_fields(self) -> list[str]:
        return [name for name, value in asdict(self).items() if value is None]


def _thermal_pressure() -> str | None:
    """Best-effort read of macOS thermal pressure. Never changes any power setting."""
    value = _sysctl("machdep.xcpm.cpu_thermal_level")
    if value is None:
        return None
    return f"cpu_thermal_level={value}"


def capture_environment() -> EnvironmentManifest:
    torch_version: str | None = None
    mps_available: bool | None = None
    mps_built: bool | None = None
    cuda_available: bool | None = None
    threads: int | None = None
    interop: int | None = None
    dtype: str | None = None
    try:
        import torch

        torch_version = torch.__version__
        mps_available = bool(torch.backends.mps.is_available())
        mps_built = bool(torch.backends.mps.is_built())
        cuda_available = bool(torch.cuda.is_available())
        threads = int(torch.get_num_threads())
        interop = int(torch.get_num_interop_threads())
        dtype = str(torch.get_default_dtype())
    except Exception:  # pragma: no cover - torch is a hard dependency, but never crash doctor
        pass

    numpy_version: str | None = None
    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:  # pragma: no cover
        pass

    memory = _sysctl("hw.memsize")
    return EnvironmentManifest(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        torch_version=torch_version,
        numpy_version=numpy_version,
        platform=platform.platform(),
        machine=platform.machine(),
        os_release=platform.mac_ver()[0] or platform.release(),
        chip=_sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        physical_memory_bytes=int(memory) if memory and memory.isdigit() else None,
        cpu_count=os.cpu_count(),
        torch_num_threads=threads,
        torch_num_interop_threads=interop,
        default_dtype=dtype,
        mps_available=mps_available,
        mps_built=mps_built,
        cuda_available=cuda_available,
        thermal_pressure=_thermal_pressure(),
        git_commit=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=(lambda s: None if s is None else bool(s))(_git("status", "--porcelain")),
    )


__all__ = ["EnvironmentManifest", "GitState", "capture_environment"]
