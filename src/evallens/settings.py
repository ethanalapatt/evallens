"""Configuration loading.

TOML in, typed objects out. Unknown keys are an error rather than being ignored: a silently
dropped `atol` would change every verdict in a run while looking like it had been configured.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evallens.reduce import ReductionBudget
from evallens.resources import DEFAULT_RSS_LIMIT_BYTES
from evallens.types import TolerancePolicy

if TYPE_CHECKING:
    from evallens.adapters.native import CaptureBudget
    from evallens.fixtures.config import ModelConfig
    from evallens.replay import ReplayBudget

DEFAULT_CONFIG = Path("configs/cpu.toml")


class ConfigError(ValueError):
    """Raised for a malformed configuration file."""


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    device: str = "cpu"
    dtype: str = "float32"
    threads: int = 4
    interop_threads: int = 1


@dataclass(frozen=True, slots=True)
class LimitSettings:
    max_tokens_per_request: int = 128
    max_batch_rows: int = 4
    max_session_requests: int = 3


@dataclass(frozen=True, slots=True)
class ResourceSettings:
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES
    case_timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    max_checkpoints: int = 4096
    max_values_per_checkpoint: int = 4096


@dataclass(frozen=True, slots=True)
class Settings:
    """The resolved configuration for one run."""

    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    tolerance: TolerancePolicy = field(default_factory=TolerancePolicy)
    limits: LimitSettings = field(default_factory=LimitSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    reduction: ReductionBudget = field(default_factory=ReductionBudget)
    fixture: str = "unit"
    stability_replays: int = 3
    source_path: str = ""

    @property
    def config_id(self) -> str:
        """Hash of everything that affects a run, recorded alongside every result."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": asdict(self.execution),
            "tolerance": self.tolerance.to_dict(),
            "limits": asdict(self.limits),
            "resources": asdict(self.resources),
            "capture": asdict(self.capture),
            "reduction": self.reduction.to_dict(),
            "fixture": self.fixture,
            "stability_replays": self.stability_replays,
        }

    @staticmethod
    def load(path: str | Path | None = None) -> Settings:
        """Load settings from TOML. With no path, returns the documented defaults."""
        if path is None:
            return Settings()
        source = Path(path)
        if not source.exists():
            raise ConfigError(f"configuration file not found: {source}")
        try:
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{source}: {exc}") from exc

        known_sections = {
            "execution",
            "tolerance",
            "limits",
            "resources",
            "capture",
            "reduction",
            "fixture",
            "stability",
        }
        unknown = set(raw) - known_sections
        if unknown:
            raise ConfigError(
                f"{source}: unknown section(s) {sorted(unknown)}; known sections are "
                f"{sorted(known_sections)}"
            )

        def section(name: str, cls: Any) -> Any:
            payload = raw.get(name, {})
            fields = set(cls.__dataclass_fields__)
            extra = set(payload) - fields
            if extra:
                raise ConfigError(f"{source}: unknown key(s) in [{name}]: {sorted(extra)}")
            return cls(**payload)

        tolerance_payload = dict(raw.get("tolerance", {}))
        tolerance_payload.pop("policy_id", None)
        allowed_tolerance = {"atol", "rtol", "zero_norm_eps", "name"}
        extra_tolerance = set(tolerance_payload) - allowed_tolerance
        if extra_tolerance:
            raise ConfigError(f"{source}: unknown key(s) in [tolerance]: {sorted(extra_tolerance)}")

        reduction_payload = raw.get("reduction", {})
        allowed_reduction = {"max_predicate_queries", "time_budget_s", "case_timeout_s"}
        extra_reduction = set(reduction_payload) - allowed_reduction
        if extra_reduction:
            raise ConfigError(f"{source}: unknown key(s) in [reduction]: {sorted(extra_reduction)}")

        stability = raw.get("stability", {})
        if set(stability) - {"replays"}:
            raise ConfigError(f"{source}: unknown key(s) in [stability]")

        resources = section("resources", ResourceSettings)
        return Settings(
            execution=section("execution", ExecutionSettings),
            tolerance=TolerancePolicy(**tolerance_payload)
            if tolerance_payload
            else TolerancePolicy(),
            limits=section("limits", LimitSettings),
            resources=resources,
            capture=section("capture", CaptureSettings),
            reduction=ReductionBudget(
                max_queries=int(reduction_payload.get("max_predicate_queries", 256)),
                time_budget_s=float(reduction_payload.get("time_budget_s", 60.0)),
                stability_replays=int(stability.get("replays", 3)),
                case_timeout_s=float(
                    reduction_payload.get("case_timeout_s", resources.case_timeout_s)
                ),
            ),
            fixture=str(raw.get("fixture", {}).get("name", "unit")),
            stability_replays=int(stability.get("replays", 3)),
            source_path=str(source),
        )

    def model_config(self) -> ModelConfig:
        """Resolve the named fixture to a `ModelConfig`."""
        from evallens.fixtures.config import SCALE_FIXTURE, UNIT_FIXTURE

        if self.fixture == "unit":
            return UNIT_FIXTURE
        if self.fixture == "scale":
            return SCALE_FIXTURE
        raise ConfigError(f"unknown fixture {self.fixture!r}; expected 'unit' or 'scale'")

    def capture_budget(self) -> CaptureBudget:
        from evallens.adapters.native import CaptureBudget

        return CaptureBudget(
            max_checkpoints=self.capture.max_checkpoints,
            max_values_per_checkpoint=self.capture.max_values_per_checkpoint,
        )

    def replay_budget(self) -> ReplayBudget:
        from evallens.replay import ReplayBudget

        return ReplayBudget(
            timeout_s=self.resources.case_timeout_s,
            rss_limit_bytes=self.resources.rss_limit_bytes,
        )


__all__ = [
    "DEFAULT_CONFIG",
    "CaptureSettings",
    "ConfigError",
    "ExecutionSettings",
    "LimitSettings",
    "ResourceSettings",
    "Settings",
]
