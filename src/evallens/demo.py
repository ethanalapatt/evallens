"""The one-command demo: detect, localize, reduce, export, verify.

`evallens demo` runs the whole pipeline against a **deliberately injected** cache fault and
writes real artifacts. Every number it reports is measured during the run; nothing here is
illustrative.

The fault is constructed directly from `Behavior`, not imported from `bench.mutants`, so the
demo works from a plain `pip install` with no benchmark corpus present. It is labeled as
injected in the record, in the exported package, and in the terminal output.

The generator is never told which fault is running. It emits valid cases from the declared
space and the comparison decides; the demo stops at the **first** stable failure and reduces
that one. Scanning for the largest failure would make the reduction ratio look better and
would be cherry-picking.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evallens.adapters.native import CandidateAdapter, NativeAdapterSpec, ReferenceAdapter
from evallens.compare import compare
from evallens.env import capture_environment
from evallens.export import ReproductionSpec, export_reproduction, verify_reproduction
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import make_weights, weights_sha256
from evallens.generate import GeneratorCapability, UniformValidGenerator, classify_case
from evallens.reduce import (
    FailurePredicate,
    PredicateCounters,
    reduce_case,
    signature_from_failure,
)
from evallens.replay import RunCounters, replay_in_subprocess, stable_comparison
from evallens.resources import ResourceGuard, set_deterministic_threads
from evallens.settings import Settings
from evallens.trace import trace_and_localize
from evallens.types import Case, Verdict

DEMO_FAULT = Behavior(cache_index="write_overwrite_last")
DEMO_FAULT_DESCRIPTION = (
    "During incremental decoding the candidate writes each new key/value pair over the most "
    "recent cache slot instead of extending the cache, so one earlier position is permanently "
    "lost. Shapes stay valid and nothing raises; only the numbers change."
)
DEMO_SEED = 20260911


class DemoError(RuntimeError):
    """Raised when the demo cannot complete — never swallowed into a fake success."""


@dataclass(slots=True)
class DemoStep:
    name: str
    detail: str
    elapsed_s: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detail": self.detail,
            "elapsed_s": round(self.elapsed_s, 4),
            "data": self.data,
        }


@dataclass(slots=True)
class DemoResult:
    run_id: str
    out_dir: Path
    record: dict[str, Any]
    steps: list[DemoStep]
    succeeded: bool
    failure_reason: str = ""

    @property
    def record_path(self) -> Path:
        return self.out_dir / "record.json"

    @property
    def repro_dir(self) -> Path:
        return self.out_dir / "repro"


def run_demo(
    out_dir: str | Path,
    *,
    settings: Settings | None = None,
    seed: int = DEMO_SEED,
    max_cases: int = 64,
    verify_export: bool = True,
) -> DemoResult:
    """Run the full pipeline and write real artifacts to ``out_dir``."""
    settings = settings or Settings()
    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    thread_report = set_deterministic_threads(settings.execution.threads)
    guard = ResourceGuard(settings.resources.rss_limit_bytes, raise_on_exceed=False)
    environment = capture_environment()
    run_id = f"demo-{time.strftime('%Y%m%d-%H%M%S')}"
    steps: list[DemoStep] = []

    config = settings.model_config()
    weights = make_weights(config)
    digest = weights_sha256(weights)
    reference = ReferenceAdapter(config, weights, capture_budget=settings.capture_budget())
    candidate = CandidateAdapter(
        config,
        weights,
        DEMO_FAULT,
        capture_budget=settings.capture_budget(),
        label="demo-injected-cache-fault",
    )

    def step(name: str) -> Any:
        started = time.perf_counter_ns()

        def finish(detail: str, data: dict[str, Any] | None = None) -> None:
            guard.sample()
            steps.append(
                DemoStep(name, detail, (time.perf_counter_ns() - started) / 1e9, data or {})
            )

        return finish

    # -- 1. search -------------------------------------------------------------------------
    finish = step("search")
    # The generator gets the configured token ceiling, not a smaller demo-specific one. The
    # demo then reduces whatever the *first* stable failure turns out to be — selecting the
    # largest failure would make the reduction ratio look better and would be cherry-picking.
    capability = GeneratorCapability.for_fixture(
        config, digest, max_tokens=settings.limits.max_tokens_per_request
    )
    # The uniform-valid baseline, not the boundary-aware generator. Both find this fault, but
    # the boundary generator deliberately concentrates on very short and very long sequences,
    # so the first failure it hits is usually a 2-4 token edge case. A demo should show what
    # reduction does to a *typical* failing input. Which generator finds more bugs under budget
    # is a benchmark question (M7), not a demo question.
    generator = UniformValidGenerator(capability)
    cases = generator.generate(max_cases, seed)

    counters = RunCounters()
    found: Case | None = None
    detection_index = -1
    detection_result = None
    search_started = time.perf_counter_ns()
    for index, case in enumerate(cases):
        outcome = stable_comparison(
            reference,
            candidate,
            case,
            settings.tolerance,
            replays=settings.stability_replays,
            budget=settings.replay_budget(),
            counters=counters,
        )
        if outcome.verdict is Verdict.FAIL and outcome.stable:
            found, detection_index, detection_result = case, index, outcome.representative
            break
    search_elapsed = (time.perf_counter_ns() - search_started) / 1e9

    if found is None or detection_result is None:
        finish(f"no stable failure in {len(cases)} cases", {"cases_examined": len(cases)})
        return DemoResult(
            run_id,
            destination,
            {},
            steps,
            succeeded=False,
            failure_reason=f"the generator produced no stable failure within {len(cases)} cases",
        )

    finish(
        f"stable failure on case {detection_index + 1} of {len(cases)} "
        f"({classify_case(found).value}), max |Δ| = {detection_result.max_abs_err:.3e}",
        {
            "cases_examined": detection_index + 1,
            "cases_available": len(cases),
            "case_id": found.case_id,
            "category": classify_case(found).value,
            "max_abs_err": detection_result.max_abs_err,
            "seconds_to_detection": round(search_elapsed, 4),
            "model_runs": counters.model_runs,
        },
    )

    # -- 2. localize (a separate traced pass, never inside the detection timing) -------------
    finish = step("localize")
    localization = trace_and_localize(reference, candidate, found, settings.tolerance)
    finish(localization.summary(), localization.to_dict(max_comparisons=64))

    # -- 3. reduce ---------------------------------------------------------------------------
    finish = step("reduce")
    signature = signature_from_failure(found, detection_result.failing_request_ids)
    predicate = FailurePredicate(
        reference,
        candidate,
        settings.tolerance,
        signature,
        budget=settings.reduction,
        counters=PredicateCounters(),
    )
    reduction = reduce_case(found, predicate, strategy="ddmin", budget=settings.reduction)
    finish(
        f"{found.total_valid_tokens} -> {reduction.reduced.total_valid_tokens} valid tokens "
        f"({reduction.token_reduction_ratio:.1f}x) in "
        f"{reduction.counters.logical_queries} predicate queries; "
        f"{reduction.minimality.value}",
        reduction.to_dict(),
    )

    # -- 4. confirm the reduced case in a fresh process ----------------------------------------
    finish = step("verify_reduced_case")
    reference_spec = NativeAdapterSpec("reference", config)
    candidate_spec = NativeAdapterSpec("candidate", config, DEMO_FAULT, "demo-injected-cache-fault")
    subprocess_replay = replay_in_subprocess(
        reduction.reduced,
        reference_spec,
        candidate_spec,
        settings.tolerance,
        replays=settings.stability_replays,
    )
    if not subprocess_replay.reproduced_failure:
        finish("the reduced case did NOT reproduce in a fresh process", subprocess_replay.to_dict())
        return DemoResult(
            run_id,
            destination,
            {},
            steps,
            succeeded=False,
            failure_reason=(
                "the reduced case did not reproduce in a fresh subprocess: "
                f"{subprocess_replay.detail}"
            ),
        )
    finish(
        f"reproduced in a fresh process ({subprocess_replay.elapsed_s:.2f}s)",
        subprocess_replay.to_dict(),
    )

    # -- 5. export ------------------------------------------------------------------------------
    finish = step("export")
    final_comparison = compare(
        reference.run(reduction.reduced), candidate.run(reduction.reduced), settings.tolerance
    )
    reduced_localization = trace_and_localize(
        reference, candidate, reduction.reduced, settings.tolerance
    )
    spec = ReproductionSpec(
        case=reduction.reduced,
        reference=reference_spec,
        candidate=candidate_spec,
        policy=settings.tolerance,
        signature=signature,
        comparison=final_comparison,
        injected_fault=True,
        fault_description=DEMO_FAULT_DESCRIPTION,
        original_case=found,
        localization_summary=reduced_localization.summary(),
        reduction_summary={
            "original_tokens": found.total_valid_tokens,
            "reduced_tokens": reduction.reduced.total_valid_tokens,
            "original_requests": len(found.requests),
            "reduced_requests": len(reduction.reduced.requests),
            "ratio": reduction.token_reduction_ratio,
            "logical_queries": reduction.counters.logical_queries,
            "minimality": reduction.minimality.value,
        },
    )
    repro_dir = destination / "repro"
    export_reproduction(spec, repro_dir)
    finish(f"wrote a self-contained package to {repro_dir.name}/", {"path": str(repro_dir)})

    # -- 6. verify the export from a clean temporary directory -------------------------------------
    finish = step("verify_export")
    verification = None
    if verify_export:
        verification = verify_reproduction(repro_dir, expect_mismatch=True)
        if not verification.ok:
            finish("the exported package did NOT reproduce", verification.to_dict())
            return DemoResult(
                run_id,
                destination,
                {},
                steps,
                succeeded=False,
                failure_reason=(
                    f"exported package exited {verification.exit_code}, expected "
                    f"{verification.expected_exit_code}: {verification.stderr[-300:]}"
                ),
            )
        finish(
            f"ran from a fresh temporary directory without importing EvalLens "
            f"({verification.elapsed_s:.2f}s)",
            verification.to_dict(),
        )
    else:
        finish("skipped by request", {})

    # -- 7. the viewer record -----------------------------------------------------------------------
    report = guard.report()
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": "demo",
        "injected_fault": True,
        "fault_banner": (
            "This example uses a fault EvalLens injected on purpose. It is not a bug "
            "discovered in PyTorch or any other third-party library."
        ),
        "fault_description": DEMO_FAULT_DESCRIPTION,
        "candidate_behavior": DEMO_FAULT.to_dict(),
        "candidate_behavior_description": DEMO_FAULT.describe(),
        "reference_adapter": reference.adapter_id,
        "candidate_adapter": candidate.adapter_id,
        "model_config": config.to_dict(),
        "weights_sha256": digest,
        "policy": settings.tolerance.to_dict(),
        "settings_id": settings.config_id,
        "generator": {
            "name": generator.name,
            "version": generator.version,
            "seed": seed,
            "budget": max_cases,
        },
        "original_case": found.to_dict(),
        "reduced_case": reduction.reduced.to_dict(),
        "comparison": final_comparison.to_dict(),
        "detection": steps[0].to_dict(),
        "localization": reduced_localization.to_dict(max_comparisons=64),
        "reduction": reduction.to_dict(),
        "subprocess_replay": subprocess_replay.to_dict(),
        "export": {
            "path": "repro",
            "verified": bool(verification.ok) if verification else None,
            "verification": verification.to_dict() if verification else None,
        },
        "steps": [s.to_dict() for s in steps],
        "resources": report.to_dict(),
        "threads": thread_report,
        "environment": environment.to_dict(),
    }

    (destination / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    (destination / "case.json").write_text(json.dumps(found.to_dict(), indent=2, sort_keys=True))
    (destination / "failure.json").write_text(
        json.dumps(
            {
                "case": found.to_dict(),
                "signature": signature.to_dict(),
                "comparison": detection_result.to_dict(),
                "reference": reference_spec.to_dict(),
                "candidate": candidate_spec.to_dict(),
                "policy": settings.tolerance.to_dict(),
                "injected_fault": True,
                "fault_description": DEMO_FAULT_DESCRIPTION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (destination / "reduced.json").write_text(
        json.dumps(
            {
                "case": reduction.reduced.to_dict(),
                "signature": signature.to_dict(),
                "reference": reference_spec.to_dict(),
                "candidate": candidate_spec.to_dict(),
                "policy": settings.tolerance.to_dict(),
                "injected_fault": True,
                "fault_description": DEMO_FAULT_DESCRIPTION,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return DemoResult(run_id, destination, record, steps, succeeded=True)


__all__ = [
    "DEMO_FAULT",
    "DEMO_FAULT_DESCRIPTION",
    "DEMO_SEED",
    "DemoError",
    "DemoResult",
    "DemoStep",
    "run_demo",
]
