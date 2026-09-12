"""Benchmark harness: acquire raw evidence, and nothing else.

This module *measures*. It computes no rates, draws no conclusions, and writes no prose. Every
aggregate in `RESULTS.md` is derived by `bench/report.py` from the JSONL records written here,
so a number in the report can always be traced back to the trial that produced it.

Freezing
--------
The manifest is written **before** any trial runs, and it contains the complete set of
expected trial ids. The report then cross-checks what actually happened against what was
declared. That ordering is what makes "we ran the declared cohort" checkable rather than
asserted: a trial that errors, times out, or is never reached shows up as a missing id instead
of quietly vanishing from a denominator.

The reduction cohort is likewise selected and recorded before either reducer is constructed,
so neither can be credited with a cohort chosen to favor it.

Order bias
----------
Within each mutant/seed pair the two generators run in an order drawn from a recorded seed,
rather than always uniform-first. The machine is a laptop: it warms up, and a fixed order
would systematically hand one generator the colder cache and the other the hotter thermal
state.

Cost
----
Trials run sequentially, one worker. Diagnostic capture and localization happen on a separate
pass after a stable failure, never inside a detection timing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.controls import CONTROLS
from bench.mutants import MUTANTS, MutantSpec, build_mutant_adapter, qualify_mutant
from bench.seeds import CALIBRATION_SEEDS, DEVELOPMENT_SEEDS, EVALUATION_SEEDS
from evallens import __version__
from evallens.adapters.native import NativeAdapterSpec, ReferenceAdapter
from evallens.compare import compare
from evallens.env import capture_environment, numeric_identity
from evallens.export import ReproductionSpec, export_reproduction, verify_reproduction
from evallens.fixtures.config import ModelConfig, WeightDict, make_weights, weights_sha256
from evallens.generate import (
    BoundaryAwareGenerator,
    GeneratorCapability,
    UniformValidGenerator,
    classify_case,
)
from evallens.reduce import (
    FailurePredicate,
    PredicateCounters,
    ReductionBudget,
    reduce_case,
    signature_from_failure,
)
from evallens.replay import RunCounters, stable_comparison
from evallens.resources import ResourceGuard, set_deterministic_threads
from evallens.settings import Settings
from evallens.trace import trace_and_localize
from evallens.types import Case, TolerancePolicy, Verdict

RUN_SCHEMA_VERSION = 1
GENERATOR_NAMES = ("uniform-valid", "boundary-aware")
STRATEGIES = ("ddmin", "greedy")


@dataclass(frozen=True, slots=True)
class BenchPreset:
    """A declared workload. These are workload parameters, never performance claims."""

    name: str
    seeds: tuple[int, ...]
    cases_per_trial: int
    control_cases_per_generator: int
    reduction_cohort_per_family: int
    export_sample: int
    reduction_budget: ReductionBudget
    max_tokens: int = 128
    order_seed: int = 4242

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "n_seeds": len(self.seeds),
            "cases_per_trial": self.cases_per_trial,
            "control_cases_per_generator": self.control_cases_per_generator,
            "reduction_cohort_per_family": self.reduction_cohort_per_family,
            "export_sample": self.export_sample,
            "reduction_budget": self.reduction_budget.to_dict(),
            "max_tokens": self.max_tokens,
            "order_seed": self.order_seed,
            "generators": list(GENERATOR_NAMES),
            "strategies": list(STRATEGIES),
        }


SMOKE = BenchPreset(
    name="smoke",
    seeds=DEVELOPMENT_SEEDS[:2],
    cases_per_trial=16,
    control_cases_per_generator=24,
    reduction_cohort_per_family=1,
    export_sample=2,
    reduction_budget=ReductionBudget(max_queries=64, time_budget_s=20.0),
    max_tokens=48,
)
"""A fast pilot on *development* seeds. Used to estimate duration, never to publish."""

FULL = BenchPreset(
    name="full",
    seeds=EVALUATION_SEEDS[:5],
    cases_per_trial=64,
    control_cases_per_generator=256,
    reduction_cohort_per_family=2,
    export_sample=8,
    reduction_budget=ReductionBudget(max_queries=256, time_budget_s=60.0),
    max_tokens=128,
)
"""The declared CPU preset: 8 families x 2 variants x 5 held-out seeds x 2 generators."""

PRESETS: dict[str, BenchPreset] = {"smoke": SMOKE, "full": FULL}


# --- trial identities ------------------------------------------------------------------------


def detection_trial_id(mutant_id: str, generator: str, seed: int) -> str:
    return f"detect/{mutant_id}/{generator}/{seed}"


def control_trial_id(generator: str, seed: int) -> str:
    return f"control/{generator}/{seed}"


def reduction_trial_id(case_key: str, strategy: str) -> str:
    return f"reduce/{case_key}/{strategy}"


def export_trial_id(case_key: str) -> str:
    return f"export/{case_key}"


# --- the frozen manifest -----------------------------------------------------------------------


def expected_trial_ids(preset: BenchPreset, cohort_keys: Sequence[str]) -> list[str]:
    """Every trial the run is committed to producing. Declared before anything executes."""
    ids: list[str] = []
    for mutant in MUTANTS:
        for generator in GENERATOR_NAMES:
            for seed in preset.seeds:
                ids.append(detection_trial_id(mutant.mutant_id, generator, seed))
    for generator in GENERATOR_NAMES:
        for seed in preset.seeds:
            ids.append(control_trial_id(generator, seed))
    for key in cohort_keys:
        for strategy in STRATEGIES:
            ids.append(reduction_trial_id(key, strategy))
    return sorted(ids)


def _jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


# --- detection ---------------------------------------------------------------------------------


def _generator_for(
    name: str, capability: GeneratorCapability
) -> UniformValidGenerator | BoundaryAwareGenerator:
    return (
        UniformValidGenerator(capability)
        if name == "uniform-valid"
        else BoundaryAwareGenerator(capability)
    )


def run_detection_trial(
    mutant: MutantSpec,
    generator_name: str,
    seed: int,
    *,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    preset: BenchPreset,
    settings: Settings,
    capability: GeneratorCapability,
) -> dict[str, Any]:
    """Run one mutant against one generator at one seed, stopping at the first stable failure.

    An undetected trial is **budget-censored**, not omitted: it is recorded with
    ``detected=false`` and the full case budget it actually consumed.
    """
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)

    generation_started = time.perf_counter_ns()
    cases = _generator_for(generator_name, capability).generate(preset.cases_per_trial, seed)
    generation_ns = time.perf_counter_ns() - generation_started

    counters = RunCounters()
    verdict_counts: dict[str, int] = {}
    detected = False
    cases_examined = 0
    detection_case: Case | None = None
    detection_detail = ""
    max_abs_err = 0.0
    guard = ResourceGuard(settings.resources.rss_limit_bytes, raise_on_exceed=False)

    comparison_started = time.perf_counter_ns()
    for case in cases:
        cases_examined += 1
        outcome = stable_comparison(
            reference,
            candidate,
            case,
            policy,
            replays=settings.stability_replays,
            budget=settings.replay_budget(),
            counters=counters,
        )
        verdict_counts[outcome.verdict.value] = verdict_counts.get(outcome.verdict.value, 0) + 1
        if outcome.verdict is Verdict.FAIL and outcome.stable:
            detected = True
            detection_case = case
            detection_detail = outcome.representative.detail
            max_abs_err = outcome.representative.max_abs_err
            break
    comparison_ns = time.perf_counter_ns() - comparison_started
    guard.sample()

    # Localization runs only after a stable failure, on a separate traced pass, and its cost
    # is recorded separately so it never enters the detection timing.
    localization: dict[str, Any] | None = None
    localization_ns = 0
    if detection_case is not None:
        localization_started = time.perf_counter_ns()
        result = trace_and_localize(reference, candidate, detection_case, policy)
        localization_ns = time.perf_counter_ns() - localization_started
        localization = {
            "available": result.available,
            "reason": result.reason,
            "earliest_observed": (
                result.earliest_observed.as_str() if result.earliest_observed else None
            ),
            "n_divergent": len(result.divergent),
            "n_compared": len(result.comparisons),
            "reconverged": result.reconverged,
            "fully_aligned": result.alignment.fully_aligned,
        }

    return {
        "kind": "detection",
        "trial_id": detection_trial_id(mutant.mutant_id, generator_name, seed),
        "mutant_id": mutant.mutant_id,
        "family_key": mutant.family_key,
        "family_number": mutant.family.number,
        "variant": mutant.variant,
        "injected_fault": True,
        "generator": generator_name,
        "seed": seed,
        "case_budget": preset.cases_per_trial,
        "detected": detected,
        "budget_censored": not detected,
        "cases_examined": cases_examined,
        "cases_unused": preset.cases_per_trial - cases_examined,
        "verdict_counts": verdict_counts,
        "detection_case_id": detection_case.case_id if detection_case else None,
        "detection_case_category": classify_case(detection_case).value if detection_case else None,
        "detection_detail": detection_detail,
        "max_abs_err": max_abs_err,
        "comparison_ns": comparison_ns,
        "comparison_s": comparison_ns / 1e9,
        "generation_ns": generation_ns,
        "localization_ns": localization_ns,
        "localization": localization,
        "model_runs": counters.model_runs,
        "errors": counters.errors,
        "peak_rss_bytes": guard.peak_rss_bytes,
    }


# --- controls ------------------------------------------------------------------------------------


def run_control_trial(
    generator_name: str,
    seed: int,
    *,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    preset: BenchPreset,
    settings: Settings,
    capability: GeneratorCapability,
) -> dict[str, Any]:
    """Known-good comparisons on generated cases: the false-positive denominator.

    The candidate here is a correct implementation. Any stable FAIL is a false positive and is
    recorded as one, with the offending case id kept so it can be investigated rather than
    averaged away.
    """
    from evallens.adapters.native import CandidateAdapter

    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights)  # Behavior() — the known-good cached path
    cases = _generator_for(generator_name, capability).generate(
        preset.control_cases_per_generator, seed
    )

    counters = RunCounters()
    verdict_counts: dict[str, int] = {}
    false_positives: list[str] = []
    per_category: dict[str, dict[str, int]] = {}

    started = time.perf_counter_ns()
    for case in cases:
        outcome = stable_comparison(
            reference,
            candidate,
            case,
            policy,
            replays=settings.stability_replays,
            budget=settings.replay_budget(),
            counters=counters,
        )
        verdict = outcome.verdict.value
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        category = classify_case(case).value
        per_category.setdefault(category, {})
        per_category[category][verdict] = per_category[category].get(verdict, 0) + 1
        if outcome.verdict is Verdict.FAIL and outcome.stable:
            false_positives.append(case.case_id)
    elapsed = time.perf_counter_ns() - started

    return {
        "kind": "control",
        "trial_id": control_trial_id(generator_name, seed),
        "generator": generator_name,
        "seed": seed,
        "injected_fault": False,
        "n_cases": len(cases),
        "verdict_counts": verdict_counts,
        "per_category": per_category,
        "false_positive_case_ids": false_positives,
        "n_false_positives": len(false_positives),
        "elapsed_ns": elapsed,
        "model_runs": counters.model_runs,
        "errors": counters.errors,
    }


def run_fixed_controls(
    config: ModelConfig, weights: WeightDict, policy: TolerancePolicy, settings: Settings
) -> list[dict[str, Any]]:
    """The handwritten known-good corpus, independent of any generator."""
    records = []
    for control in CONTROLS:
        case = control.build_case(config, weights)
        reference, candidate = control.build_adapters(config, weights)
        outcome = stable_comparison(
            reference, candidate, case, policy, replays=settings.stability_replays
        )
        records.append(
            {
                "kind": "fixed_control",
                "trial_id": f"fixed_control/{control.control_id}",
                "control_id": control.control_id,
                "injected_fault": False,
                "description": control.description,
                "exercises": control.exercises,
                "verdict": outcome.verdict.value,
                "stable": outcome.stable,
                "max_abs_err": outcome.representative.max_abs_err,
                "detail": outcome.representative.detail,
                "false_positive": outcome.verdict is Verdict.FAIL and outcome.stable,
            }
        )
    return records


# --- the frozen reduction cohort -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortEntry:
    """One starting case for the paired reducer comparison."""

    key: str
    mutant_id: str
    family_key: str
    case: Case

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "mutant_id": self.mutant_id,
            "family_key": self.family_key,
            "case_id": self.case.case_id,
            "case": self.case.to_dict(),
            "n_valid_tokens": self.case.total_valid_tokens,
            "n_requests": len(self.case.requests),
            "category": classify_case(self.case).value,
        }


def select_reduction_cohort(
    preset: BenchPreset,
    *,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    settings: Settings,
) -> list[CohortEntry]:
    """Choose the paired-comparison starting cases **before** either reducer exists.

    Selection is by a fixed rule, not by inspecting reducer behavior: for each variant, take
    the first cases from a dedicated cohort seed that produce a stable failure. A cohort chosen
    after seeing results could favor either method, so it is frozen into the manifest.
    """
    cohort: list[CohortEntry] = []
    per_family: dict[str, int] = {}
    capability = GeneratorCapability.for_fixture(
        config, weights_sha256(weights), max_tokens=preset.max_tokens
    )
    reference = ReferenceAdapter(config, weights)

    for mutant in MUTANTS:
        if per_family.get(mutant.family_key, 0) >= preset.reduction_cohort_per_family:
            continue
        candidate = build_mutant_adapter(mutant, config, weights)
        # A dedicated calibration seed: the cohort is deliberately disjoint from the held-out
        # evaluation seeds used for detection.
        cases = UniformValidGenerator(capability).generate(48, CALIBRATION_SEEDS[0])
        for case in cases:
            outcome = stable_comparison(
                reference, candidate, case, policy, replays=settings.stability_replays
            )
            if outcome.verdict is Verdict.FAIL and outcome.stable:
                cohort.append(
                    CohortEntry(
                        key=f"{mutant.mutant_id}#{case.case_id[:12]}",
                        mutant_id=mutant.mutant_id,
                        family_key=mutant.family_key,
                        case=case,
                    )
                )
                per_family[mutant.family_key] = per_family.get(mutant.family_key, 0) + 1
                break
    return cohort


def run_reduction_trial(
    entry: CohortEntry,
    strategy: str,
    *,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    preset: BenchPreset,
    settings: Settings,
) -> dict[str, Any]:
    """Reduce one cohort case with one strategy, under the shared budget."""
    mutant = next(m for m in MUTANTS if m.mutant_id == entry.mutant_id)
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)

    outcome = stable_comparison(
        reference, candidate, entry.case, policy, replays=settings.stability_replays
    )
    signature = signature_from_failure(entry.case, outcome.representative.failing_request_ids)
    predicate = FailurePredicate(
        reference,
        candidate,
        policy,
        signature,
        budget=preset.reduction_budget,
        counters=PredicateCounters(),
    )
    result = reduce_case(entry.case, predicate, strategy=strategy, budget=preset.reduction_budget)

    # Independent confirmation that the answer still fails.
    verification = stable_comparison(
        reference, candidate, result.reduced, policy, replays=settings.stability_replays
    )

    return {
        "kind": "reduction",
        "trial_id": reduction_trial_id(entry.key, strategy),
        "cohort_key": entry.key,
        "mutant_id": entry.mutant_id,
        "family_key": entry.family_key,
        "injected_fault": True,
        "strategy": strategy,
        "original_case_id": entry.case.case_id,
        "reduced_case_id": result.reduced.case_id,
        "original_tokens": entry.case.total_valid_tokens,
        "reduced_tokens": result.reduced.total_valid_tokens,
        "original_requests": len(entry.case.requests),
        "reduced_requests": len(result.reduced.requests),
        "original_padding": entry.case.total_padding_tokens,
        "reduced_padding": result.reduced.total_padding_tokens,
        "token_reduction_ratio": result.token_reduction_ratio,
        "minimality": result.minimality.value,
        "budget_exhausted": result.budget_exhausted,
        "counters": result.counters.to_dict(),
        "wall_time_ns": result.wall_time_ns,
        "n_accepted_steps": len(result.accepted_steps),
        "still_fails": verification.verdict is Verdict.FAIL and verification.stable,
        "verification_verdict": verification.verdict.value,
        "reduced_case": result.reduced.to_dict(),
    }


# --- exports -------------------------------------------------------------------------------------------


def run_export_trial(
    entry: CohortEntry,
    reduced_case: Case,
    *,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    out_dir: Path,
) -> dict[str, Any]:
    """Export a reduced case and verify it in a clean room."""
    mutant = next(m for m in MUTANTS if m.mutant_id == entry.mutant_id)
    reference = ReferenceAdapter(config, weights)
    candidate = build_mutant_adapter(mutant, config, weights)
    comparison = compare(reference.run(reduced_case), candidate.run(reduced_case), policy)

    spec = ReproductionSpec(
        case=reduced_case,
        reference=NativeAdapterSpec("reference", config),
        candidate=NativeAdapterSpec("candidate", config, mutant.behavior, mutant.mutant_id),
        policy=policy,
        signature=signature_from_failure(reduced_case, comparison.failing_request_ids),
        comparison=comparison,
        injected_fault=True,
        fault_description=mutant.description,
        original_case=entry.case,
    )
    package = export_reproduction(spec, out_dir / entry.key.replace("/", "_").replace("#", "_"))
    verification = verify_reproduction(package, expect_mismatch=True)

    return {
        "kind": "export",
        "trial_id": export_trial_id(entry.key),
        "cohort_key": entry.key,
        "mutant_id": entry.mutant_id,
        "family_key": entry.family_key,
        "injected_fault": True,
        "case_id": reduced_case.case_id,
        "reproduced": verification.reproduced,
        "ok": verification.ok,
        "exit_code": verification.exit_code,
        "expected_exit_code": verification.expected_exit_code,
        "imported_evallens": verification.imported_evallens,
        "elapsed_s": verification.elapsed_s,
    }


# --- the run ------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class RunPaths:
    root: Path
    manifest: Path = field(init=False)
    detection: Path = field(init=False)
    controls: Path = field(init=False)
    reductions: Path = field(init=False)
    exports: Path = field(init=False)
    summary: Path = field(init=False)
    exports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.json"
        self.detection = self.root / "detection.jsonl"
        self.controls = self.root / "controls.jsonl"
        self.reductions = self.root / "reductions.jsonl"
        self.exports = self.root / "exports.jsonl"
        self.summary = self.root / "summary.json"
        self.exports_dir = self.root / "repros"


def run_benchmark(
    preset: BenchPreset,
    out_dir: str | Path,
    *,
    settings: Settings | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Execute a preset and write raw evidence. Computes no rates."""
    settings = settings or Settings()
    paths = RunPaths(Path(out_dir).resolve())
    for path in (paths.detection, paths.controls, paths.reductions, paths.exports):
        path.unlink(missing_ok=True)

    thread_report = set_deterministic_threads(settings.execution.threads)
    environment = capture_environment()
    config = settings.model_config()

    init_started = time.perf_counter_ns()
    weights = make_weights(config)
    digest = weights_sha256(weights)
    ReferenceAdapter(config, weights)  # pay model construction once, and record what it cost
    init_ns = time.perf_counter_ns() - init_started

    policy = settings.tolerance
    capability = GeneratorCapability.for_fixture(config, digest, max_tokens=preset.max_tokens)

    # Qualify every variant on its own trigger before anything is measured. A variant that
    # stops qualifying must be visible as incomplete, never silently dropped.
    qualification = [qualify_mutant(m, config, weights, policy).to_dict() for m in MUTANTS]
    cohort = select_reduction_cohort(
        preset, config=config, weights=weights, policy=policy, settings=settings
    )
    cohort_keys = [entry.key for entry in cohort]

    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": f"{preset.name}-{time.strftime('%Y%m%d-%H%M%S')}",
        "evallens_version": __version__,
        "preset": preset.to_dict(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": config.to_dict(),
        "weights_sha256": digest,
        "policy": policy.to_dict(),
        "settings_id": settings.config_id,
        "settings": settings.to_dict(),
        "environment": environment.to_dict(),
        "environment_numeric_id": numeric_identity(),
        "threads": thread_report,
        "source_commit": environment.git_commit,
        "source_clean": environment.git_dirty is False,
        "publishable": environment.git.publishable,
        "seed_sets": {
            "calibration": list(CALIBRATION_SEEDS),
            "development": list(DEVELOPMENT_SEEDS),
            "evaluation": list(EVALUATION_SEEDS),
        },
        "mutant_manifest": [m.to_dict() for m in MUTANTS],
        "qualification": qualification,
        "qualified_mutant_ids": [q["mutant_id"] for q in qualification if q["qualified"]],
        "unqualified_mutant_ids": [q["mutant_id"] for q in qualification if not q["qualified"]],
        "control_manifest": [c.to_dict() for c in CONTROLS],
        "generator_versions": {
            "uniform-valid": UniformValidGenerator.version,
            "boundary-aware": BoundaryAwareGenerator.version,
        },
        "reduction_cohort": [entry.to_dict() for entry in cohort],
        "expected_trial_ids": expected_trial_ids(preset, cohort_keys),
        "model_init_ns": init_ns,
        "synthetic": False,
        "injected_faults_note": (
            "Every fault in this run was injected by EvalLens on purpose as test material for "
            "its own detector. None is a bug discovered in PyTorch or any other library."
        ),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def say(message: str) -> None:
        if progress:
            print(message, flush=True)

    say(f"run {manifest['run_id']}: {len(manifest['expected_trial_ids'])} declared trials")
    say(f"  qualified variants: {len(manifest['qualified_mutant_ids'])}/{len(MUTANTS)}")
    say(f"  reduction cohort:   {len(cohort)} cases")

    started = time.perf_counter_ns()
    guard = ResourceGuard(settings.resources.rss_limit_bytes, raise_on_exceed=False)

    # -- detection, alternating generator order to reduce order bias ------------------------
    order_rng = np.random.default_rng(preset.order_seed)
    completed = 0
    total = len(manifest["expected_trial_ids"])
    for mutant in MUTANTS:
        for seed in preset.seeds:
            names = list(GENERATOR_NAMES)
            if bool(order_rng.integers(0, 2)):
                names.reverse()
            for generator_name in names:
                record = run_detection_trial(
                    mutant,
                    generator_name,
                    seed,
                    config=config,
                    weights=weights,
                    policy=policy,
                    preset=preset,
                    settings=settings,
                    capability=capability,
                )
                record["generator_order"] = names
                _jsonl(paths.detection, record)
                guard.sample()
                completed += 1
        say(f"  [{completed}/{total}] detection: {mutant.mutant_id}")

    # -- controls --------------------------------------------------------------------------------
    for record in run_fixed_controls(config, weights, policy, settings):
        _jsonl(paths.controls, record)
    for generator_name in GENERATOR_NAMES:
        for seed in preset.seeds:
            record = run_control_trial(
                generator_name,
                seed,
                config=config,
                weights=weights,
                policy=policy,
                preset=preset,
                settings=settings,
                capability=capability,
            )
            _jsonl(paths.controls, record)
            guard.sample()
            completed += 1
        say(f"  [{completed}/{total}] controls: {generator_name}")

    # -- paired reduction on the frozen cohort ------------------------------------------------------
    reduced_by_key: dict[str, Case] = {}
    for entry in cohort:
        for strategy in STRATEGIES:
            record = run_reduction_trial(
                entry,
                strategy,
                config=config,
                weights=weights,
                policy=policy,
                preset=preset,
                settings=settings,
            )
            _jsonl(paths.reductions, record)
            if strategy == "ddmin":
                reduced_by_key[entry.key] = Case.from_dict(record["reduced_case"])
            guard.sample()
            completed += 1
        say(f"  [{completed}/{total}] reduction: {entry.key}")

    # -- exports ------------------------------------------------------------------------------------
    for entry in cohort[: preset.export_sample]:
        reduced = reduced_by_key.get(entry.key)
        if reduced is None:
            continue
        record = run_export_trial(
            entry,
            reduced,
            config=config,
            weights=weights,
            policy=policy,
            out_dir=paths.exports_dir,
        )
        _jsonl(paths.exports, record)
        guard.sample()
    say(f"  exports: {min(len(cohort), preset.export_sample)} attempted")

    elapsed_ns = time.perf_counter_ns() - started
    summary = {
        "run_id": manifest["run_id"],
        "preset": preset.name,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_ns": elapsed_ns,
        "wall_time_s": elapsed_ns / 1e9,
        "model_init_ns": init_ns,
        "peak_rss_bytes": guard.peak_rss_bytes,
        "rss_limit_bytes": settings.resources.rss_limit_bytes,
        "rss_exceeded": guard.exceeded,
        "trials_written": completed,
        "trials_declared": total,
        "complete": completed == total,
    }
    paths.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    say(f"done in {summary['wall_time_s']:.1f}s — {paths.root}")
    return summary


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run an EvalLens benchmark preset and write raw JSONL evidence. Computes no "
            "rates; use bench/report.py to derive RESULTS.md from these records."
        )
    )
    parser.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    parser.add_argument("--out", required=True, help="Run directory.")
    parser.add_argument("--config", default=None, help="TOML configuration file.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings.load(args.config) if args.config else Settings()
    summary = run_benchmark(
        PRESETS[args.preset], args.out, settings=settings, progress=not args.quiet
    )
    return 0 if summary["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
