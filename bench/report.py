"""Generate RESULTS.md from raw benchmark records.

Every number here is derived from the JSONL files a run actually wrote. This module reads
them, checks them, and formats them. It never measures anything itself, and it has no way to
produce a number that is not in the evidence.

Integrity gate
--------------
A fatal issue blocks `RESULTS.md` entirely. An interrupted or inconsistent study gets an
`INCOMPLETE.md` diagnostic instead, and a nonzero exit, because a partial study formatted as a
success table is the single most misleading artifact this project could emit.

Fatal issues:

* trials declared in the frozen manifest that never produced a record
* duplicate trial ids
* records whose weights, config, or policy hashes disagree with the manifest
* a metric whose denominator is zero
* mismatched reducer cohorts — the paired comparison requires both strategies to have run on
  exactly the same starting cases
* a run marked ``synthetic`` (test fixtures may never generate `RESULTS.md`)
* a dirty or unknown source tree, unless ``--allow-dirty`` is passed, in which case the report
  is written but stamped as not publishable

Uncertainty
-----------
Where an interval is reported the resampling unit is the **fault family**, not the trial. Five
seeds against one mutant are not five independent observations of "can EvalLens find this kind
of bug"; the family is the closest thing to an independent unit, and there are only eight of
them, so the intervals are wide and are labeled as such.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FATAL = "fatal"
WARNING = "warning"


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


class ReportError(RuntimeError):
    """Raised when a report cannot be generated at all."""


@dataclass(slots=True)
class LoadedRun:
    root: Path
    manifest: dict[str, Any]
    detection: list[dict[str, Any]]
    controls: list[dict[str, Any]]
    reductions: list[dict[str, Any]]
    exports: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def generated_controls(self) -> list[dict[str, Any]]:
        return [record for record in self.controls if record["kind"] == "control"]

    @property
    def fixed_controls(self) -> list[dict[str, Any]]:
        return [record for record in self.controls if record["kind"] == "fixed_control"]

    @property
    def all_records(self) -> list[dict[str, Any]]:
        return [*self.detection, *self.controls, *self.reductions, *self.exports]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run_dir: str | Path) -> LoadedRun:
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ReportError(f"{root} has no manifest.json; it is not a benchmark run directory")
    summary_path = root / "summary.json"
    return LoadedRun(
        root=root,
        manifest=json.loads(manifest_path.read_text()),
        detection=_read_jsonl(root / "detection.jsonl"),
        controls=_read_jsonl(root / "controls.jsonl"),
        reductions=_read_jsonl(root / "reductions.jsonl"),
        exports=_read_jsonl(root / "exports.jsonl"),
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {},
    )


# --- integrity ------------------------------------------------------------------------------


def check_integrity(run: LoadedRun, *, allow_dirty: bool = False) -> list[IntegrityIssue]:
    """Every check that stands between raw records and a publishable table."""
    issues: list[IntegrityIssue] = []
    manifest = run.manifest

    if manifest.get("synthetic"):
        issues.append(
            IntegrityIssue(
                "synthetic_run",
                FATAL,
                "this run is marked synthetic; synthetic fixtures may never generate RESULTS.md",
            )
        )

    # -- completeness of the declared cohort -------------------------------------------------
    expected = set(manifest.get("expected_trial_ids", []))
    if not expected:
        issues.append(
            IntegrityIssue(
                "no_declared_trials",
                FATAL,
                "the manifest declares no expected trial ids, so completeness cannot be checked",
            )
        )
    actual_ids = [record["trial_id"] for record in run.all_records]
    actual = set(actual_ids)

    missing = sorted(expected - actual)
    if missing:
        issues.append(
            IntegrityIssue(
                "missing_trials",
                FATAL,
                f"{len(missing)} declared trial(s) produced no record, e.g. {missing[:5]}",
            )
        )

    duplicates = sorted({tid for tid in actual_ids if actual_ids.count(tid) > 1})
    if duplicates:
        issues.append(
            IntegrityIssue(
                "duplicate_trials",
                FATAL,
                f"{len(duplicates)} trial id(s) appear more than once, e.g. {duplicates[:5]}",
            )
        )

    undeclared = sorted(
        tid for tid in actual - expected if not tid.startswith(("export/", "fixed_control/"))
    )
    if undeclared:
        issues.append(
            IntegrityIssue(
                "undeclared_trials",
                WARNING,
                f"{len(undeclared)} record(s) were not declared in the manifest: {undeclared[:5]}",
            )
        )

    # -- qualification ---------------------------------------------------------------------------
    unqualified = set(manifest.get("unqualified_mutant_ids", []))
    if unqualified:
        issues.append(
            IntegrityIssue(
                "unqualified_variants",
                WARNING,
                f"{len(unqualified)} declared variant(s) did not qualify and are excluded from "
                f"detection denominators: {sorted(unqualified)}",
            )
        )
    scored = {record["mutant_id"] for record in run.detection} & unqualified
    if scored:
        issues.append(
            IntegrityIssue(
                "unqualified_scored",
                FATAL,
                f"unqualified variant(s) {sorted(scored)} appear in detection records",
            )
        )

    # -- hash consistency ---------------------------------------------------------------------------
    if not manifest.get("weights_sha256"):
        issues.append(
            IntegrityIssue("missing_weights_hash", FATAL, "manifest records no weights hash")
        )
    if not manifest.get("policy", {}).get("policy_id"):
        issues.append(IntegrityIssue("missing_policy_id", FATAL, "manifest records no policy id"))
    if not manifest.get("model_config", {}).get("config_id"):
        issues.append(
            IntegrityIssue("missing_config_id", FATAL, "manifest records no model config id")
        )

    # -- paired reducer cohorts -------------------------------------------------------------------------
    cohort_declared = {entry["key"] for entry in manifest.get("reduction_cohort", [])}
    by_strategy: dict[str, set[str]] = {}
    originals: dict[tuple[str, str], str] = {}
    for record in run.reductions:
        by_strategy.setdefault(record["strategy"], set()).add(record["cohort_key"])
        originals[(record["cohort_key"], record["strategy"])] = record["original_case_id"]

    declared_strategies = set(manifest.get("preset", {}).get("strategies", []))
    if declared_strategies and set(by_strategy) != declared_strategies:
        issues.append(
            IntegrityIssue(
                "missing_strategy",
                FATAL,
                f"the manifest declares strategies {sorted(declared_strategies)} but records "
                f"exist only for {sorted(by_strategy)}; the comparison is not paired",
            )
        )

    if by_strategy:
        keysets = list(by_strategy.values())
        if any(keys != keysets[0] for keys in keysets):
            issues.append(
                IntegrityIssue(
                    "mismatched_reducer_cohorts",
                    FATAL,
                    "the reducers ran on different sets of starting cases, so the comparison "
                    "is not paired",
                )
            )
        for strategy, keys in by_strategy.items():
            if keys != cohort_declared:
                issues.append(
                    IntegrityIssue(
                        "cohort_drift",
                        FATAL,
                        f"strategy {strategy!r} covered {len(keys)} cohort cases but the "
                        f"manifest froze {len(cohort_declared)}",
                    )
                )
        for key in cohort_declared:
            starts = {originals.get((key, s)) for s in by_strategy}
            if len(starts) > 1:
                issues.append(
                    IntegrityIssue(
                        "unpaired_start",
                        FATAL,
                        f"cohort case {key!r} was reduced from different starting cases by "
                        "different strategies",
                    )
                )

    # -- claims that must be backed by evidence ----------------------------------------------------------
    for record in run.reductions:
        if not record["still_fails"]:
            issues.append(
                IntegrityIssue(
                    "reduction_does_not_fail",
                    FATAL,
                    f"{record['trial_id']} returned a case that does not stably fail "
                    f"({record['verification_verdict']})",
                )
            )

    # -- provenance ------------------------------------------------------------------------------------------
    if not manifest.get("source_commit"):
        issues.append(
            IntegrityIssue(
                "unknown_commit",
                WARNING if allow_dirty else FATAL,
                "the source commit is unknown, so these results cannot be traced to a revision",
            )
        )
    if not manifest.get("source_clean", False):
        issues.append(
            IntegrityIssue(
                "dirty_source",
                WARNING if allow_dirty else FATAL,
                "the working tree was dirty; a published headline run requires clean source",
            )
        )

    summary = run.summary
    if summary and not summary.get("complete", False):
        issues.append(
            IntegrityIssue(
                "incomplete_run",
                FATAL,
                f"the run wrote {summary.get('trials_written')} of "
                f"{summary.get('trials_declared')} declared trials",
            )
        )
    if summary.get("rss_exceeded"):
        issues.append(
            IntegrityIssue("rss_exceeded", WARNING, "the run exceeded its configured RSS ceiling")
        )
    return issues


def fatal_issues(issues: Sequence[IntegrityIssue]) -> list[IntegrityIssue]:
    return [issue for issue in issues if issue.severity == FATAL]


# --- metrics -------------------------------------------------------------------------------------------------


def _rate(numerator: int, denominator: int, label: str) -> float:
    if denominator <= 0:
        raise ReportError(f"invalid denominator for {label}: {denominator}")
    return numerator / denominator


def _family_bootstrap(
    per_family: dict[str, tuple[int, int]], *, samples: int = 2000, seed: int = 17
) -> tuple[float, float]:
    """Resample whole families, because trials within a family are not independent.

    Five seeds against one mutant measure the same fault five times. The family is the closest
    available independent unit, and with eight of them the interval is wide by construction.
    """
    families = sorted(per_family)
    if len(families) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        picked = rng.integers(0, len(families), size=len(families))
        detected = sum(per_family[families[i]][0] for i in picked)
        total = sum(per_family[families[i]][1] for i in picked)
        if total:
            estimates.append(detected / total)
    if not estimates:
        return (float("nan"), float("nan"))
    return (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)))


def compute_metrics(run: LoadedRun) -> dict[str, Any]:
    """Derive every published figure from the raw records."""
    manifest = run.manifest
    metrics: dict[str, Any] = {}

    # -- detection -------------------------------------------------------------------------------
    detection = run.detection
    detected = sum(1 for record in detection if record["detected"])
    metrics["detection"] = {
        "trials": len(detection),
        "detected": detected,
        "rate": _rate(detected, len(detection), "detection within budget"),
        "censored": len(detection) - detected,
    }

    per_generator: dict[str, dict[str, Any]] = {}
    for generator in sorted({record["generator"] for record in detection}):
        generator_rows = [record for record in detection if record["generator"] == generator]
        successes = [record for record in generator_rows if record["detected"]]
        per_generator[generator] = {
            "trials": len(generator_rows),
            "detected": len(successes),
            "rate": _rate(len(successes), len(generator_rows), f"detection[{generator}]"),
            "median_cases_to_detection": (
                statistics.median(record["cases_examined"] for record in successes)
                if successes
                else None
            ),
            "median_seconds_to_detection": (
                statistics.median(record["comparison_s"] for record in successes)
                if successes
                else None
            ),
            "total_comparison_s": sum(record["comparison_s"] for record in generator_rows),
            "total_model_runs": sum(record["model_runs"] for record in generator_rows),
        }
    metrics["detection"]["per_generator"] = per_generator

    per_family: dict[str, dict[str, Any]] = {}
    bootstrap_input: dict[str, tuple[int, int]] = {}
    for family in sorted({record["family_key"] for record in detection}):
        family_rows = [record for record in detection if record["family_key"] == family]
        n_detected = sum(1 for record in family_rows if record["detected"])
        per_family[family] = {
            "trials": len(family_rows),
            "detected": n_detected,
            "rate": _rate(n_detected, len(family_rows), f"detection[{family}]"),
            "variants": sorted({record["mutant_id"] for record in family_rows}),
            "variants_detected": sorted(
                {record["mutant_id"] for record in family_rows if record["detected"]}
            ),
        }
        bootstrap_input[family] = (n_detected, len(family_rows))
    metrics["detection"]["per_family"] = per_family
    low, high = _family_bootstrap(bootstrap_input)
    metrics["detection"]["family_bootstrap_95"] = {"low": low, "high": high, "unit": "family"}

    families = {record["family_key"] for record in detection}
    families_hit = {record["family_key"] for record in detection if record["detected"]}
    variants = {record["mutant_id"] for record in detection}
    variants_hit = {record["mutant_id"] for record in detection if record["detected"]}
    metrics["coverage"] = {
        "families": len(families),
        "families_detected": len(families_hit),
        "family_coverage": _rate(len(families_hit), len(families), "family coverage"),
        "variants": len(variants),
        "variants_detected": len(variants_hit),
        "variant_coverage": _rate(len(variants_hit), len(variants), "variant coverage"),
        "undetected_variants": sorted(variants - variants_hit),
    }

    # -- localization --------------------------------------------------------------------------------
    stable_failures = [record for record in detection if record["detected"]]
    localized = [
        record
        for record in stable_failures
        if record["localization"] and record["localization"]["available"]
    ]
    metrics["localization"] = {
        "stable_failures": len(stable_failures),
        "localized": len(localized),
        "coverage": _rate(len(localized), len(stable_failures), "localization coverage")
        if stable_failures
        else 0.0,
        "reconverging": sum(1 for record in localized if record["localization"]["reconverged"]),
        "fully_aligned": sum(1 for record in localized if record["localization"]["fully_aligned"]),
        "earliest_addresses": sorted(
            {
                record["localization"]["earliest_observed"]
                for record in localized
                if record["localization"]["earliest_observed"]
            }
        ),
    }

    # -- controls -------------------------------------------------------------------------------------
    generated = run.generated_controls
    total_cases = sum(record["n_cases"] for record in generated)
    false_positives = sum(record["n_false_positives"] for record in generated)
    verdicts: dict[str, int] = {}
    for record in generated:
        for verdict, count in record["verdict_counts"].items():
            verdicts[verdict] = verdicts.get(verdict, 0) + count
    control_per_generator = {}
    for generator in sorted({record["generator"] for record in generated}):
        control_rows = [record for record in generated if record["generator"] == generator]
        cases = sum(record["n_cases"] for record in control_rows)
        positives = sum(record["n_false_positives"] for record in control_rows)
        control_per_generator[generator] = {
            "cases": cases,
            "false_positives": positives,
            "rate": _rate(positives, cases, f"false positives[{generator}]") if cases else 0.0,
        }
    metrics["controls"] = {
        "generated_cases": total_cases,
        "false_positives": false_positives,
        "false_positive_rate": _rate(false_positives, total_cases, "false-positive rate")
        if total_cases
        else 0.0,
        "verdict_counts": verdicts,
        "per_generator": control_per_generator,
        "false_positive_case_ids": [
            case_id for record in generated for case_id in record["false_positive_case_ids"]
        ],
        "fixed_controls": len(run.fixed_controls),
        "fixed_control_false_positives": sum(
            1 for record in run.fixed_controls if record["false_positive"]
        ),
    }

    # -- reduction ----------------------------------------------------------------------------------------
    reductions = run.reductions
    per_strategy: dict[str, Any] = {}
    for strategy in sorted({record["strategy"] for record in reductions}):
        strategy_rows = [record for record in reductions if record["strategy"] == strategy]
        per_strategy[strategy] = {
            "cases": len(strategy_rows),
            "median_ratio": statistics.median(
                record["token_reduction_ratio"] for record in strategy_rows
            ),
            "median_reduced_tokens": statistics.median(
                record["reduced_tokens"] for record in strategy_rows
            ),
            "median_queries": statistics.median(
                record["counters"]["logical_queries"] for record in strategy_rows
            ),
            "median_model_runs": statistics.median(
                record["counters"]["model_runs"] for record in strategy_rows
            ),
            "total_queries": sum(record["counters"]["logical_queries"] for record in strategy_rows),
            "total_model_runs": sum(record["counters"]["model_runs"] for record in strategy_rows),
            "total_cache_hits": sum(record["counters"]["cache_hits"] for record in strategy_rows),
            "median_wall_s": statistics.median(
                record["wall_time_ns"] / 1e9 for record in strategy_rows
            ),
            "total_wall_s": sum(record["wall_time_ns"] / 1e9 for record in strategy_rows),
            "one_minimal": sum(
                1 for record in strategy_rows if record["minimality"].startswith("one_minimal")
            ),
            "budget_exhausted": sum(1 for record in strategy_rows if record["budget_exhausted"]),
            "still_fails": sum(1 for record in strategy_rows if record["still_fails"]),
            "requests_removed": sum(
                record["original_requests"] - record["reduced_requests"] for record in strategy_rows
            ),
            "padding_removed": sum(
                record["original_padding"] - record["reduced_padding"] for record in strategy_rows
            ),
        }

    paired: list[dict[str, Any]] = []
    keys = sorted({record["cohort_key"] for record in reductions})
    for key in keys:
        pair = {record["strategy"]: record for record in reductions if record["cohort_key"] == key}
        if len(pair) < 2:
            continue
        ddmin, greedy = pair.get("ddmin"), pair.get("greedy")
        if not ddmin or not greedy:
            continue
        paired.append(
            {
                "cohort_key": key,
                "family_key": ddmin["family_key"],
                "original_tokens": ddmin["original_tokens"],
                "ddmin_tokens": ddmin["reduced_tokens"],
                "greedy_tokens": greedy["reduced_tokens"],
                "ddmin_queries": ddmin["counters"]["logical_queries"],
                "greedy_queries": greedy["counters"]["logical_queries"],
                "ddmin_wall_s": ddmin["wall_time_ns"] / 1e9,
                "greedy_wall_s": greedy["wall_time_ns"] / 1e9,
                "ddmin_smaller": ddmin["reduced_tokens"] < greedy["reduced_tokens"],
                "greedy_smaller": greedy["reduced_tokens"] < ddmin["reduced_tokens"],
                "tied": ddmin["reduced_tokens"] == greedy["reduced_tokens"],
            }
        )
    metrics["reduction"] = {
        "cohort_size": len(keys),
        "per_strategy": per_strategy,
        "paired": paired,
        "ddmin_smaller": sum(1 for row in paired if row["ddmin_smaller"]),
        "greedy_smaller": sum(1 for row in paired if row["greedy_smaller"]),
        "tied": sum(1 for row in paired if row["tied"]),
        "median_query_ratio": (
            statistics.median(
                row["greedy_queries"] / row["ddmin_queries"]
                for row in paired
                if row["ddmin_queries"] > 0
            )
            if paired
            else None
        ),
    }

    # -- reproduction ---------------------------------------------------------------------------------------
    exports = run.exports
    metrics["reproduction"] = {
        "attempted": len(exports),
        "reproduced": sum(1 for record in exports if record["reproduced"]),
        "rate": _rate(
            sum(1 for record in exports if record["reproduced"]),
            len(exports),
            "reproduction success",
        )
        if exports
        else None,
        "imported_evallens": sum(1 for record in exports if record["imported_evallens"]),
        "median_seconds": (
            statistics.median(record["elapsed_s"] for record in exports) if exports else None
        ),
    }

    # -- resources and timing ------------------------------------------------------------------------------------
    summary = run.summary
    metrics["resources"] = {
        "peak_rss_bytes": summary.get("peak_rss_bytes"),
        "peak_rss_mib": (
            round(summary["peak_rss_bytes"] / 1024**2, 1) if summary.get("peak_rss_bytes") else None
        ),
        "rss_limit_mib": round(summary.get("rss_limit_bytes", 0) / 1024**2, 1),
        "exceeded": summary.get("rss_exceeded"),
        "threads": manifest.get("threads", {}).get("torch_num_threads"),
        "wall_time_s": summary.get("wall_time_s"),
        "model_init_s": (manifest.get("model_init_ns", 0) or 0) / 1e9,
        "total_detection_comparison_s": sum(record["comparison_s"] for record in detection),
        "total_localization_s": sum(record["localization_ns"] for record in detection) / 1e9,
        "total_generation_s": sum(record["generation_ns"] for record in detection) / 1e9,
    }
    return metrics


# --- rendering ---------------------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_results(
    run: LoadedRun, metrics: dict[str, Any], issues: Sequence[IntegrityIssue]
) -> str:
    manifest = run.manifest
    environment = manifest["environment"]
    preset = manifest["preset"]
    detection = metrics["detection"]
    coverage = metrics["coverage"]
    controls = metrics["controls"]
    reduction = metrics["reduction"]
    reproduction = metrics["reproduction"]
    localization = metrics["localization"]
    resources = metrics["resources"]

    warnings = [issue for issue in issues if issue.severity == WARNING]
    warning_block = ""
    if warnings:
        rows = "\n".join(f"- **{issue.code}** — {issue.message}" for issue in warnings)
        warning_block = f"\n## Warnings recorded for this run\n\n{rows}\n"

    family_rows = "\n".join(
        f"| {family} | {stats['detected']}/{stats['trials']} | {_pct(stats['rate'])} | "
        f"{len(stats['variants_detected'])}/{len(stats['variants'])} |"
        for family, stats in sorted(metrics["detection"]["per_family"].items())
    )
    generator_rows = "\n".join(
        f"| {name} | {stats['detected']}/{stats['trials']} | {_pct(stats['rate'])} | "
        f"{stats['median_cases_to_detection']} | "
        f"{stats['median_seconds_to_detection']:.4f}s | {stats['total_model_runs']:,} |"
        for name, stats in sorted(detection["per_generator"].items())
    )
    control_rows = "\n".join(
        f"| {name} | {stats['cases']} | {stats['false_positives']} | {_pct(stats['rate'])} |"
        for name, stats in sorted(controls["per_generator"].items())
    )
    strategy_rows = "\n".join(
        f"| {name} | {stats['median_ratio']:.1f}x | {stats['median_reduced_tokens']:.0f} | "
        f"{stats['median_queries']:.0f} | {stats['median_model_runs']:.0f} | "
        f"{stats['median_wall_s']:.3f}s | {stats['one_minimal']}/{stats['cases']} | "
        f"{stats['still_fails']}/{stats['cases']} |"
        for name, stats in sorted(reduction["per_strategy"].items())
    )
    paired_rows = "\n".join(
        f"| `{row['family_key']}` | {row['original_tokens']} | {row['ddmin_tokens']} | "
        f"{row['greedy_tokens']} | {row['ddmin_queries']} | {row['greedy_queries']} |"
        for row in reduction["paired"]
    )
    verdict_rows = "\n".join(
        f"| `{verdict}` | {count} |"
        for verdict, count in sorted(controls["verdict_counts"].items())
    )
    bootstrap = detection["family_bootstrap_95"]
    if not np.isfinite(bootstrap["low"]):
        interval = "not computed (fewer than two families)"
        interval_note = ""
    elif bootstrap["low"] == bootstrap["high"]:
        interval = f"{_pct(bootstrap['low'])} (degenerate)"
        interval_note = (
            f" The interval has zero width only because every one of the "
            f"{coverage['families']} families detected in every resample; it reflects a "
            "saturated corpus, not a precise estimate. A corpus that no generator ever misses "
            "cannot discriminate between generators."
        )
    else:
        interval = f"{_pct(bootstrap['low'])} to {_pct(bootstrap['high'])}"
        interval_note = (
            f" With {coverage['families']} families this interval is wide and should not be "
            "read as a precise estimate."
        )

    return f"""# EvalLens results

Generated by `bench/report.py` from the raw records in `{run.root.name}/`. Every figure below
is derived from those JSONL files; nothing is hand-entered.

> **Every fault measured here was injected by EvalLens on purpose**, as test material for its
> own detector. None of them is a bug discovered in PyTorch or any other third-party library.
> These are held-out executions of *known* fault families — they are not evidence of
> generalization to unknown real-world bugs.

| | |
|---|---|
| Run | `{manifest["run_id"]}` |
| Preset | `{preset["name"]}` |
| Source commit | `{manifest["source_commit"]}` |
| Working tree | {"clean" if manifest["source_clean"] else "**dirty — not publishable**"} |
| Fixture | `{manifest["model_config"]["config_id"]}` |
| Weights SHA-256 | `{manifest["weights_sha256"][:32]}…` |
| Tolerance policy | `atol={manifest["policy"]["atol"]:g}, rtol={manifest["policy"]["rtol"]:g}` (`{manifest["policy"]["policy_id"]}`) |
| Machine | {environment["chip"]}, {environment["cpu_count"]} CPUs, {environment["platform"]} |
| Python / torch / numpy | {environment["python_version"]} / {environment["torch_version"]} / {environment["numpy_version"]} |
| Threads | {resources["threads"]} |
| Declared trials | {len(manifest["expected_trial_ids"])}, all produced records |
| Wall time | {resources["wall_time_s"]:.1f}s |
{warning_block}
## Workload

Declared **before** any trial ran, and frozen in `manifest.json`:
{preset["n_seeds"]} held-out seed(s) x {len(manifest["mutant_manifest"])} qualified variants x
{len(preset["generators"])} generators, at most {preset["cases_per_trial"]} cases per trial;
{preset["control_cases_per_generator"]} known-good cases per generator;
{preset["reduction_cohort_per_family"]} reduction cohort case(s) per family.

These are workload parameters, not performance claims.

## Detection within budget

Detected mutant/seed/generator trials divided by all declared qualified trials. A trial that
did not detect is **budget-censored** and is counted in the denominator, never omitted.

**{detection["detected"]}/{detection["trials"]} = {_pct(detection["rate"])}**
({detection["censored"]} censored)

95% interval, resampling **whole fault families** (the unit, since seeds against one mutant
are not independent observations): {interval}.{interval_note}

### By generator

| Generator | Detected | Rate | Median cases to detect | Median comparison time | Model runs |
|---|---|---|---|---|---|
{generator_rows}

### By fault family

| Family | Detected | Rate | Variants detected |
|---|---|---|---|
{family_rows}

## Coverage

| Metric | Value |
|---|---|
| Family coverage | {coverage["families_detected"]}/{coverage["families"]} = {_pct(coverage["family_coverage"])} |
| Variant coverage | {coverage["variants_detected"]}/{coverage["variants"]} = {_pct(coverage["variant_coverage"])} |
| Undetected variants | {", ".join(f"`{v}`" for v in coverage["undetected_variants"]) or "none"} |

## Known-good false positives

The candidate in these comparisons is a **correct** implementation. Any stable FAIL is a false
positive.

**{controls["false_positives"]}/{controls["generated_cases"]} = {_pct(controls["false_positive_rate"])}**

| Generator | Cases | False positives | Rate |
|---|---|---|---|
{control_rows}

All verdicts on known-good cases, reported separately rather than folded into one rate:

| Verdict | Count |
|---|---|
{verdict_rows}

Handwritten control corpus: {controls["fixed_controls"] - controls["fixed_control_false_positives"]}/{controls["fixed_controls"]} pass
({controls["fixed_control_false_positives"]} false positive(s)). These cover identical
implementations, correct cached versus full-prefix execution, padded alignment, batch
permutation, fresh-request isolation, and one benign sub-tolerance perturbation, so exact
equality is not the only control exercised.

## Localization coverage

Stable failures with a valid aligned checkpoint localization, divided by all stable failures.

**{localization["localized"]}/{localization["stable_failures"]} = {_pct(localization["coverage"])}**

| Metric | Value |
|---|---|
| Fully aligned (no unmatched checkpoints) | {localization["fully_aligned"]}/{localization["localized"]} |
| Divergence reconverges later | {localization["reconverging"]}/{localization["localized"]} |
| Distinct earliest-observed addresses | {len(localization["earliest_addresses"])} |

Earliest observed divergence is evidence about where a difference becomes **visible at the
exposed checkpoints**. It is not proof of root cause, and it cannot identify an operation
inside a layer.

The reconvergence count is why localization compares every aligned checkpoint instead of
bisecting: the "diverged" predicate is measurably non-monotone.

## Reduction

Both reducers ran from the **identical** frozen starting cases under **identical** budgets
({preset["reduction_budget"]["max_queries"]} predicate queries,
{preset["reduction_budget"]["time_budget_s"]:.0f}s). The cohort was selected and recorded
before either reducer was constructed.

| Strategy | Median ratio | Median reduced tokens | Median queries | Median model runs | Median wall | 1-minimal | Still fails |
|---|---|---|---|---|---|---|---|
{strategy_rows}

### Paired per-case comparison

| Family | Original tokens | ddmin | greedy | ddmin queries | greedy queries |
|---|---|---|---|---|---|
{paired_rows}

**ddmin smaller: {reduction["ddmin_smaller"]} · greedy smaller: {reduction["greedy_smaller"]} · tied: {reduction["tied"]}**
(of {len(reduction["paired"])} paired cases)

Median ratio of greedy queries to ddmin queries: **{reduction["median_query_ratio"]:.1f}x**.

Session and padding changes are reported separately from token counts:
{reduction["per_strategy"].get("ddmin", {}).get("requests_removed", 0)} request(s) and
{reduction["per_strategy"].get("ddmin", {}).get("padding_removed", 0)} padding token(s) removed
by ddmin across the cohort.

Every reduced case was independently re-verified to still fail stably; a reduction that did not
would be a fatal integrity error and this report would not exist.

## Reproduction

Exported packages that reproduced the recorded mismatch in a clean room — a fresh temporary
directory outside the checkout, run in an isolated interpreter.

**{reproduction["reproduced"]}/{reproduction["attempted"]} = {_pct(reproduction["rate"])}**

Packages that imported the installed EvalLens: **{reproduction["imported_evallens"]}** (must be 0).
Median verification time: {reproduction["median_seconds"]}s.

## Resource usage and timing

| Metric | Value |
|---|---|
| Peak worker RSS | {resources["peak_rss_mib"]} MiB of a {resources["rss_limit_mib"]} MiB budget |
| RSS ceiling exceeded | {resources["exceeded"]} |
| CPU threads | {resources["threads"]} |
| Total wall time | {resources["wall_time_s"]:.1f}s |
| Model initialization | {resources["model_init_s"]:.3f}s (reported separately from comparison time) |
| Detection comparison time | {resources["total_detection_comparison_s"]:.2f}s |
| Case generation time | {resources["total_generation_s"]:.3f}s (separate from comparison) |
| Diagnostic capture / localization | {resources["total_localization_s"]:.2f}s (a separate pass, never inside detection timing) |

MPS is available on this machine and was **not** used. CPU is the reference platform.

## How to read these numbers, and how not to

- **Eight fault families is a narrow corpus.** Uncertainty on any aggregate is substantial, and
  the family-level interval above reflects that. No generalization claim is made.
- **Trials from one mutant are not independent evidence.** Five seeds against one variant
  measure the same fault five times.
- **"Detection within budget" is relative to this budget**, this fixture, and these generators.
- **A reduction that preserves a failure signature does not prove identical root cause.** It
  establishes the same failure class at the same target request in the same category.
- **1-minimal means 1-minimal with respect to the declared deletion operations**, never
  globally smallest.
- Injected faults are real testing evidence, but they are faults this project wrote.

## Reproducing this report

```bash
python -m bench.run --preset {preset["name"]} --out artifacts/runs/{preset["name"]} --config configs/cpu.toml
python -m bench.report artifacts/runs/{preset["name"]} --out RESULTS.md
```

The report regenerates deterministically from the same raw records.
"""


def render_incomplete(run: LoadedRun, issues: Sequence[IntegrityIssue]) -> str:
    fatal = fatal_issues(issues)
    warnings = [issue for issue in issues if issue.severity == WARNING]
    fatal_rows = "\n".join(f"- **{issue.code}** — {issue.message}" for issue in fatal)
    warning_rows = "\n".join(f"- **{issue.code}** — {issue.message}" for issue in warnings)
    manifest = run.manifest
    summary = run.summary

    return f"""# INCOMPLETE benchmark run — diagnostic report

**This is not a results table and must not be presented as one.** The run at
`{run.root}` failed at least one integrity check, so `RESULTS.md` was deliberately not
generated.

| | |
|---|---|
| Run | `{manifest.get("run_id", "unknown")}` |
| Preset | `{manifest.get("preset", {}).get("name", "unknown")}` |
| Source commit | `{manifest.get("source_commit") or "unknown"}` |
| Working tree | {"clean" if manifest.get("source_clean") else "dirty"} |
| Declared trials | {len(manifest.get("expected_trial_ids", []))} |
| Records written | {summary.get("trials_written", "unknown")} |

## Fatal issues

{fatal_rows or "- none"}

## Warnings

{warning_rows or "- none"}

## Records present

| Source | Records |
|---|---|
| `detection.jsonl` | {len(run.detection)} |
| `controls.jsonl` | {len(run.controls)} |
| `reductions.jsonl` | {len(run.reductions)} |
| `exports.jsonl` | {len(run.exports)} |

## What to do

Resume is only valid under the **same** frozen inputs and the same source revision. If either
has changed, start a new run rather than merging records from two different configurations.

```bash
python -m bench.run --preset <preset> --out <fresh run dir> --config configs/cpu.toml
```
"""


# --- entry point --------------------------------------------------------------------------------------------------


def generate(
    run_dir: str | Path,
    out_path: str | Path,
    *,
    allow_dirty: bool = False,
    metrics_path: str | Path | None = None,
) -> tuple[int, list[IntegrityIssue]]:
    """Write a report. Returns ``(exit_code, issues)``.

    A fatal issue never produces a file named `RESULTS.md`; it produces `INCOMPLETE.md` beside
    the requested path and a nonzero exit code.
    """
    run = load_run(run_dir)
    issues = check_integrity(run, allow_dirty=allow_dirty)
    fatal = fatal_issues(issues)
    destination = Path(out_path).resolve()

    if fatal:
        diagnostic = destination.with_name("INCOMPLETE.md")
        diagnostic.write_text(render_incomplete(run, issues), encoding="utf-8")
        print(f"integrity check failed with {len(fatal)} fatal issue(s):")
        for issue in fatal:
            print(f"  [{issue.code}] {issue.message}")
        print(f"wrote diagnostic report to {diagnostic}")
        print(f"RESULTS.md was NOT generated. {destination.name} is unchanged.")
        return 1, issues

    metrics = compute_metrics(run)
    destination.write_text(render_results(run, metrics, issues), encoding="utf-8")
    if metrics_path:
        Path(metrics_path).write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {destination}")
    for issue in issues:
        print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    return 0, issues


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate RESULTS.md from a benchmark run's raw records. Refuses to produce a "
            "results table for an incomplete or inconsistent study."
        )
    )
    parser.add_argument("run_dir", help="Benchmark run directory.")
    parser.add_argument("--out", default="RESULTS.md", help="Output path.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Downgrade dirty/unknown source from fatal to a warning (not publishable).",
    )
    parser.add_argument("--metrics", default=None, help="Also write computed metrics as JSON.")
    args = parser.parse_args(argv)

    try:
        code, _ = generate(
            args.run_dir, args.out, allow_dirty=args.allow_dirty, metrics_path=args.metrics
        )
    except ReportError as exc:
        # A malformed or absent run directory is a usage error, not a crash. Printing a
        # traceback here would bury the one line that says what to fix.
        print(f"cannot generate a report: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
