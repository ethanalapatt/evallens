"""Report integrity: the checks that stand between raw records and a publishable table.

Every run built in this file is **synthetic** — hand-constructed JSON, not a measurement. It
is labeled `synthetic: true` in its manifest, and one of the tests below asserts that such a
run can never produce a file named `RESULTS.md`. That is the guard against a test fixture ever
being mistaken for evidence.

The tests exercise falsification directly: falsified coverage, duplicated trials, wrong
hashes, impossible counts, and failures hidden as skipped rows must each be rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench.report import (
    FATAL,
    ReportError,
    check_integrity,
    compute_metrics,
    fatal_issues,
    generate,
    load_run,
)


def _detection(
    mutant_id: str, family: str, generator: str, seed: int, detected: bool
) -> dict[str, Any]:
    return {
        "kind": "detection",
        "trial_id": f"detect/{mutant_id}/{generator}/{seed}",
        "mutant_id": mutant_id,
        "family_key": family,
        "family_number": 1,
        "variant": mutant_id.split(".")[-1],
        "injected_fault": True,
        "generator": generator,
        "seed": seed,
        "case_budget": 8,
        "detected": detected,
        "budget_censored": not detected,
        "cases_examined": 3 if detected else 8,
        "cases_unused": 5 if detected else 0,
        "verdict_counts": {"pass": 2, "fail": 1} if detected else {"pass": 8},
        "detection_case_id": "case_abc" if detected else None,
        "detection_case_category": "cached_decode" if detected else None,
        "detection_detail": "policy violated" if detected else "",
        "max_abs_err": 0.25 if detected else 0.0,
        "comparison_ns": 1_000_000,
        "comparison_s": 0.001,
        "generation_ns": 1000,
        "localization_ns": 500 if detected else 0,
        "localization": (
            {
                "available": True,
                "reason": "ok",
                "earliest_observed": "r0/block0/pos1/attn_out",
                "n_divergent": 5,
                "n_compared": 20,
                "reconverged": True,
                "fully_aligned": True,
            }
            if detected
            else None
        ),
        "model_runs": 18,
        "errors": 0,
        "peak_rss_bytes": 400_000_000,
    }


def _control(generator: str, seed: int, false_positives: int = 0) -> dict[str, Any]:
    cases = 10
    return {
        "kind": "control",
        "trial_id": f"control/{generator}/{seed}",
        "generator": generator,
        "seed": seed,
        "injected_fault": False,
        "n_cases": cases,
        "verdict_counts": {"pass": cases - false_positives, "fail": false_positives},
        "per_category": {"cached_decode": {"pass": cases}},
        "false_positive_case_ids": [f"case_fp{i}" for i in range(false_positives)],
        "n_false_positives": false_positives,
        "elapsed_ns": 5_000_000,
        "model_runs": 60,
        "errors": 0,
    }


def _reduction(key: str, family: str, strategy: str, **overrides: Any) -> dict[str, Any]:
    record = {
        "kind": "reduction",
        "trial_id": f"reduce/{key}/{strategy}",
        "cohort_key": key,
        "mutant_id": f"{family}.v1",
        "family_key": family,
        "injected_fault": True,
        "strategy": strategy,
        "original_case_id": "case_start",
        "reduced_case_id": "case_small",
        "original_tokens": 40,
        "reduced_tokens": 2,
        "original_requests": 1,
        "reduced_requests": 1,
        "original_padding": 0,
        "reduced_padding": 0,
        "token_reduction_ratio": 20.0,
        "minimality": "one_minimal_wrt_declared_operations",
        "budget_exhausted": False,
        "counters": {
            "logical_queries": 5 if strategy == "ddmin" else 39,
            "model_runs": 30 if strategy == "ddmin" else 234,
            "cache_hits": 0,
        },
        "wall_time_ns": 30_000_000,
        "n_accepted_steps": 3,
        "still_fails": True,
        "verification_verdict": "fail",
        "reduced_case": {},
    }
    record.update(overrides)
    return record


def _export(key: str, reproduced: bool = True) -> dict[str, Any]:
    return {
        "kind": "export",
        "trial_id": f"export/{key}",
        "cohort_key": key,
        "mutant_id": "fam_a.v1",
        "family_key": "fam_a",
        "injected_fault": True,
        "case_id": "case_small",
        "reproduced": reproduced,
        "ok": reproduced,
        "exit_code": 0 if reproduced else 4,
        "expected_exit_code": 0,
        "imported_evallens": False,
        "elapsed_s": 0.6,
    }


def make_run(root: Path, **manifest_overrides: Any) -> Path:
    """Build a complete, internally consistent SYNTHETIC run directory."""
    root.mkdir(parents=True, exist_ok=True)
    families = ["fam_a", "fam_b"]
    mutants = [f"{family}.v1" for family in families]
    generators = ["uniform-valid", "boundary-aware"]
    seeds = [3000, 3001]
    cohort = [
        {"key": f"{m}#case_start", "mutant_id": m, "family_key": m.split(".")[0]} for m in mutants
    ]

    detection = [
        _detection(m, m.split(".")[0], g, s, True)
        for m in mutants
        for g in generators
        for s in seeds
    ]
    controls = [_control(g, s) for g in generators for s in seeds]
    controls.append(
        {
            "kind": "fixed_control",
            "trial_id": "fixed_control/identical",
            "control_id": "identical",
            "injected_fault": False,
            "description": "d",
            "exercises": "e",
            "verdict": "pass",
            "stable": True,
            "max_abs_err": 0.0,
            "detail": "",
            "false_positive": False,
        }
    )
    reductions = [
        _reduction(entry["key"], entry["family_key"], strategy)
        for entry in cohort
        for strategy in ("ddmin", "greedy")
    ]
    exports = [_export(entry["key"]) for entry in cohort]

    expected = sorted(
        [record["trial_id"] for record in detection]
        + [record["trial_id"] for record in controls if record["kind"] == "control"]
        + [record["trial_id"] for record in reductions]
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "synthetic-test-run",
        "evallens_version": "0.1.0",
        "synthetic": True,  # <-- never a measurement
        "preset": {
            "name": "synthetic",
            "seeds": seeds,
            "n_seeds": len(seeds),
            "cases_per_trial": 8,
            "control_cases_per_generator": 10,
            "reduction_cohort_per_family": 1,
            "export_sample": 2,
            "reduction_budget": {"max_queries": 64, "time_budget_s": 20.0},
            "max_tokens": 48,
            "order_seed": 1,
            "generators": generators,
            "strategies": ["ddmin", "greedy"],
        },
        "model_config": {"config_id": "tiny-test"},
        "weights_sha256": "a" * 64,
        "policy": {"atol": 1e-5, "rtol": 1e-4, "policy_id": "pid123", "name": "t"},
        "environment": {
            "chip": "Apple M3",
            "cpu_count": 8,
            "platform": "macOS",
            "python_version": "3.13.7",
            "torch_version": "2.14.0",
            "numpy_version": "2.5.3",
        },
        "threads": {"torch_num_threads": 4},
        "source_commit": "0" * 40,
        "source_clean": True,
        "mutant_manifest": [{"mutant_id": m} for m in mutants],
        "qualification": [{"mutant_id": m, "qualified": True} for m in mutants],
        "qualified_mutant_ids": mutants,
        "unqualified_mutant_ids": [],
        "control_manifest": [],
        "generator_versions": dict.fromkeys(generators, "1"),
        "reduction_cohort": cohort,
        "expected_trial_ids": expected,
        "model_init_ns": 1_000_000,
    }
    manifest.update(manifest_overrides)

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    for name, records in (
        ("detection.jsonl", detection),
        ("controls.jsonl", controls),
        ("reductions.jsonl", reductions),
        ("exports.jsonl", exports),
    ):
        (root / name).write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "preset": "synthetic",
                "wall_time_s": 1.0,
                "peak_rss_bytes": 500_000_000,
                "rss_limit_bytes": 4 * 1024**3,
                "rss_exceeded": False,
                "trials_written": len(expected),
                "trials_declared": len(expected),
                "complete": True,
            }
        )
    )
    return root


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def _fatal_codes(run_dir: Path) -> set[str]:
    return {issue.code for issue in fatal_issues(check_integrity(load_run(run_dir)))}


# --- the synthetic guard -----------------------------------------------------------------------


def test_a_synthetic_run_can_never_produce_results_md(tmp_path) -> None:
    """The rule that keeps a test fixture from ever being mistaken for a measurement."""
    run = make_run(tmp_path / "run")
    out = tmp_path / "RESULTS.md"
    code, issues = generate(run, out)

    assert code == 1
    assert "synthetic_run" in _codes(fatal_issues(issues))
    assert not out.exists()
    assert (tmp_path / "INCOMPLETE.md").exists()
    assert "must not be presented as one" in (tmp_path / "INCOMPLETE.md").read_text()


def test_a_non_synthetic_consistent_run_produces_results(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    out = tmp_path / "RESULTS.md"
    code, issues = generate(run, out)

    assert code == 0, [i.message for i in fatal_issues(issues)]
    assert out.exists()
    text = out.read_text()
    assert "injected by EvalLens on purpose" in text
    assert "**8/8 = 100.0%**" in text  # overall detection
    assert "| uniform-valid | 4/4 | 100.0% |" in text  # per-generator detection


# --- falsification --------------------------------------------------------------------------------


def test_a_missing_declared_trial_is_fatal(tmp_path) -> None:
    """A trial that quietly vanishes would shrink a denominator and inflate a rate."""
    run = make_run(tmp_path / "run", synthetic=False)
    lines = (run / "detection.jsonl").read_text().splitlines()
    (run / "detection.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert "missing_trials" in _fatal_codes(run)


def test_a_duplicated_trial_is_fatal(tmp_path) -> None:
    """Counting one success twice is the cheapest way to fake a rate."""
    run = make_run(tmp_path / "run", synthetic=False)
    lines = (run / "detection.jsonl").read_text().splitlines()
    (run / "detection.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n")
    assert "duplicate_trials" in _fatal_codes(run)


def test_falsified_coverage_cannot_hide_a_missing_family(tmp_path) -> None:
    """Deleting a whole family's trials is caught as missing, not silently re-normalized."""
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "detection.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kept = [r for r in records if r["family_key"] != "fam_b"]
    (run / "detection.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in kept) + "\n"
    )
    assert "missing_trials" in _fatal_codes(run)


def test_a_failure_hidden_as_a_skipped_row_is_fatal(tmp_path) -> None:
    """Dropping an awkward trial is exactly what the declared-id check exists to catch."""
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    kept = [r for r in records if r["strategy"] != "greedy"]
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in kept) + "\n"
    )
    codes = _fatal_codes(run)
    assert "missing_trials" in codes
    assert "missing_strategy" in codes


def test_an_unqualified_variant_in_the_scored_set_is_fatal(tmp_path) -> None:
    """A variant that never qualified must not contribute to a detection rate."""
    run = make_run(tmp_path / "run", synthetic=False, unqualified_mutant_ids=["fam_a.v1"])
    assert "unqualified_scored" in _fatal_codes(run)


def test_missing_hashes_are_fatal(tmp_path) -> None:
    for key, code in (
        ("weights_sha256", "missing_weights_hash"),
        ("policy", "missing_policy_id"),
        ("model_config", "missing_config_id"),
    ):
        run = make_run(
            tmp_path / f"run_{key}", synthetic=False, **{key: {} if key != "weights_sha256" else ""}
        )
        assert code in _fatal_codes(run), key


def test_a_dirty_source_tree_blocks_publication(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False, source_clean=False)
    assert "dirty_source" in _fatal_codes(run)

    # ...but can be downgraded explicitly, and then the report says so.
    out = tmp_path / "RESULTS.md"
    code, _ = generate(run, out, allow_dirty=True)
    assert code == 0
    assert "not publishable" in out.read_text()


def test_an_unknown_commit_blocks_publication(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False, source_commit=None)
    assert "unknown_commit" in _fatal_codes(run)


def test_mismatched_reducer_cohorts_are_fatal(tmp_path) -> None:
    """A paired comparison requires both strategies on the same starting cases."""
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        if record["strategy"] == "greedy":
            record["cohort_key"] = record["cohort_key"] + "_other"
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    codes = _fatal_codes(run)
    assert "mismatched_reducer_cohorts" in codes or "cohort_drift" in codes


def test_unpaired_starting_cases_are_fatal(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        if record["strategy"] == "greedy":
            record["original_case_id"] = "case_somewhere_else"
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    assert "unpaired_start" in _fatal_codes(run)


def test_a_reduction_that_does_not_fail_is_fatal(tmp_path) -> None:
    """Claiming a reduction that no longer reproduces would be the worst kind of wrong."""
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records[0]["still_fails"] = False
    records[0]["verification_verdict"] = "pass"
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    assert "reduction_does_not_fail" in _fatal_codes(run)


def test_an_incomplete_run_is_fatal(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    summary = json.loads((run / "summary.json").read_text())
    summary["complete"] = False
    summary["trials_written"] = 3
    (run / "summary.json").write_text(json.dumps(summary))
    assert "incomplete_run" in _fatal_codes(run)


def test_a_manifest_with_no_declared_trials_is_fatal(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False, expected_trial_ids=[])
    assert "no_declared_trials" in _fatal_codes(run)


def test_an_exceeded_rss_ceiling_is_a_warning_not_a_silent_pass(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    summary = json.loads((run / "summary.json").read_text())
    summary["rss_exceeded"] = True
    (run / "summary.json").write_text(json.dumps(summary))
    issues = check_integrity(load_run(run))
    assert "rss_exceeded" in _codes(issues)
    assert not any(i.code == "rss_exceeded" and i.severity == FATAL for i in issues)


def test_a_directory_without_a_manifest_is_rejected(tmp_path) -> None:
    with pytest.raises(ReportError, match=r"no manifest\.json"):
        load_run(tmp_path)


# --- arithmetic -----------------------------------------------------------------------------------------


def test_metrics_are_derived_from_the_records_not_assumed(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "detection.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records[0]["detected"] = False
    records[0]["budget_censored"] = True
    records[0]["localization"] = None
    (run / "detection.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )

    metrics = compute_metrics(load_run(run))
    assert metrics["detection"]["trials"] == 8
    assert metrics["detection"]["detected"] == 7
    assert metrics["detection"]["rate"] == pytest.approx(7 / 8)
    assert metrics["detection"]["censored"] == 1
    assert metrics["localization"]["stable_failures"] == 7


def test_false_positives_are_counted_and_the_offending_cases_kept(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "controls.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        if record["kind"] == "control" and record["generator"] == "uniform-valid":
            record["n_false_positives"] = 1
            record["false_positive_case_ids"] = ["case_bad"]
            record["verdict_counts"] = {"pass": 9, "fail": 1}
    (run / "controls.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )

    metrics = compute_metrics(load_run(run))
    assert metrics["controls"]["generated_cases"] == 40
    assert metrics["controls"]["false_positives"] == 2
    assert metrics["controls"]["false_positive_rate"] == pytest.approx(2 / 40)
    assert metrics["controls"]["false_positive_case_ids"] == ["case_bad", "case_bad"]
    assert metrics["controls"]["verdict_counts"]["fail"] == 2


def test_a_zero_denominator_raises_rather_than_reporting_a_rate(tmp_path) -> None:
    """An impossible count must stop the report, not divide by zero into nonsense."""
    run = make_run(tmp_path / "run", synthetic=False)
    (run / "detection.jsonl").write_text("")
    with pytest.raises(ReportError, match="invalid denominator"):
        compute_metrics(load_run(run))


def test_the_paired_comparison_reports_wins_losses_and_ties(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        if record["cohort_key"].startswith("fam_a") and record["strategy"] == "greedy":
            record["reduced_tokens"] = 5
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )

    metrics = compute_metrics(load_run(run))
    assert metrics["reduction"]["ddmin_smaller"] == 1
    assert metrics["reduction"]["greedy_smaller"] == 0
    assert metrics["reduction"]["tied"] == 1
    assert metrics["reduction"]["median_query_ratio"] == pytest.approx(39 / 5)


def test_a_losing_result_is_published_not_suppressed(tmp_path) -> None:
    """If ddmin loses to its baseline, the report says so."""
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "reductions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        record["reduced_tokens"] = 9 if record["strategy"] == "ddmin" else 2
    (run / "reductions.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )

    out = tmp_path / "RESULTS.md"
    assert generate(run, out)[0] == 0
    metrics = compute_metrics(load_run(run))
    assert metrics["reduction"]["greedy_smaller"] == 2
    assert metrics["reduction"]["ddmin_smaller"] == 0
    assert "greedy smaller: 2" in out.read_text()


def test_reproduction_failures_are_reported(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    records = [
        json.loads(line)
        for line in (run / "exports.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records[0]["reproduced"] = False
    (run / "exports.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    )
    metrics = compute_metrics(load_run(run))
    assert metrics["reproduction"]["rate"] == pytest.approx(0.5)


# --- determinism -----------------------------------------------------------------------------------------------


def test_the_report_regenerates_deterministically(tmp_path) -> None:
    """The same raw records must always produce the same bytes."""
    run = make_run(tmp_path / "run", synthetic=False)
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    generate(run, first)
    generate(run, second)
    assert first.read_text() == second.read_text()


def test_metrics_are_deterministic_including_the_bootstrap(tmp_path) -> None:
    run = make_run(tmp_path / "run", synthetic=False)
    loaded = load_run(run)
    assert compute_metrics(loaded) == compute_metrics(loaded)
