"""Reproduction export, clean-room verification, and the one-command demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evallens.adapters.native import CandidateAdapter, NativeAdapterSpec, ReferenceAdapter
from evallens.compare import compare
from evallens.demo import DEMO_FAULT, run_demo
from evallens.export import (
    EXIT_MISMATCH,
    EXIT_NOT_REPRODUCED,
    EXIT_OK,
    EXIT_SETUP_ERROR,
    ExportError,
    ReproductionSpec,
    export_reproduction,
    verify_reproduction,
)
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.reduce import signature_from_failure
from evallens.replay import stable_comparison
from evallens.settings import Settings
from evallens.types import Case, ExecutionMode, Request, TolerancePolicy, Verdict

POLICY = TolerancePolicy()
FAULT = Behavior(cache_index="write_overwrite_last")


def _failing_case(config: ModelConfig, weights: WeightDict) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=weights_sha256(weights),
        requests=[Request("r0", (21, 22, 23, 24, 25), prefix_length=2)],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=1,
    )


@pytest.fixture(scope="module")
def package(tmp_path_factory, config: ModelConfig, weights: WeightDict) -> Path:
    """One exported package, reused across the export tests."""
    case = _failing_case(config, weights)
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, FAULT)
    outcome = stable_comparison(reference, candidate, case, POLICY)
    assert outcome.verdict is Verdict.FAIL

    spec = ReproductionSpec(
        case=case,
        reference=NativeAdapterSpec("reference", config),
        candidate=NativeAdapterSpec("candidate", config, FAULT, "test-fault"),
        policy=POLICY,
        signature=signature_from_failure(case, outcome.representative.failing_request_ids),
        comparison=compare(reference.run(case), candidate.run(case), POLICY),
        injected_fault=True,
        fault_description="A cached write overwrites the most recent slot instead of extending.",
    )
    return export_reproduction(spec, tmp_path_factory.mktemp("export") / "repro")


# --- package contents --------------------------------------------------------------------


def test_the_package_contains_everything_needed_to_run(package: Path) -> None:
    for name in (
        "repro.py",
        "README.md",
        "manifest.json",
        "case.json",
        "policy.json",
        "weights.npz",
    ):
        assert (package / name).exists(), name
    vendored = package / "repro_fixture"
    for name in ("__init__.py", "types.py", "compare.py", "config.py", "tiny_transformer.py"):
        assert (vendored / name).exists(), name


def test_no_vendored_source_imports_evallens(package: Path) -> None:
    """The single property that makes the package portable."""
    for path in package.rglob("*.py"):
        text = path.read_text()
        assert "from evallens" not in text, path.name
        assert "import evallens" not in text, path.name


def test_no_file_references_the_original_checkout(package: Path) -> None:
    import evallens

    checkout = str(Path(evallens.__file__).resolve().parents[2])
    for path in package.rglob("*"):
        if path.suffix not in {".py", ".md"}:
            continue
        assert checkout not in path.read_text(errors="ignore"), path.name


def test_weights_are_a_non_object_npz_matching_the_recorded_hash(package: Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    with np.load(package / "weights.npz", allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    assert all(array.dtype == np.float32 for array in arrays.values())
    assert weights_sha256(arrays) == manifest["weights_sha256"]


def test_the_manifest_records_provenance_and_the_injected_label(package: Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["injected_fault"] is True
    assert manifest["fault_description"]
    assert manifest["schema_version"] == 1
    assert manifest["reduced_case_id"].startswith("case_")
    assert manifest["recorded"]["verdict"] == "fail"
    assert manifest["recorded"]["max_abs_err"] > 0
    assert manifest["environment"]["torch_version"]
    assert set(manifest["exit_codes"]) == {
        "ok",
        "mismatch",
        "setup_error",
        "execution_error",
        "not_reproduced",
    }


def test_the_readme_labels_the_fault_as_injected(package: Path) -> None:
    """A reviewer must never mistake this for a discovered third-party bug."""
    readme = (package / "README.md").read_text()
    assert "deliberately injected fault" in readme
    assert "not a bug discovered in PyTorch" in readme
    assert "--expect-mismatch" in readme
    assert "random, untrained weights" in readme


def test_the_vendored_model_is_the_real_one_not_a_reimplementation(package: Path) -> None:
    import evallens

    original = (
        Path(evallens.__file__).resolve().parent / "fixtures/tiny_transformer.py"
    ).read_text()
    vendored = (package / "repro_fixture" / "tiny_transformer.py").read_text()
    # Identical apart from the rewritten import lines.
    original_body = [ln for ln in original.splitlines() if "import" not in ln]
    vendored_body = [ln for ln in vendored.splitlines() if "import" not in ln]
    assert original_body == vendored_body


# --- running the package -------------------------------------------------------------------


def _run(package: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-I", "repro.py", *args],
        cwd=package,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_expect_mismatch_exits_zero_when_the_mismatch_reproduces(package: Path) -> None:
    assert _run(package, "--expect-mismatch").returncode == EXIT_OK


def test_default_mode_exits_one_when_the_mismatch_is_present(package: Path) -> None:
    """A documented nonzero code, distinct from a setup or execution error."""
    assert _run(package).returncode == EXIT_MISMATCH


def test_the_package_reports_the_recorded_error_magnitude(package: Path) -> None:
    result = _run(package, "--json", "--expect-mismatch")
    assert result.returncode == EXIT_OK
    payload = json.loads(result.stdout)
    assert payload["mismatch_reproduced"] is True
    assert payload["stable"] is True
    assert payload["injected_fault"] is True
    assert payload["max_abs_err"] == pytest.approx(payload["expected_max_abs_err"], rel=1e-6)


def test_a_corrupted_weights_hash_is_a_setup_error_not_a_reproduction(
    package: Path, tmp_path
) -> None:
    """The most dangerous confusion: a broken package must not look like a success."""
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(package, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["weights_sha256"] = "0" * 64
    (copy / "manifest.json").write_text(json.dumps(manifest))

    result = _run(copy, "--expect-mismatch")
    assert result.returncode == EXIT_SETUP_ERROR
    assert "weights hash mismatch" in result.stderr


def test_a_missing_input_file_is_a_setup_error(package: Path, tmp_path) -> None:
    import shutil

    copy = tmp_path / "incomplete"
    shutil.copytree(package, copy)
    (copy / "case.json").unlink()

    result = _run(copy, "--expect-mismatch")
    assert result.returncode == EXIT_SETUP_ERROR
    assert "cannot read reproduction inputs" in result.stderr


def test_a_clean_candidate_does_not_reproduce_and_is_reported_as_such(
    package: Path, tmp_path
) -> None:
    """The negative control: an unrelated clean case must stay clean."""
    import shutil

    copy = tmp_path / "clean"
    shutil.copytree(package, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["candidate_behavior"] = Behavior().to_dict()
    (copy / "manifest.json").write_text(json.dumps(manifest))

    assert _run(copy).returncode == EXIT_OK
    assert _run(copy, "--expect-mismatch").returncode == EXIT_NOT_REPRODUCED


# --- clean-room verification ---------------------------------------------------------------------


def test_verification_runs_from_a_fresh_directory_without_importing_evallens(package: Path) -> None:
    result = verify_reproduction(package, expect_mismatch=True)
    assert result.ok is True
    assert result.reproduced is True
    assert result.imported_evallens is False
    assert str(package) not in result.temp_dir
    assert result.exit_code == EXIT_OK


def test_verification_checks_the_default_mode_code_too(package: Path) -> None:
    result = verify_reproduction(package, expect_mismatch=False)
    assert result.ok is True
    assert result.exit_code == EXIT_MISMATCH


def test_verification_rejects_a_directory_without_a_runner(tmp_path) -> None:
    with pytest.raises(ExportError, match=r"does not contain repro\.py"):
        verify_reproduction(tmp_path)


def test_export_refuses_a_weights_hash_that_does_not_match_the_case(
    config: ModelConfig, weights: WeightDict, tmp_path
) -> None:
    """A package whose weights differ from the case's would never reproduce."""
    case = Case.create(
        model_config_id=config.config_id,
        weights_sha256="9" * 64,
        requests=[Request("r0", (5, 6, 7), 1)],
        execution_mode=ExecutionMode.CACHED_DECODE,
        input_seed=1,
    )
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, FAULT)
    real = _failing_case(config, weights)
    spec = ReproductionSpec(
        case=case,
        reference=NativeAdapterSpec("reference", config),
        candidate=NativeAdapterSpec("candidate", config, FAULT),
        policy=POLICY,
        signature=signature_from_failure(real, ("r0",)),
        comparison=compare(reference.run(real), candidate.run(real), POLICY),
        injected_fault=True,
        fault_description="x",
    )
    with pytest.raises(ExportError, match="does not match the case"):
        export_reproduction(spec, tmp_path / "bad")


# --- the demo ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return run_demo(tmp_path_factory.mktemp("demo"), settings=Settings(), max_cases=48)


def test_the_demo_completes_end_to_end(demo) -> None:
    assert demo.succeeded is True, demo.failure_reason
    assert demo.record_path.exists()
    assert demo.repro_dir.exists()


def test_the_demo_runs_every_stage(demo) -> None:
    names = [step.name for step in demo.steps]
    assert names == [
        "search",
        "localize",
        "reduce",
        "verify_reduced_case",
        "export",
        "verify_export",
    ]
    assert all(step.elapsed_s >= 0 for step in demo.steps)


def test_the_demo_detects_a_real_stable_failure(demo) -> None:
    record = demo.record
    assert record["comparison"]["verdict"] == "fail"
    assert record["comparison"]["max_abs_err"] > POLICY.atol
    assert record["detection"]["data"]["cases_examined"] >= 1


def test_the_demo_actually_reduces_the_input(demo) -> None:
    record = demo.record
    original = sum(len(r["token_ids"]) for r in record["original_case"]["requests"])
    reduced = sum(len(r["token_ids"]) for r in record["reduced_case"]["requests"])
    assert reduced < original
    assert record["reduction"]["token_reduction_ratio"] > 1.0
    assert record["reduction"]["n_accepted_steps"] > 0


def test_the_demo_localizes_the_failure(demo) -> None:
    localization = demo.record["localization"]
    assert localization["available"] is True
    assert localization["earliest_observed_str"]
    assert "not proof of root cause" in localization["interpretation"]


def test_the_demo_verifies_its_own_export(demo) -> None:
    export = demo.record["export"]
    assert export["verified"] is True
    assert export["verification"]["imported_evallens"] is False
    assert export["verification"]["reproduced"] is True


def test_the_demo_labels_the_fault_as_injected_everywhere(demo) -> None:
    record = demo.record
    assert record["injected_fault"] is True
    assert "injected" in record["fault_banner"].lower()
    assert record["candidate_behavior"] == DEMO_FAULT.to_dict()

    readme = (demo.repro_dir / "README.md").read_text()
    assert "deliberately injected fault" in readme
    manifest = json.loads((demo.repro_dir / "manifest.json").read_text())
    assert manifest["injected_fault"] is True


def test_the_demo_record_is_viewer_ready(demo) -> None:
    record = json.loads(demo.record_path.read_text())
    for key in (
        "run_id",
        "comparison",
        "original_case",
        "reduced_case",
        "localization",
        "reduction",
        "export",
        "steps",
        "environment",
        "policy",
    ):
        assert key in record, key
    assert record["schema_version"] == 1


def test_the_demo_writes_replayable_failure_bundles(demo) -> None:
    for name in ("failure.json", "reduced.json", "case.json"):
        payload = json.loads((demo.out_dir / name).read_text())
        assert payload
    failure = json.loads((demo.out_dir / "failure.json").read_text())
    assert {"case", "signature", "reference", "candidate", "policy"} <= set(failure)


def test_the_demo_records_real_resource_usage(demo) -> None:
    resources = demo.record["resources"]
    assert resources["peak_rss_bytes"] > 0
    assert resources["exceeded"] is False
    assert resources["peak_rss_bytes"] < resources["limit_bytes"]
