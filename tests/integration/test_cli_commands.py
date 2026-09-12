"""The published command surface, exercised the way the README tells a reviewer to use it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evallens.cli import build_parser, main
from evallens.demo import run_demo
from evallens.settings import ConfigError, Settings
from evallens.viewer import LOOPBACK_HOSTS, viewer_root


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory) -> Path:
    result = run_demo(tmp_path_factory.mktemp("cli-demo"), settings=Settings(), max_cases=48)
    assert result.succeeded, result.failure_reason
    return result.out_dir


# --- help and structure ---------------------------------------------------------------------


def test_every_published_command_exists() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subcommands registered"
    assert set(actions[0].choices) == {
        "doctor",
        "demo",
        "compare",
        "reduce",
        "export",
        "bench",
        "report",
        "view",
    }


@pytest.mark.parametrize("command", ["doctor", "demo", "compare", "reduce", "export", "view"])
def test_each_command_has_useful_help(command: str, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert len(out) > 200, f"{command} help is too thin to be useful"


def test_a_missing_required_argument_fails_clearly(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["compare"])
    assert exit_info.value.code == 2
    assert "--case" in capsys.readouterr().err


# --- compare -----------------------------------------------------------------------------------


def test_compare_on_a_recorded_failure_exits_one(demo_dir: Path, capsys) -> None:
    """Exit 1 means a stable failure — distinct from 2, which means anything else."""
    assert main(["compare", "--case", str(demo_dir / "failure.json")]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "max |Δ|" in out
    assert "injected on purpose" in out


def test_compare_emits_json(demo_dir: Path, capsys) -> None:
    assert main(["compare", "--case", str(demo_dir / "failure.json"), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["stability"]["verdict"] == "fail"
    assert payload["stability"]["stable"] is True
    assert payload["injected_fault"] is True


def test_compare_can_localize(demo_dir: Path, capsys) -> None:
    assert main(["compare", "--case", str(demo_dir / "failure.json"), "--localize"]) == 1
    assert "localization" in capsys.readouterr().out


def test_compare_on_a_clean_case_exits_zero(demo_dir: Path, capsys) -> None:
    """A bare case file defaults to a correct reference/candidate pair, which must pass."""
    assert main(["compare", "--case", str(demo_dir / "case.json")]) == 0
    assert "PASS" in capsys.readouterr().out


# --- reduce --------------------------------------------------------------------------------------


def test_reduce_shrinks_a_recorded_failure(demo_dir: Path, tmp_path, capsys) -> None:
    out = tmp_path / "reduced"
    assert main(["reduce", "--failure", str(demo_dir / "failure.json"), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "ratio" in printed
    assert "minimality" in printed

    bundle = json.loads((out / "failure.json").read_text())
    reduction = json.loads((out / "reduction.json").read_text())
    assert reduction["token_reduction_ratio"] >= 1.0
    assert bundle["injected_fault"] is True
    assert {"case", "signature", "reference", "candidate", "policy"} <= set(bundle)


def test_reduce_refuses_a_case_that_does_not_fail(demo_dir: Path, tmp_path, capsys) -> None:
    """Reducing a passing case would produce a meaningless "reduction"."""
    assert main(["reduce", "--failure", str(demo_dir / "case.json"), "--out", str(tmp_path)]) == 1
    assert "refusing to reduce" in capsys.readouterr().err


def test_reduce_supports_the_greedy_baseline(demo_dir: Path, tmp_path) -> None:
    out = tmp_path / "greedy"
    assert (
        main(
            [
                "reduce",
                "--failure",
                str(demo_dir / "failure.json"),
                "--out",
                str(out),
                "--strategy",
                "greedy",
            ]
        )
        == 0
    )
    assert json.loads((out / "reduction.json").read_text())["strategy"] == "greedy"


# --- export ----------------------------------------------------------------------------------------


def test_export_writes_and_verifies_a_package(demo_dir: Path, tmp_path, capsys) -> None:
    out = tmp_path / "repro"
    assert main(["export", "--failure", str(demo_dir / "reduced.json"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "reproduced the recorded mismatch" in printed
    assert "imported evallens: False" in printed
    assert (out / "repro.py").exists()


def test_export_refuses_a_case_that_does_not_fail(demo_dir: Path, tmp_path, capsys) -> None:
    """An export that does not reproduce is worse than no export."""
    assert (
        main(["export", "--failure", str(demo_dir / "case.json"), "--out", str(tmp_path / "x")])
        == 1
    )
    assert "refusing to export" in capsys.readouterr().err


# --- view ---------------------------------------------------------------------------------------------


def test_view_refuses_a_non_loopback_host(demo_dir: Path, capsys) -> None:
    """The viewer must never be exposed off this machine."""
    assert main(["view", str(demo_dir), "--host", "0.0.0.0"]) == 2
    assert "loopback-only" in capsys.readouterr().err


def test_view_reports_a_missing_record_instead_of_serving_nothing(tmp_path, capsys) -> None:
    assert main(["view", str(tmp_path)]) == 2
    assert "no record.json" in capsys.readouterr().out


def test_view_rejects_a_corrupt_record(tmp_path, capsys) -> None:
    (tmp_path / "record.json").write_text("{not json")
    assert main(["view", str(tmp_path)]) == 2
    assert "not valid JSON" in capsys.readouterr().out


def test_the_viewer_assets_exist_and_are_static() -> None:
    root = viewer_root()
    assert (root / "index.html").exists()
    assert (root / "viewer.css").exists()
    assert (root / "viewer.js").exists()
    assert "127.0.0.1" in LOOPBACK_HOSTS


def test_the_viewer_has_no_external_dependencies() -> None:
    """Offline means offline: no CDN, no remote fonts, no analytics."""
    html = (viewer_root() / "index.html").read_text()
    for marker in ("http://", "https://", "cdn.", "//unpkg"):
        assert marker not in html, f"viewer references {marker}"


# --- settings -------------------------------------------------------------------------------------------


def test_the_shipped_config_loads() -> None:
    settings = Settings.load("configs/cpu.toml")
    assert settings.execution.device == "cpu"
    assert settings.tolerance.atol == 1e-5
    assert settings.reduction.max_queries == 256
    assert settings.model_config().config_id


def test_an_unknown_config_section_is_an_error(tmp_path) -> None:
    """A silently ignored key would change verdicts while looking configured."""
    path = tmp_path / "bad.toml"
    path.write_text("[nonsense]\nvalue = 1\n")
    with pytest.raises(ConfigError, match="unknown section"):
        Settings.load(path)


def test_an_unknown_tolerance_key_is_an_error(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[tolerance]\natol = 1e-5\natoll = 1e-3\n")
    with pytest.raises(ConfigError, match="unknown key"):
        Settings.load(path)


def test_a_missing_config_file_is_an_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        Settings.load("configs/does-not-exist.toml")


def test_config_id_changes_with_the_tolerance(tmp_path) -> None:
    strict = tmp_path / "strict.toml"
    strict.write_text("[tolerance]\natol = 1e-9\n")
    assert Settings.load(strict).config_id != Settings().config_id


# --- viewer contract ------------------------------------------------------------------------------------
#
# The viewer's rendering was NOT visually confirmed in a browser: no browser-automation
# extension was available in this environment (see PROGRESS.md). What is checked here is the
# contract that rendering depends on — every field viewer.js reads must exist in a real
# record, with the type it expects. A field the viewer reads but the record never writes would
# render as "not recorded" forever, and that is the failure this catches.


def test_the_record_supplies_every_field_the_viewer_reads(demo_dir: Path) -> None:
    record = json.loads((demo_dir / "record.json").read_text())

    assert isinstance(record["run_id"], str)
    assert isinstance(record["injected_fault"], bool)
    assert isinstance(record["fault_banner"], str)
    assert isinstance(record["fault_description"], str)
    assert isinstance(record["candidate_behavior_description"], str)
    assert isinstance(record["reference_adapter"], str)
    assert isinstance(record["candidate_adapter"], str)
    assert isinstance(record["weights_sha256"], str)

    assert isinstance(record["comparison"]["verdict"], str)
    assert isinstance(record["comparison"]["max_abs_err"], float)
    assert isinstance(record["comparison"]["detail"], str)
    assert isinstance(record["policy"]["atol"], float)
    assert isinstance(record["policy"]["rtol"], float)
    assert isinstance(record["policy"]["policy_id"], str)

    detection = record["detection"]["data"]
    assert isinstance(detection["cases_examined"], int)
    assert isinstance(detection["seconds_to_detection"], float)

    for key in ("name", "version", "seed", "budget"):
        assert key in record["generator"], key

    for case_key in ("original_case", "reduced_case"):
        case = record[case_key]
        assert isinstance(case["case_id"], str)
        assert isinstance(case["execution_mode"], str)
        for request in case["requests"]:
            assert isinstance(request["request_id"], str)
            assert isinstance(request["token_ids"], list)
            assert isinstance(request["prefix_length"], int)
            assert isinstance(request["pad_left"], int)


def test_the_record_supplies_every_localization_field_the_viewer_reads(demo_dir: Path) -> None:
    localization = json.loads((demo_dir / "record.json").read_text())["localization"]
    assert isinstance(localization["available"], bool)
    assert isinstance(localization["reason"], str)
    assert isinstance(localization["interpretation"], str)
    assert isinstance(localization["reconverged"], bool)
    assert isinstance(localization["n_divergent"], int)
    assert isinstance(localization["n_compared"], int)
    assert isinstance(localization["divergent_layers"], list)
    assert isinstance(localization["comparisons_truncated"], bool)
    assert isinstance(localization["alignment"]["n_matched"], int)
    assert isinstance(localization["alignment"]["fully_aligned"], bool)

    comparison = localization["comparisons"][0]
    assert isinstance(comparison["address_str"], str)
    assert isinstance(comparison["diverged"], bool)
    assert isinstance(comparison["values_available"], bool)
    if comparison["diff"]:
        assert isinstance(comparison["diff"]["max_abs_err"], float)
        assert isinstance(comparison["diff"]["n_violations"], int)
        assert isinstance(comparison["diff"]["n_elements"], int)


def test_the_record_supplies_every_reduction_and_export_field_the_viewer_reads(
    demo_dir: Path,
) -> None:
    record = json.loads((demo_dir / "record.json").read_text())

    reduction = record["reduction"]
    assert isinstance(reduction["token_reduction_ratio"], float)
    assert isinstance(reduction["minimality"], str)
    assert isinstance(reduction["minimality_note"], str)
    assert isinstance(reduction["wall_time_s"], float)
    for key in ("logical_queries", "model_runs", "cache_hits"):
        assert isinstance(reduction["counters"][key], int), key
    step = next(s for s in reduction["steps"] if s["accepted"])
    for key in ("operation", "strategy", "queries_at_step"):
        assert key in step, key
    for key in ("n_requests", "n_valid_tokens", "n_padding_tokens"):
        assert key in step["before"] and key in step["after"], key

    export = record["export"]
    assert isinstance(export["path"], str)
    assert isinstance(export["verified"], bool)
    verification = export["verification"]
    assert isinstance(verification["exit_code"], int)
    assert isinstance(verification["expected_exit_code"], int)
    assert isinstance(verification["imported_evallens"], bool)
    assert isinstance(verification["ran_from"], str)

    replay = record["subprocess_replay"]
    assert isinstance(replay["verdict"], str)
    assert isinstance(replay["stable"], bool)

    environment = record["environment"]
    for key in ("python_version", "torch_version", "numpy_version", "platform", "cpu_count"):
        assert environment[key] is not None, key
    resources = record["resources"]
    assert isinstance(resources["peak_rss_mib"], float)
    assert isinstance(resources["limit_mib"], float)


def test_the_viewer_never_hardcodes_a_measurement() -> None:
    """Guard against illustrative numbers creeping into the UI."""
    source = (viewer_root() / "viewer.js").read_text()
    assert "not recorded" in source, "the viewer must have an explicit missing-value marker"
    for forbidden in ("Math.random", "sampleData", "exampleRecord", "0.95", "99%"):
        assert forbidden not in source, f"viewer.js contains {forbidden!r}"


# --- bench and report delegation ------------------------------------------------------------


def test_bench_rejects_an_unknown_preset_and_names_the_real_ones(capsys) -> None:
    code = main(["bench", "--preset", "no-such-preset", "--out", "unused"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no-such-preset" in err
    assert "smoke" in err and "full" in err


def test_bench_and_report_run_a_real_smoke_study_end_to_end(tmp_path) -> None:
    """The two commands the README gives a reviewer, on an actual (tiny) study.

    This is the only test that runs the benchmark through the CLI rather than through
    `bench.run` directly, so it is what catches a delegation bug -- a dropped flag, or a
    `bench` package the console script cannot import.
    """
    run_dir = tmp_path / "smoke"
    assert main(["bench", "--preset", "smoke", "--out", str(run_dir), "--quiet"]) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["synthetic"] is False
    assert manifest["expected_trial_ids"], "the trial matrix must be frozen before the run"

    out = tmp_path / "SMOKE.md"
    metrics = tmp_path / "metrics.json"
    # --allow-dirty because a test run is made from whatever tree the developer has; the
    # published RESULTS.md is generated without it, which is what makes it publishable.
    code = main(
        ["report", str(run_dir), "--out", str(out), "--allow-dirty", "--metrics", str(metrics)]
    )
    assert code == 0
    text = out.read_text()
    assert "injected" in text.lower()
    assert json.loads(metrics.read_text())["detection"]["trials"] > 0


def test_report_refuses_a_run_directory_that_is_not_one(tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["report", str(empty), "--out", str(tmp_path / "x.md")]) != 0
    assert not (tmp_path / "x.md").exists()


def test_the_checkout_root_is_found_from_a_subdirectory(monkeypatch, tmp_path) -> None:
    """`bench` is not installed, so the CLI locates it relative to the working directory."""
    from evallens.cli import _checkout_root

    monkeypatch.chdir(Path(__file__).parent)
    root = _checkout_root()
    assert root is not None
    assert (root / "bench" / "run.py").is_file()

    monkeypatch.chdir(tmp_path)
    assert _checkout_root() is None
