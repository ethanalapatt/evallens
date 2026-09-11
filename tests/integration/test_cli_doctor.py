"""`evallens doctor` end to end, as a reviewer would actually run it."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from evallens.cli import main


def test_doctor_passes_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[PASS]" in out
    assert "fixture vs FP64 NumPy oracle" in out


def test_doctor_json_reports_real_measured_values(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["self_test"]["passed"] is True
    assert 0.0 < payload["self_test"]["max_abs_err"] < payload["self_test"]["tolerance"]
    assert payload["environment"]["torch_version"]
    assert payload["fixture"]["parameter_count"] > 0
    assert len(payload["fixture"]["weights_sha256"]) == 64
    assert payload["rss_bytes"] > 0


def test_doctor_reports_the_real_thread_setting(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--json", "--threads", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["threads"]["torch_num_threads"] == 1


def test_no_subcommand_prints_help_and_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "differential debugger" in capsys.readouterr().out


def test_installed_console_script_runs() -> None:
    """The published command must work as installed, not only as an importable function."""
    result = subprocess.run(
        [sys.executable, "-m", "evallens.cli", "doctor", "--json"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["self_test"]["passed"] is True
