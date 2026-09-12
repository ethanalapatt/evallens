"""The viewer's real render path, executed.

`test_cli_commands.py` checks the *contract* — that every field `viewer.js` reads exists in a
real record. These tests are stronger and answer a different question: does running the actual
viewer source over an actual record produce the page it claims to?

No browser-automation extension was available on this machine, so the rendered page has still
never been confirmed visually (layout, CSS, dark mode). That limitation is recorded in
PROGRESS.md. What is verified here is the JavaScript: the render executes, the element ids it
reaches for exist in index.html, real recorded values reach the output, and the empty and error
states are selected correctly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evallens.demo import run_demo
from evallens.settings import Settings

HARNESS = Path(__file__).resolve().parents[1] / "viewer" / "render.mjs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _render(argument: str) -> dict:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, str(HARNESS), argument], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, f"harness failed: {proc.stderr}"
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def record_path(tmp_path_factory) -> Path:
    result = run_demo(tmp_path_factory.mktemp("viewer-demo"), settings=Settings(), max_cases=48)
    assert result.succeeded, result.failure_reason
    return result.out_dir / "record.json"


@pytest.fixture(scope="module")
def rendered(record_path: Path) -> dict:
    return _render(str(record_path))


def test_the_viewer_renders_a_real_record_without_throwing(rendered: dict) -> None:
    assert rendered["error"] is None
    assert rendered["section"] == "content"
    assert len(rendered["html"]) > 2000


def test_every_element_the_viewer_reaches_for_exists_in_the_markup(rendered: dict) -> None:
    """The harness raises if `viewer.js` requests an id `index.html` does not declare.

    A browser would not raise; it would silently render a blank panel, which is exactly the
    class of bug a contract test over the *record* cannot catch.
    """
    assert rendered["error"] is None
    assert {"content", "loading", "empty", "error", "error-detail"} <= set(rendered["declaredIds"])


def test_the_injected_fault_banner_is_rendered_not_merely_stored(rendered: dict) -> None:
    html = rendered["html"]
    assert "Deliberately injected fault" in html
    assert "injected on purpose" in html
    assert "not a bug discovered in PyTorch" in html


def test_recorded_values_reach_the_page(rendered: dict, record_path: Path) -> None:
    record = json.loads(record_path.read_text())
    html = rendered["html"]
    assert record["weights_sha256"] in html
    assert record["model_config"]["config_id"] in html
    assert record["policy"]["policy_id"] in html
    assert record["run_id"] in html
    assert record["comparison"]["verdict"].upper() in html


def test_the_verdict_is_labeled_in_text_not_only_by_color(rendered: dict) -> None:
    html = rendered["html"]
    assert "FAIL" in html or "PASS" in html
    assert "within tolerance" in html or "diverged" in html


def test_both_the_original_and_the_reduced_case_are_shown(
    rendered: dict, record_path: Path
) -> None:
    record = json.loads(record_path.read_text())
    html = rendered["html"]
    assert record["original_case"]["case_id"] in html
    assert record["reduced_case"]["case_id"] in html
    assert "Original failing input" in html
    assert "Reduced input" in html


def test_the_minimality_claim_is_qualified_on_the_page(rendered: dict) -> None:
    assert "declared deletion operations" in rendered["html"]
    assert "globally smallest" in rendered["html"]


def test_a_missing_record_selects_the_empty_state_not_an_error(rendered: dict) -> None:
    result = _render("--missing")
    assert result["error"] is None
    assert result["section"] == "empty"
    assert result["html"] == ""


def test_a_record_that_is_not_a_run_record_selects_the_error_state() -> None:
    result = _render("--corrupt")
    assert result["error"] is None
    assert result["section"] == "error"
    assert "does not look like an EvalLens run record" in result["errorDetail"]


def test_a_record_with_missing_sections_renders_not_recorded(tmp_path, record_path: Path) -> None:
    """A viewer that invents a plausible default is worse than one that shows nothing."""
    record = json.loads(record_path.read_text())
    for key in ("localization", "reduction", "export", "subprocess_replay"):
        record.pop(key, None)
    record["weights_sha256"] = None
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps(record))

    result = _render(str(sparse))
    assert result["error"] is None
    assert result["section"] == "content"
    assert result["html"].count("not recorded") >= 4
    # and it must not have invented anything in their place
    assert "NaN" not in result["html"]
    assert "undefined" not in result["html"]
