"""Documentation is checked the same way results are: against the source of truth.

Two things are verified here. Every relative link in the project's markdown resolves to a file
that exists, so a reviewer following an evidence link never lands on a 404. And every headline
figure quoted in prose appears verbatim in the generated `RESULTS.md`, so documentation cannot
drift away from the measurements it describes -- regenerate the report with different numbers
and these tests fail until the prose is corrected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN = sorted(
    [
        ROOT / "README.md",
        ROOT / "SPEC.md",
        ROOT / "PROGRESS.md",
        ROOT / "CLAUDE.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "examples").rglob("*.md"),
    ]
)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(path: Path) -> None:
    broken = []
    for target in LINK.findall(path.read_text()):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(ROOT)} links to missing files: {broken}"


@pytest.fixture(scope="module")
def results() -> str:
    path = ROOT / "RESULTS.md"
    if not path.exists():
        pytest.skip("RESULTS.md has not been generated yet")
    return path.read_text()


def _spellings(figure: str) -> set[str]:
    """The ways prose may legitimately write a figure the report prints as ``a/b``.

    Notation is allowed to differ -- "108 of 160" reads better in a sentence than "108/160" --
    but the *numbers* are not. Thousands separators are covered because the report writes 2560
    and English prose writes 2,560.
    """
    forms = {figure}
    if "/" in figure:
        numerator, denominator = figure.split("/", 1)
        forms |= {f"{numerator} of {denominator}", f"{numerator}/{denominator}"}
        with_commas = (f"{int(n):,}" for n in (numerator, denominator) if n.isdigit())
        grouped = list(with_commas)
        if len(grouped) == 2:
            forms |= {f"{grouped[0]}/{grouped[1]}", f"{grouped[0]} of {grouped[1]}"}
    return forms


@pytest.mark.parametrize(
    "figure",
    [
        "160/160",  # detection within budget
        "0/2560",  # false positives on known-good cases
        "108/160",  # stable failures that reconverge
        "8/8",  # clean-room reproduction
        "6.2x",  # ddmin's query-cost advantage over greedy
        "35.2x",  # median reduction ratio
    ],
)
def test_the_headline_figures_quoted_in_prose_are_in_the_generated_report(
    figure: str, results: str
) -> None:
    """Each of these is stated somewhere in README.md or docs/; none may be hand-authored."""
    assert figure in results, f"{figure!r} is quoted in prose but is not in RESULTS.md"
    prose = "\n".join(
        (ROOT / name).read_text()
        for name in (
            "README.md",
            "docs/ARCHITECTURE.md",
            "docs/SCOPE.md",
            "docs/INTERVIEW_GUIDE.md",
        )
    )
    assert any(spelling in prose for spelling in _spellings(figure)), (
        f"{figure!r} is no longer quoted anywhere; drop it from this test"
    )


def test_the_injected_fault_disclosure_appears_everywhere_results_are_shown(results: str) -> None:
    """An injected fault must never be presented as an upstream discovery."""
    for path in (
        ROOT / "README.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "SCOPE.md",
        ROOT / "docs" / "INTERVIEW_GUIDE.md",
        ROOT / "docs" / "RECORDING.md",
    ):
        text = path.read_text().lower()
        assert "inject" in text, f"{path.name} shows results without disclosing injected faults"
    assert "injected" in results.lower()
    assert "not a bug" in results.lower() or "none of them is a bug" in results.lower()


def test_no_document_claims_a_globally_smallest_reduction() -> None:
    for path in MARKDOWN:
        text = path.read_text()
        for phrase in ("globally smallest counterexample is", "smallest possible input"):
            assert phrase not in text, f"{path.name} overclaims minimality: {phrase!r}"


def test_the_verified_viewer_screenshot_is_committed_and_documented() -> None:
    """The visual acceptance gate needs binary evidence, not only a prose claim.

    Parse the PNG's IHDR directly so the check stays dependency-free and cannot pass on an
    empty placeholder or a renamed text file.
    """
    screenshot = ROOT / "docs" / "assets" / "viewer-demo.png"
    data = screenshot.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    assert int.from_bytes(data[16:20], "big") == 1440
    assert int.from_bytes(data[20:24], "big") == 4200

    recording = (ROOT / "docs" / "RECORDING.md").read_text()
    assert "assets/viewer-demo.png" in recording
    for stale_claim in ("viewer page has never been seen", "no screenshot, gif, or video"):
        assert stale_claim not in recording.lower()
