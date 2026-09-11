"""The search policy must not be able to see the answer key.

A generator or reducer that can read mutant ids, source patches, trigger fixtures, or
expected verdicts is not being measured — it is being told. This is the structural property
that makes every detection number in the benchmark mean something, so it is enforced two
ways: statically, by reading the source, and dynamically, by importing the modules in a clean
interpreter and inspecting what got loaded.

Do not weaken these tests to make an implementation convenient.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SEARCH_POLICY_MODULES = ["evallens.generate", "evallens.reduce"]
"""Every module that drives the search and must not be able to see the answer key."""

FORBIDDEN_IMPORT_ROOTS = {"bench"}
FORBIDDEN_NAMES = {"Behavior", "MutantSpec", "MUTANTS", "CONTROLS", "qualify_mutant"}

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "evallens"


def _module_path(module: str) -> Path:
    return SOURCE_ROOT / (module.split(".", 1)[1] + ".py")


def _existing_policy_modules() -> list[str]:
    return [module for module in SEARCH_POLICY_MODULES if _module_path(module).exists()]


def test_the_policy_module_list_is_not_empty() -> None:
    """Guard against this whole file silently passing because nothing is checked."""
    assert _existing_policy_modules(), "no search-policy module exists to check"


@pytest.mark.parametrize("module", SEARCH_POLICY_MODULES)
def test_source_contains_no_import_of_the_fault_corpus(module: str) -> None:
    path = _module_path(module)
    if not path.exists():
        pytest.skip(f"{module} is not implemented yet")
    tree = ast.parse(path.read_text(), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in FORBIDDEN_IMPORT_ROOTS, f"{module} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in FORBIDDEN_IMPORT_ROOTS, f"{module} imports from {node.module}"
            for alias in node.names:
                assert alias.name not in FORBIDDEN_NAMES, (
                    f"{module} imports {alias.name!r}, which belongs to the scoring layer"
                )


@pytest.mark.parametrize("module", SEARCH_POLICY_MODULES)
def test_source_never_mentions_a_mutant_identity(module: str) -> None:
    """A string comparison against a mutant id would evade the import check."""
    path = _module_path(module)
    if not path.exists():
        pytest.skip(f"{module} is not implemented yet")
    source = path.read_text()
    for marker in ("mutant", "MUTANT", "injected", "trigger fixture"):
        assert marker not in source.replace("mutant corpus", "").replace("which candidate", ""), (
            f"{module} mentions {marker!r}"
        )


@pytest.mark.parametrize("module", SEARCH_POLICY_MODULES)
def test_importing_a_search_module_does_not_load_the_fault_corpus(module: str) -> None:
    """Dynamic check: a transitive import would still put `bench` into sys.modules."""
    script = (
        f"import sys; import {module}; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'bench'); "
        "print(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"{module} loaded fault-corpus modules: {result.stdout}"


def test_the_reducer_reaches_models_only_through_the_adapter_protocol() -> None:
    """It must not need a ModelConfig, weights, or anything else that implies fixture access."""
    import inspect

    from evallens.reduce import FailurePredicate

    parameters = set(inspect.signature(FailurePredicate.__init__).parameters)
    assert parameters == {
        "self",
        "reference",
        "candidate",
        "policy",
        "signature",
        "budget",
        "counters",
        "environment_id",
    }
    assert not parameters & {"config", "weights", "behavior", "mutant", "model"}


def test_the_generator_api_never_receives_a_behavior() -> None:
    """Capability is the only thing a generator is told, and it carries no fault information."""
    from evallens.generate import GeneratorCapability

    fields = set(GeneratorCapability.__dataclass_fields__)
    assert not fields & {"behavior", "mutant_id", "expected_verdict", "adapter"}
    assert fields == {
        "model_config_id",
        "weights_sha256",
        "vocab_size",
        "max_tokens_per_request",
        "max_batch_rows",
        "max_session_requests",
        "max_pad_left",
        "categories",
    }


def test_generated_cases_carry_no_fault_information() -> None:
    from evallens.fixtures.config import UNIT_FIXTURE
    from evallens.generate import GeneratorCapability, UniformValidGenerator

    capability = GeneratorCapability.for_fixture(UNIT_FIXTURE, "a" * 64)
    for case in UniformValidGenerator(capability).generate(12, seed=1):
        blob = str(case.to_dict()).lower()
        assert "mutant" not in blob
        assert "behavior" not in blob
        assert "fault" not in blob
