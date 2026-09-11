"""Portable reproduction packages.

An exported package is a directory someone else can unzip on a machine that has never heard
of EvalLens, run one command, and see the same mismatch.

How the model gets in there
---------------------------
The fixture source is **copied** out of the installed package and its imports rewritten into
a local `repro_fixture` package. The tempting alternative — hand-writing a separate "minimal"
transformer for reproductions — produces two implementations that drift, and a reproduction
that runs a *different* model from the one that found the bug is worse than no reproduction.
Copying means the exported model is byte-for-byte the model under test.

Independence, and how it is checked
-----------------------------------
`repro.py` must not import the installed EvalLens package and must not reference absolute
paths into the original checkout. Both are verified, and `repro.py` also checks itself: after
its imports it asserts that no `evallens` module is loaded, and exits with a setup error if
one is. Verification copies the package to a fresh temporary directory outside the checkout
and runs it in an isolated interpreter.

Exit codes are deliberately distinct, because "the reproduction script crashed on import" and
"the reproduction reproduced the mismatch" must never be confused:

===== ==========================================================================
0     As expected: default mode found no mismatch, or `--expect-mismatch` did
1     Default mode found a mismatch (documented, expected nonzero)
2     Setup error: inputs missing, hashes wrong, or EvalLens leaked into the run
3     Execution error: the comparison itself raised
4     `--expect-mismatch` was requested and the mismatch did **not** reproduce
===== ==========================================================================

Weights travel as a non-object NPZ and are verified by hash before anything runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import evallens
from evallens.adapters.native import NativeAdapterSpec
from evallens.env import capture_environment
from evallens.fixtures.config import ModelConfig, make_weights, weights_sha256
from evallens.types import Case, ComparisonResult, FailureSignature, TolerancePolicy

EXPORT_SCHEMA_VERSION = 1
REPRO_PACKAGE = "repro_fixture"

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_SETUP_ERROR = 2
EXIT_EXECUTION_ERROR = 3
EXIT_NOT_REPRODUCED = 4

COPIED_MODULES: dict[str, str] = {
    "types.py": "types.py",
    "compare.py": "compare.py",
    "fixtures/config.py": "config.py",
    "fixtures/behavior.py": "behavior.py",
    "fixtures/tiny_transformer.py": "tiny_transformer.py",
    "adapters/encoding.py": "encoding.py",
    "adapters/native.py": "native.py",
}
"""Source path inside ``evallens`` → flattened path inside the exported package."""

IMPORT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "from evallens.fixtures.tiny_transformer import",
        f"from {REPRO_PACKAGE}.tiny_transformer import",
    ),
    ("from evallens.fixtures.behavior import", f"from {REPRO_PACKAGE}.behavior import"),
    ("from evallens.fixtures.config import", f"from {REPRO_PACKAGE}.config import"),
    ("from evallens.adapters.encoding import", f"from {REPRO_PACKAGE}.encoding import"),
    ("from evallens.compare import", f"from {REPRO_PACKAGE}.compare import"),
    ("from evallens.types import", f"from {REPRO_PACKAGE}.types import"),
)


class ExportError(RuntimeError):
    """Raised when a package cannot be written or is self-evidently incomplete."""


@dataclass(frozen=True, slots=True)
class ReproductionSpec:
    """Everything an exported package needs to describe and re-run one failure."""

    case: Case
    reference: NativeAdapterSpec
    candidate: NativeAdapterSpec
    policy: TolerancePolicy
    signature: FailureSignature
    comparison: ComparisonResult
    injected_fault: bool
    fault_description: str
    original_case: Case | None = None
    localization_summary: str = ""
    reduction_summary: dict[str, Any] | None = None

    @property
    def config(self) -> ModelConfig:
        return self.candidate.config


def _rewrite_imports(source: str) -> str:
    for original, replacement in IMPORT_REWRITES:
        source = source.replace(original, replacement)
    return source


def _copy_fixture_sources(destination: Path) -> list[str]:
    """Copy the real fixture sources, rewriting their imports for the flattened package."""
    package_root = Path(evallens.__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for source_path, target_name in COPIED_MODULES.items():
        source_file = package_root / source_path
        if not source_file.exists():
            raise ExportError(f"cannot export: missing source file {source_file}")
        text = _rewrite_imports(source_file.read_text(encoding="utf-8"))
        if "from evallens" in text or "import evallens" in text:
            raise ExportError(
                f"{source_path} still imports evallens after rewriting; the export would not "
                "be self-contained"
            )
        (destination / target_name).write_text(text, encoding="utf-8")
        written.append(target_name)

    (destination / "__init__.py").write_text(
        '"""Vendored EvalLens fixture sources.\n\n'
        "Copied verbatim from the EvalLens package that produced this reproduction, with\n"
        "imports rewritten for this flattened layout. This is the same model code that found\n"
        "the mismatch, not a re-implementation of it.\n"
        '"""\n',
        encoding="utf-8",
    )
    written.append("__init__.py")
    return sorted(written)


def _write_weights(path: Path, config: ModelConfig) -> str:
    """Write the canonical weights as a non-object NPZ and return their hash."""
    weights = make_weights(config)
    digest = weights_sha256(weights)
    arrays = {
        name: np.ascontiguousarray(value, dtype=np.float32) for name, value in weights.items()
    }
    # numpy's stub types savez's second positional parameter as `allow_pickle`, so a
    # **kwargs expansion of named arrays does not type-check even though it is the documented
    # calling convention.
    np.savez(str(path), **arrays)  # type: ignore[arg-type]
    # allow_pickle stays off on load; confirm nothing object-typed slipped in.
    with np.load(path, allow_pickle=False) as loaded:
        for name in arrays:
            if loaded[name].dtype != np.float32:
                raise ExportError(f"weight {name!r} did not round-trip as float32")
    return digest


REPRO_SCRIPT = '''#!/usr/bin/env python3
"""Standalone reproduction of a recorded inference mismatch.

Run:
    python repro.py                    # compare; nonzero exit (1) if the mismatch is present
    python repro.py --expect-mismatch  # exit 0 only if the recorded mismatch reproduces
    python repro.py --json             # machine-readable result

This script does not import EvalLens. It checks that itself: if an `evallens` module is
loaded, it exits with a setup error rather than reporting a result that came from somewhere
other than the vendored sources next to this file.

Exit codes:
    0  as expected      1  mismatch found (default mode)      2  setup error
    3  execution error  4  --expect-mismatch did not reproduce
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_SETUP_ERROR = 2
EXIT_EXECUTION_ERROR = 3
EXIT_NOT_REPRODUCED = 4

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def fail_setup(message: str) -> int:
    print(f"SETUP ERROR: {message}", file=sys.stderr)
    return EXIT_SETUP_ERROR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-mismatch",
        action="store_true",
        help="Exit 0 only when the recorded mismatch reproduces.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    parser.add_argument(
        "--replays", type=int, default=3, help="Runs from clean state (default: 3)."
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch  # noqa: F401
    except Exception as exc:
        return fail_setup(f"missing dependency: {exc}")

    try:
        manifest = json.loads((HERE / "manifest.json").read_text())
        case_payload = json.loads((HERE / "case.json").read_text())
        policy_payload = json.loads((HERE / "policy.json").read_text())
    except Exception as exc:
        return fail_setup(f"cannot read reproduction inputs: {exc}")

    try:
        from repro_fixture.compare import compare
        from repro_fixture.config import ModelConfig, weights_sha256
        from repro_fixture.behavior import Behavior
        from repro_fixture.native import CandidateAdapter, ReferenceAdapter
        from repro_fixture.types import Case, TolerancePolicy, Verdict
    except Exception as exc:
        return fail_setup(f"cannot import the vendored fixture: {exc}")

    leaked = sorted(m for m in sys.modules if m == "evallens" or m.startswith("evallens."))
    if leaked:
        return fail_setup(
            "this reproduction imported the installed EvalLens package "
            f"({', '.join(leaked)}); it is meant to be self-contained"
        )

    try:
        config = ModelConfig.from_dict(manifest["model_config"])
        with np.load(HERE / "weights.npz", allow_pickle=False) as loaded:
            weights = {name: np.asarray(loaded[name], dtype=np.float32) for name in loaded.files}
    except Exception as exc:
        return fail_setup(f"cannot load weights: {exc}")

    digest = weights_sha256(weights)
    if digest != manifest["weights_sha256"]:
        return fail_setup(
            f"weights hash mismatch: file is {digest[:16]}..., manifest records "
            f"{manifest['weights_sha256'][:16]}..."
        )

    try:
        case = Case.from_dict(case_payload)
        policy = TolerancePolicy.from_dict(policy_payload)
        reference = ReferenceAdapter(config, weights)
        candidate = CandidateAdapter(
            config, weights, Behavior.from_dict(manifest["candidate_behavior"])
        )
    except Exception as exc:
        return fail_setup(f"cannot build the comparison: {exc}")

    verdicts = []
    results = []
    try:
        for _ in range(max(1, args.replays)):
            reference.reset()
            candidate.reset()
            result = compare(reference.run(case), candidate.run(case), policy)
            verdicts.append(result.verdict)
            results.append(result)
    except Exception as exc:
        print(f"EXECUTION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_ERROR

    stable = len(set(verdicts)) == 1
    result = results[0]
    mismatch = stable and result.verdict is Verdict.FAIL

    payload = {
        "case_id": case.case_id,
        "verdict": result.verdict.value,
        "stable": stable,
        "mismatch_reproduced": mismatch,
        "max_abs_err": result.max_abs_err,
        "failing_requests": list(result.failing_request_ids),
        "detail": result.detail,
        "expected_max_abs_err": manifest["recorded"]["max_abs_err"],
        "injected_fault": manifest["injected_fault"],
        "fault_description": manifest["fault_description"],
        "replays": len(verdicts),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"case            {case.case_id}")
        print(f"reference       {reference.adapter_id}")
        print(f"candidate       {candidate.adapter_id}")
        print(f"policy          atol={policy.atol:g} rtol={policy.rtol:g}")
        print(f"replays         {len(verdicts)} from clean state, stable={stable}")
        print(f"verdict         {result.verdict.value.upper()}")
        print(f"max |Δ|         {result.max_abs_err:.6e}  (recorded {manifest['recorded']['max_abs_err']:.6e})")
        print(f"detail          {result.detail}")
        if manifest["injected_fault"]:
            print()
            print("NOTE: this is a DELIBERATELY INJECTED fault, not a bug discovered in any")
            print(f"      third-party library. {manifest['fault_description']}")

    if args.expect_mismatch:
        return EXIT_OK if mismatch else EXIT_NOT_REPRODUCED
    return EXIT_MISMATCH if mismatch else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
'''


def _readme(spec: ReproductionSpec, manifest: dict[str, Any]) -> str:
    fault_banner = (
        "> **This is a deliberately injected fault.** It was introduced by EvalLens as test\n"
        "> material for its own detector. It is not a bug discovered in PyTorch or any other\n"
        "> third-party library.\n"
        if spec.injected_fault
        else "> This reproduction records an observed mismatch. It is an observation to\n"
        "> investigate, not an established third-party bug.\n"
    )
    reduction = ""
    if spec.reduction_summary:
        summary = spec.reduction_summary
        reduction = (
            f"\n## How this input was found\n\n"
            f"The original failing case had {summary['original_tokens']} valid tokens across "
            f"{summary['original_requests']} request(s). EvalLens reduced it to "
            f"{summary['reduced_tokens']} token(s) across {summary['reduced_requests']} "
            f"request(s) — a {summary['ratio']:.1f}x token reduction — using "
            f"{summary['logical_queries']} predicate queries. Minimality: "
            f"`{summary['minimality']}`.\n\n"
            f"This input is the output of the actual reducer. It was not handwritten.\n"
        )
    localization = (
        f"\n## Where the divergence first becomes visible\n\n{spec.localization_summary}\n\n"
        "That is evidence about where a difference becomes observable at the checkpoints these\n"
        "adapters expose. It is not proof of root cause.\n"
        if spec.localization_summary
        else ""
    )

    return f"""# Reproduction: {spec.case.case_id}

{fault_banner}
A reference implementation and a candidate implementation, given identical weights and
identical inputs, produce different outputs. Nothing crashes and the shapes agree.

**What differs:** {spec.fault_description}

**Recorded result:** `{spec.comparison.verdict.value.upper()}`, max absolute error
`{spec.comparison.max_abs_err:.6e}` against the policy `atol={spec.policy.atol:g},
rtol={spec.policy.rtol:g}`. Failing request(s): {", ".join(spec.comparison.failing_request_ids) or "none"}.

## Run it

Requires Python 3.11+, PyTorch, and NumPy. Nothing else — in particular, **not** EvalLens.

```bash
python repro.py --expect-mismatch   # exits 0 only if the recorded mismatch reproduces
python repro.py                     # exits 1 if the mismatch is present, 0 if it is not
python repro.py --json              # machine-readable
```

Exit codes are distinct on purpose, so a script that fails to start is never mistaken for a
mismatch that reproduced:

| Code | Meaning |
|---|---|
| 0 | As expected |
| 1 | Default mode found the mismatch |
| 2 | Setup error (missing inputs, wrong hashes, or EvalLens leaked into the run) |
| 3 | Execution error |
| 4 | `--expect-mismatch` was requested and the mismatch did not reproduce |
{reduction}{localization}
## What is in here

| Path | Contents |
|---|---|
| `repro.py` | The runner. Imports only the vendored sources next to it. |
| `repro_fixture/` | The fixture and adapter sources, copied from the EvalLens build that produced this package. Same code, not a re-implementation. |
| `case.json` | The exact input: tokens, prefill boundary, padding, execution mode. |
| `weights.npz` | Deterministic float32 weights, no pickled objects. Verified by SHA-256 before use. |
| `policy.json` | The numerical tolerance policy in force. |
| `manifest.json` | Hashes, adapter identities, source commit, and the recorded result. |

## Provenance

| Field | Value |
|---|---|
| EvalLens version | `{manifest["evallens_version"]}` |
| Source commit | `{manifest["source_commit"] or "unknown"}` |
| Source clean | `{manifest["source_clean"]}` |
| Original case | `{manifest["original_case_id"] or "n/a"}` |
| Reduced case | `{spec.case.case_id}` |
| Model config | `{spec.config.config_id}` |
| Weights SHA-256 | `{manifest["weights_sha256"]}` |
| Exported | `{manifest["created_at"]}` |
| Produced on | {manifest["environment"]["platform"]}, Python {manifest["environment"]["python_version"]}, torch {manifest["environment"]["torch_version"]} |

The model is a small transformer with **random, untrained weights**. It demonstrates
execution correctness, not language capability.
"""


def export_reproduction(spec: ReproductionSpec, out_dir: str | Path) -> Path:
    """Write a self-contained reproduction package and return its directory."""
    destination = Path(out_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    vendored = _copy_fixture_sources(destination / REPRO_PACKAGE)
    digest = _write_weights(destination / "weights.npz", spec.config)
    if digest != spec.case.weights_sha256:
        raise ExportError(
            f"exported weights hash {digest[:16]}... does not match the case's recorded "
            f"{spec.case.weights_sha256[:16]}...; the package would not reproduce"
        )

    environment = capture_environment()
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "evallens_version": evallens.__version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": environment.git_commit,
        "source_clean": environment.git_dirty is False,
        "injected_fault": spec.injected_fault,
        "fault_description": spec.fault_description,
        "model_config": spec.config.to_dict(),
        "weights_sha256": digest,
        "reference_adapter": spec.reference.to_dict(),
        "candidate_adapter": spec.candidate.to_dict(),
        "candidate_behavior": spec.candidate.behavior.to_dict(),
        "original_case_id": spec.original_case.case_id if spec.original_case else None,
        "reduced_case_id": spec.case.case_id,
        "case_content_hash": spec.case.content_hash(),
        "failure_signature": spec.signature.to_dict(),
        "recorded": {
            "verdict": spec.comparison.verdict.value,
            "max_abs_err": spec.comparison.max_abs_err,
            "failing_request_ids": list(spec.comparison.failing_request_ids),
            "detail": spec.comparison.detail,
        },
        "localization_summary": spec.localization_summary,
        "reduction_summary": spec.reduction_summary,
        "environment": environment.to_dict(),
        "vendored_modules": vendored,
        "exit_codes": {
            "ok": EXIT_OK,
            "mismatch": EXIT_MISMATCH,
            "setup_error": EXIT_SETUP_ERROR,
            "execution_error": EXIT_EXECUTION_ERROR,
            "not_reproduced": EXIT_NOT_REPRODUCED,
        },
    }

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (destination / "case.json").write_text(
        json.dumps(spec.case.to_dict(), indent=2, sort_keys=True)
    )
    (destination / "policy.json").write_text(
        json.dumps(spec.policy.to_dict(), indent=2, sort_keys=True)
    )
    (destination / "repro.py").write_text(REPRO_SCRIPT, encoding="utf-8")
    (destination / "README.md").write_text(_readme(spec, manifest), encoding="utf-8")

    _assert_no_absolute_checkout_paths(destination)
    return destination


def _assert_no_absolute_checkout_paths(destination: Path) -> None:
    """A package that points back at this checkout is not portable."""
    checkout = str(Path(evallens.__file__).resolve().parents[2])
    for path in destination.rglob("*"):
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if checkout in text and path.name != "manifest.json":
            raise ExportError(
                f"{path.name} references the original checkout path; the package is not portable"
            )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of running an exported package in a clean environment."""

    ok: bool
    exit_code: int
    expected_exit_code: int
    reproduced: bool
    imported_evallens: bool
    stdout: str
    stderr: str
    elapsed_s: float
    temp_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "expected_exit_code": self.expected_exit_code,
            "reproduced": self.reproduced,
            "imported_evallens": self.imported_evallens,
            "elapsed_s": round(self.elapsed_s, 3),
            "ran_from": self.temp_dir,
            "stderr_tail": self.stderr[-400:],
        }


def verify_reproduction(
    package_dir: str | Path,
    *,
    expect_mismatch: bool = True,
    timeout_s: float = 300.0,
) -> VerificationResult:
    """Run an exported package from a fresh temporary directory with `PYTHONPATH` cleared.

    Copying to a temp directory outside the checkout is the point: a package that only works
    where it was built is not portable, and running it in place would never reveal that.
    """
    source = Path(package_dir).resolve()
    if not (source / "repro.py").exists():
        raise ExportError(f"{source} does not contain repro.py")

    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="evallens-repro-") as temp:
        staged = Path(temp) / "package"
        shutil.copytree(source, staged)
        command = [sys.executable, "-I", "repro.py", "--json"]
        if expect_mismatch:
            command.append("--expect-mismatch")
        completed = subprocess.run(
            command,
            cwd=staged,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        temp_dir = str(staged)

    elapsed = (time.perf_counter_ns() - started) / 1e9
    expected = EXIT_OK if expect_mismatch else EXIT_MISMATCH

    reproduced = False
    with contextlib.suppress(json.JSONDecodeError, KeyError):
        reproduced = bool(json.loads(completed.stdout)["mismatch_reproduced"])

    return VerificationResult(
        ok=completed.returncode == expected,
        exit_code=completed.returncode,
        expected_exit_code=expected,
        reproduced=reproduced,
        imported_evallens="imported the installed EvalLens package" in completed.stderr,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_s=elapsed,
        temp_dir=temp_dir,
    )


__all__ = [
    "EXIT_EXECUTION_ERROR",
    "EXIT_MISMATCH",
    "EXIT_NOT_REPRODUCED",
    "EXIT_OK",
    "EXIT_SETUP_ERROR",
    "EXPORT_SCHEMA_VERSION",
    "REPRO_PACKAGE",
    "ExportError",
    "ReproductionSpec",
    "VerificationResult",
    "export_reproduction",
    "verify_reproduction",
]
