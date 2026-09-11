#!/usr/bin/env python3
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
