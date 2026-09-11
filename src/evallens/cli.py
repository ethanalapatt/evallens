"""EvalLens command-line interface.

Subcommands appear here only when they do real work. There are no placeholders that print
"not implemented".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evallens import __version__

if TYPE_CHECKING:
    from evallens.adapters.native import NativeAdapterSpec

INJECTED_FAULT_BANNER = (
    "NOTE: this example uses a fault EvalLens injected on purpose. It is not a bug "
    "discovered in PyTorch or any other third-party library."
)


def _add_doctor(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="Check the local environment and run a fast numerical self-test.",
        description=(
            "Reports the actual interpreter, torch/numpy versions, thread settings, memory, "
            "and device availability, then rebuilds the unit fixture and checks it against "
            "the independent FP64 NumPy oracle. Exits nonzero if the self-test fails."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the manifest as JSON.")
    parser.add_argument(
        "--threads", type=int, default=4, help="Thread count to pin before testing (default: 4)."
    )
    parser.set_defaults(func=_run_doctor)


def _run_doctor(args: argparse.Namespace) -> int:
    import numpy as np

    from evallens.env import capture_environment
    from evallens.fixtures.config import UNIT_FIXTURE, make_weights, parameter_count, weights_sha256
    from evallens.fixtures.numpy_oracle import oracle_forward
    from evallens.fixtures.tiny_transformer import build_model
    from evallens.resources import current_rss_bytes, set_deterministic_threads

    thread_report = set_deterministic_threads(args.threads)
    manifest = capture_environment()

    import torch

    config = UNIT_FIXTURE
    weights = make_weights(config)
    digest = weights_sha256(weights)
    model = build_model(config, weights)

    rng = np.random.default_rng(7)
    tokens = rng.integers(1, config.vocab_size, size=(2, 6), dtype=np.int64)
    positions = np.tile(np.arange(6, dtype=np.int64), (2, 1))
    valid = np.ones((2, 6), dtype=bool)

    torch_logits = (
        model(
            torch.from_numpy(tokens),
            torch.from_numpy(positions),
            torch.from_numpy(valid),
        )
        .detach()
        .numpy()
        .astype(np.float64)
    )
    oracle_logits = oracle_forward(config, weights, tokens, positions, valid)
    max_abs_err = float(np.abs(torch_logits - oracle_logits).max())
    tolerance = 2e-4
    self_test_passed = bool(np.isfinite(max_abs_err) and max_abs_err <= tolerance)

    process_rss = current_rss_bytes()
    payload: dict[str, Any] = {
        "evallens_version": __version__,
        "environment": manifest.to_dict(),
        "threads": thread_report,
        "rss_bytes": process_rss,
        "fixture": {
            "config_id": config.config_id,
            "weights_sha256": digest,
            "parameter_count": parameter_count(config),
        },
        "self_test": {
            "name": "unit fixture vs independent FP64 NumPy oracle",
            "shape": list(torch_logits.shape),
            "max_abs_err": max_abs_err,
            "tolerance": tolerance,
            "passed": self_test_passed,
        },
        "missing_environment_fields": manifest.missing_fields(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        env = manifest
        print(f"EvalLens {__version__}")
        print(f"  python            {env.python_version}  ({env.python_executable})")
        print(f"  torch / numpy     {env.torch_version} / {env.numpy_version}")
        print(f"  platform          {env.platform}")
        print(f"  chip / cpus       {env.chip} / {env.cpu_count}")
        memory = env.physical_memory_bytes
        print(
            f"  physical memory   {'unknown' if memory is None else f'{memory / 1024**3:.1f} GiB'}"
        )
        print(
            f"  torch threads     {env.torch_num_threads} (interop {env.torch_num_interop_threads})"
        )
        print(f"  default dtype     {env.default_dtype}")
        print(f"  mps / cuda        available={env.mps_available} / available={env.cuda_available}")
        print(f"  thermal           {env.thermal_pressure or 'unavailable'}")
        print(f"  git               {env.git_commit or 'none'} dirty={env.git_dirty}")
        rss = process_rss
        print(f"  process rss       {'unknown' if rss is None else f'{rss / 1024**2:.1f} MiB'}")
        print()
        print(f"  fixture           {config.config_id}")
        print(f"  parameters        {parameter_count(config):,}")
        print(f"  weights sha256    {digest[:32]}...")
        print()
        status = "PASS" if self_test_passed else "FAIL"
        print(f"  self-test         [{status}] fixture vs FP64 NumPy oracle")
        print(f"                    max |Δ| = {max_abs_err:.3e}  (tolerance {tolerance:.1e})")
        if env.cuda_available:
            print("\n  note: CUDA is visible but EvalLens targets CPU as its reference platform.")
        if not self_test_passed:
            print("\n  The fixture does not match its independent oracle on this machine.")
            print("  Do not trust comparison results until this is resolved.")

    return 0 if self_test_passed else 1


# --- demo ------------------------------------------------------------------------------------


def _add_demo(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "demo",
        help="Run the full pipeline on a deliberately injected fault and write real artifacts.",
        description=(
            "One command: generate valid cases, detect a stable regression, localize the "
            "earliest observed divergence, reduce the failing input, export a standalone "
            "reproduction, and verify that reproduction from a clean temporary directory. "
            "The fault is injected by EvalLens on purpose and is labeled as such everywhere."
        ),
    )
    parser.add_argument("--out", default="artifacts/demo", help="Output directory.")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu"], help="CPU is the reference platform."
    )
    parser.add_argument("--config", default=None, help="Path to a TOML configuration file.")
    parser.add_argument("--seed", type=int, default=None, help="Generator seed.")
    parser.add_argument("--max-cases", type=int, default=64, help="Case budget for the search.")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip clean-room verification of the export."
    )
    parser.add_argument("--json", action="store_true", help="Emit the run record as JSON.")
    parser.set_defaults(func=_run_demo)


def _run_demo(args: argparse.Namespace) -> int:
    from evallens.demo import DEMO_SEED, run_demo
    from evallens.settings import Settings

    settings = Settings.load(args.config) if args.config else Settings()
    result = run_demo(
        args.out,
        settings=settings,
        seed=args.seed if args.seed is not None else DEMO_SEED,
        max_cases=args.max_cases,
        verify_export=not args.no_verify,
    )

    if args.json:
        print(json.dumps(result.record, indent=2, sort_keys=True))
        return 0 if result.succeeded else 1

    print(f"EvalLens demo — {result.run_id}")
    print(INJECTED_FAULT_BANNER)
    print()
    for step in result.steps:
        print(f"  [{step.elapsed_s:6.2f}s] {step.name:22s} {step.detail}")

    if not result.succeeded:
        print(f"\nDemo did not complete: {result.failure_reason}")
        return 1

    record = result.record
    reduced = record["reduced_case"]
    tokens = [t for request in reduced["requests"] for t in request["token_ids"]]
    print()
    print(f"  original case     {record['original_case']['case_id']}")
    print(f"  reduced case      {reduced['case_id']}  tokens={tokens}")
    print(f"  earliest observed {record['localization']['earliest_observed_str']}")
    print(f"  max |Δ|           {record['comparison']['max_abs_err']:.6e}")
    print()
    print(f"  artifacts         {result.out_dir}")
    print(f"  reproduction      {result.repro_dir}")
    print(f"  run it            python {result.repro_dir / 'repro.py'} --expect-mismatch")
    print(f"  view it           evallens view {result.out_dir}")
    return 0


# --- shared loading -----------------------------------------------------------------------------


def _load_case_bundle(path: str) -> dict[str, Any]:
    """Accept either a bare case file or a failure bundle that wraps one."""
    payload = json.loads(Path(path).read_text())
    return payload if "case" in payload else {"case": payload}


def _adapters_from_bundle(
    bundle: dict[str, Any], settings: Any
) -> tuple[NativeAdapterSpec, NativeAdapterSpec]:
    """Resolve adapter specs from a bundle, defaulting to a correct reference/candidate pair."""
    from evallens.adapters.native import NativeAdapterSpec

    config = settings.model_config()
    reference_payload = bundle.get("reference") or {"role": "reference", "config": config.to_dict()}
    candidate_payload = bundle.get("candidate") or {"role": "candidate", "config": config.to_dict()}
    return (
        NativeAdapterSpec.from_dict(reference_payload),
        NativeAdapterSpec.from_dict(candidate_payload),
    )


# --- compare ---------------------------------------------------------------------------------


def _add_compare(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Compare a reference and candidate adapter on one case.",
        description=(
            "Runs both adapters from clean state, repeats to establish stability, and prints "
            "the verdict with its numerical evidence. Exits 0 on PASS, 1 on a stable FAIL, "
            "and 2 for anything else (INVALID, ERROR, TIMEOUT, RESOURCE_LIMIT, UNSTABLE)."
        ),
    )
    parser.add_argument("--case", required=True, help="Path to a case or failure JSON file.")
    parser.add_argument("--config", default=None, help="Path to a TOML configuration file.")
    parser.add_argument("--localize", action="store_true", help="Also run a traced pass.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.set_defaults(func=_run_compare)


def _run_compare(args: argparse.Namespace) -> int:
    from evallens.replay import stable_comparison
    from evallens.settings import Settings
    from evallens.trace import trace_and_localize
    from evallens.types import Case, TolerancePolicy, Verdict

    settings = Settings.load(args.config) if args.config else Settings()
    bundle = _load_case_bundle(args.case)
    case = Case.from_dict(bundle["case"])
    policy = (
        TolerancePolicy.from_dict(bundle["policy"]) if "policy" in bundle else settings.tolerance
    )
    reference_spec, candidate_spec = _adapters_from_bundle(bundle, settings)
    reference, candidate = reference_spec.build(), candidate_spec.build()

    outcome = stable_comparison(
        reference,
        candidate,
        case,
        policy,
        replays=settings.stability_replays,
        budget=settings.replay_budget(),
    )
    representative = outcome.representative

    localization = None
    if args.localize and outcome.verdict is Verdict.FAIL:
        localization = trace_and_localize(reference, candidate, case, policy)

    if args.json:
        payload = {
            "case_id": case.case_id,
            "stability": outcome.to_dict(),
            "localization": localization.to_dict(max_comparisons=64) if localization else None,
            "injected_fault": bundle.get("injected_fault"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"case            {case.case_id}  ({case.execution_mode.value})")
        print(f"reference       {reference.adapter_id}")
        print(f"candidate       {candidate.adapter_id}")
        print(f"policy          atol={policy.atol:g} rtol={policy.rtol:g} [{policy.policy_id}]")
        print(f"replays         {outcome.replays}, stable={outcome.stable}")
        print(f"verdict         {outcome.verdict.value.upper()}")
        print(f"detail          {representative.detail}")
        if representative.diffs:
            print(f"max |Δ|         {representative.max_abs_err:.6e}")
        if localization:
            print(f"localization    {localization.summary()}")
        if bundle.get("injected_fault"):
            print()
            print(INJECTED_FAULT_BANNER)

    if outcome.verdict is Verdict.PASS:
        return 0
    if outcome.verdict is Verdict.FAIL and outcome.stable:
        return 1
    return 2


# --- reduce ------------------------------------------------------------------------------------


def _add_reduce(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reduce",
        help="Minimize a failing case while preserving its failure signature.",
        description=(
            "Reads a failure JSON (as written by `evallens demo`), reduces the input, and "
            "writes the reduced case. Exits 1 if the input does not actually fail."
        ),
    )
    parser.add_argument("--failure", required=True, help="Path to a failure JSON file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--config", default=None, help="Path to a TOML configuration file.")
    parser.add_argument(
        "--strategy", default="ddmin", choices=["ddmin", "greedy"], help="Reduction strategy."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.set_defaults(func=_run_reduce)


def _run_reduce(args: argparse.Namespace) -> int:
    from evallens.reduce import (
        FailurePredicate,
        PredicateCounters,
        reduce_case,
        signature_from_failure,
    )
    from evallens.replay import stable_comparison
    from evallens.settings import Settings
    from evallens.types import Case, TolerancePolicy, Verdict

    settings = Settings.load(args.config) if args.config else Settings()
    bundle = _load_case_bundle(args.failure)
    case = Case.from_dict(bundle["case"])
    policy = (
        TolerancePolicy.from_dict(bundle["policy"]) if "policy" in bundle else settings.tolerance
    )
    reference_spec, candidate_spec = _adapters_from_bundle(bundle, settings)
    reference, candidate = reference_spec.build(), candidate_spec.build()

    outcome = stable_comparison(
        reference, candidate, case, policy, replays=settings.stability_replays
    )
    if outcome.verdict is not Verdict.FAIL or not outcome.stable:
        print(
            f"refusing to reduce: the input case is {outcome.verdict.value.upper()}, not a "
            f"stable failure ({outcome.representative.detail})",
            file=sys.stderr,
        )
        return 1

    signature = signature_from_failure(case, outcome.representative.failing_request_ids)
    predicate = FailurePredicate(
        reference,
        candidate,
        policy,
        signature,
        budget=settings.reduction,
        counters=PredicateCounters(),
    )
    result = reduce_case(case, predicate, strategy=args.strategy, budget=settings.reduction)

    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reduced_bundle = {
        "case": result.reduced.to_dict(),
        "signature": signature.to_dict(),
        "reference": reference_spec.to_dict(),
        "candidate": candidate_spec.to_dict(),
        "policy": policy.to_dict(),
        "injected_fault": bundle.get("injected_fault", False),
        "fault_description": bundle.get("fault_description", ""),
    }
    (destination / "failure.json").write_text(json.dumps(reduced_bundle, indent=2, sort_keys=True))
    (destination / "reduction.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True)
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"strategy        {result.strategy}")
        print(
            f"size            {result.original_size.as_tuple()} -> {result.reduced_size.as_tuple()}"
        )
        print(f"tokens          {case.total_valid_tokens} -> {result.reduced.total_valid_tokens}")
        print(f"ratio           {result.token_reduction_ratio:.1f}x")
        print(
            f"queries         {result.counters.logical_queries} logical, "
            f"{result.counters.executed_queries} executed, {result.counters.cache_hits} cached"
        )
        print(f"model runs      {result.counters.model_runs}")
        print(f"minimality      {result.minimality.value}")
        print(f"                {result.minimality_note}")
        print(f"wrote           {destination / 'failure.json'}")
    return 0


# --- export ---------------------------------------------------------------------------------------


def _add_export(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export",
        help="Write a standalone reproduction package for a failing case.",
        description=(
            "Produces a directory that runs on a machine without EvalLens installed. "
            "Verifies it from a fresh temporary directory unless --no-verify is given."
        ),
    )
    parser.add_argument("--failure", required=True, help="Path to a failure JSON file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--config", default=None, help="Path to a TOML configuration file.")
    parser.add_argument("--no-verify", action="store_true", help="Skip clean-room verification.")
    parser.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace) -> int:
    from evallens.compare import compare
    from evallens.export import ReproductionSpec, export_reproduction, verify_reproduction
    from evallens.reduce import signature_from_failure
    from evallens.replay import stable_comparison
    from evallens.settings import Settings
    from evallens.trace import trace_and_localize
    from evallens.types import Case, FailureSignature, TolerancePolicy, Verdict

    settings = Settings.load(args.config) if args.config else Settings()
    bundle = _load_case_bundle(args.failure)
    case = Case.from_dict(bundle["case"])
    policy = (
        TolerancePolicy.from_dict(bundle["policy"]) if "policy" in bundle else settings.tolerance
    )
    reference_spec, candidate_spec = _adapters_from_bundle(bundle, settings)
    reference, candidate = reference_spec.build(), candidate_spec.build()

    outcome = stable_comparison(
        reference, candidate, case, policy, replays=settings.stability_replays
    )
    if outcome.verdict is not Verdict.FAIL or not outcome.stable:
        print(
            f"refusing to export: the case is {outcome.verdict.value.upper()}, not a stable "
            "failure. An export that does not reproduce is worse than no export.",
            file=sys.stderr,
        )
        return 1

    signature = (
        FailureSignature.from_dict(bundle["signature"])
        if "signature" in bundle
        else signature_from_failure(case, outcome.representative.failing_request_ids)
    )
    comparison = compare(reference.run(case), candidate.run(case), policy)
    localization = trace_and_localize(reference, candidate, case, policy)

    spec = ReproductionSpec(
        case=case,
        reference=reference_spec,
        candidate=candidate_spec,
        policy=policy,
        signature=signature,
        comparison=comparison,
        injected_fault=bool(bundle.get("injected_fault", False)),
        fault_description=str(bundle.get("fault_description", "see manifest.json")),
        localization_summary=localization.summary(),
    )
    destination = export_reproduction(spec, args.out)
    print(f"wrote           {destination}")

    if args.no_verify:
        print("verification    skipped by request")
        return 0

    verification = verify_reproduction(destination, expect_mismatch=True)
    print(
        f"verification    exit {verification.exit_code} "
        f"(expected {verification.expected_exit_code})"
    )
    print(f"                ran from {verification.temp_dir}")
    print(f"                imported evallens: {verification.imported_evallens}")
    if not verification.ok:
        print(f"                FAILED: {verification.stderr[-300:]}", file=sys.stderr)
        return 1
    print(f"                reproduced the recorded mismatch in {verification.elapsed_s:.2f}s")
    return 0


# --- view ------------------------------------------------------------------------------------------


def _add_view(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "view",
        help="Serve the offline trace viewer on loopback.",
        description=(
            "Starts a local, loopback-only static file server for the viewer and a run "
            "directory. No backend service, and nothing is exposed off this machine."
        ),
    )
    parser.add_argument("run_dir", help="A run directory containing record.json.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (loopback only).")
    parser.add_argument("--port", type=int, default=8777, help="Port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    parser.set_defaults(func=_run_view)


def _run_view(args: argparse.Namespace) -> int:
    from evallens.viewer import serve_viewer

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"refusing to bind {args.host!r}: the viewer is loopback-only by design.",
            file=sys.stderr,
        )
        return 2
    return serve_viewer(
        args.run_dir, host=args.host, port=args.port, open_browser=not args.no_browser
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evallens",
        description=(
            "A local differential debugger for neural-network inference. Compares a reference "
            "and a candidate implementation on identical weights and inputs, localizes the "
            "earliest observed divergence, minimizes failing inputs, and exports standalone "
            "reproductions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"evallens {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_doctor(subparsers)
    _add_demo(subparsers)
    _add_compare(subparsers)
    _add_reduce(subparsers)
    _add_export(subparsers)
    _add_view(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
