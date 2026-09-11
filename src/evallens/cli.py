"""EvalLens command-line interface.

Subcommands are added as the milestones that implement them land. A command appears here
only when it does real work; there are no placeholders that print "not implemented".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from evallens import __version__


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
