"""Replay one case in a fresh process.

Run as ``python -m evallens._subprocess_replay`` with a JSON request on stdin; a JSON response
comes back on stdout. This exists for two reasons that an in-process replay cannot provide:

* **A hard timeout.** In-process execution cannot interrupt a running kernel, so an
  in-process ``TIMEOUT`` means "this exceeded its budget", not "this was stopped at its
  budget". A subprocess can actually be killed.
* **Genuine state isolation.** A case that reproduces here reproduces without any residue
  from the run that discovered it — no warm caches, no module-level state, no adapter that
  was reset incorrectly.

Exit status is always 0 when the protocol completed; the *verdict* is in the payload. A
nonzero exit means the harness itself failed, which is deliberately distinguishable from a
reproduced mismatch.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

EXIT_OK = 0
EXIT_HARNESS_ERROR = 3


def run_request(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one replay request and return its response payload."""
    from evallens.adapters.native import NativeAdapterSpec
    from evallens.replay import ReplayBudget, RunCounters, stable_comparison
    from evallens.resources import set_deterministic_threads
    from evallens.types import Case, TolerancePolicy

    threads = int(request.get("threads", 4))
    set_deterministic_threads(threads)

    case = Case.from_dict(request["case"])
    policy = TolerancePolicy.from_dict(request["policy"])
    reference = NativeAdapterSpec.from_dict(request["reference"]).build()
    candidate = NativeAdapterSpec.from_dict(request["candidate"]).build()
    replays = int(request.get("replays", 3))

    counters = RunCounters()
    outcome = stable_comparison(
        reference,
        candidate,
        case,
        policy,
        replays=replays,
        budget=ReplayBudget(timeout_s=float(request.get("timeout_s", 60.0))),
        counters=counters,
    )
    return {
        "ok": True,
        "case_id": case.case_id,
        "reference_adapter": reference.adapter_id,
        "candidate_adapter": candidate.adapter_id,
        "stability": outcome.to_dict(),
        "counters": counters.to_dict(),
        "threads": threads,
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"bad request: {exc}"}))
        return EXIT_HARNESS_ERROR

    try:
        response = run_request(request)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        )
        return EXIT_HARNESS_ERROR

    print(json.dumps(response))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())
