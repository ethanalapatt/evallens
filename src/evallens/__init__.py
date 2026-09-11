"""EvalLens: a local differential debugger for neural-network inference.

Detects output regressions between a reference and a candidate implementation, localizes the
earliest *observed* divergence at semantically aligned checkpoints, minimizes a failing input
while preserving its failure signature, and exports a standalone reproduction.
"""

from evallens.compare import compare, compare_arrays
from evallens.replay import (
    ReplayBudget,
    RunCounters,
    StabilityResult,
    run_comparison,
    stable_comparison,
)
from evallens.types import (
    Adapter,
    Case,
    CaseSize,
    ComparisonResult,
    ExecutionMode,
    ExecutionResult,
    FailureSignature,
    Request,
    TolerancePolicy,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "Case",
    "CaseSize",
    "ComparisonResult",
    "ExecutionMode",
    "ExecutionResult",
    "FailureSignature",
    "ReplayBudget",
    "Request",
    "RunCounters",
    "StabilityResult",
    "TolerancePolicy",
    "Verdict",
    "__version__",
    "compare",
    "compare_arrays",
    "run_comparison",
    "stable_comparison",
]
