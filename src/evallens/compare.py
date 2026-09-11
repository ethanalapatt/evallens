"""Numerical policy and verdict classification.

The comparator answers exactly one question — *is this candidate's output a stable
discrepancy from the reference's under the declared policy?* — and refuses to answer any
adjacent question implicitly. In particular it never turns a crash, an invalid input, or an
intermediate-activation difference into an output-regression detection.

Policy
------
``violation = abs(candidate - reference) > atol + rtol * abs(reference)``

Nonfinite handling is explicit rather than emergent. A position counts as a violation when:

* both values are finite and exceed the tolerance band, **or**
* exactly one of the two is nonfinite, **or**
* both are nonfinite but of different kinds (NaN vs +inf vs -inf).

Two NaNs at the same position are *agreement*, not a violation: the reference genuinely
produced a NaN there and the candidate matched it.

Relative L2 error uses a declared zero-norm rule. When the reference tensor's L2 norm falls
below ``zero_norm_eps`` the result is reported as the absolute L2 difference with
``rel_l2_denominator_degenerate`` set, rather than dividing by a near-zero norm to
manufacture an impressive-looking relative error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evallens.types import (
    ComparisonResult,
    ExecutionResult,
    TensorDiff,
    TolerancePolicy,
    Verdict,
)


def compare_arrays(
    key: str, reference: np.ndarray, candidate: np.ndarray, policy: TolerancePolicy
) -> TensorDiff:
    """Element-wise comparison of one aligned tensor pair.

    Shapes must already agree; a shape disagreement is an output-contract violation and is
    handled by :func:`compare`, which has the request context needed to report it usefully.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(
            f"compare_arrays requires matching shapes, got {ref.shape} vs {cand.shape}"
        )

    ref_finite = np.isfinite(ref)
    cand_finite = np.isfinite(cand)
    both_finite = ref_finite & cand_finite

    abs_err = np.zeros(ref.shape, dtype=np.float64)
    np.subtract(cand, ref, out=abs_err, where=both_finite)
    np.abs(abs_err, out=abs_err)

    band = policy.atol + policy.rtol * np.abs(np.where(ref_finite, ref, 0.0))
    violations = both_finite & (abs_err > band)

    # Disagreement about finiteness, and disagreement about which nonfinite value it is.
    finiteness_disagrees = ref_finite != cand_finite
    neither_finite = ~ref_finite & ~cand_finite
    nonfinite_kind_differs = neither_finite & ~_same_nonfinite_kind(ref, cand)
    violations = violations | finiteness_disagrees | nonfinite_kind_differs

    diff_sq = float((abs_err[both_finite] ** 2).sum())
    ref_norm = float(np.sqrt((ref[both_finite] ** 2).sum()))
    degenerate = ref_norm < policy.zero_norm_eps
    rel_l2 = np.sqrt(diff_sq) if degenerate else np.sqrt(diff_sq) / ref_norm

    n_elements = int(ref.size)
    return TensorDiff(
        key=key,
        shape=tuple(int(s) for s in ref.shape),
        dtype=str(np.asarray(reference).dtype),
        max_abs_err=float(abs_err[both_finite].max()) if both_finite.any() else 0.0,
        rel_l2_err=float(rel_l2),
        rel_l2_denominator_degenerate=degenerate,
        violating_fraction=float(int(violations.sum()) / n_elements) if n_elements else 0.0,
        n_violations=int(violations.sum()),
        n_elements=n_elements,
        reference_nonfinite=int((~ref_finite).sum()),
        candidate_nonfinite=int((~cand_finite).sum()),
    )


def _same_nonfinite_kind(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """True where two nonfinite values are the same kind (NaN/NaN, +inf/+inf, -inf/-inf)."""
    both_nan = np.isnan(ref) & np.isnan(cand)
    both_pos_inf = (ref == np.inf) & (cand == np.inf)
    both_neg_inf = (ref == -np.inf) & (cand == -np.inf)
    same: np.ndarray = both_nan | both_pos_inf | both_neg_inf
    return same


def compare(
    reference: ExecutionResult,
    candidate: ExecutionResult,
    policy: TolerancePolicy,
) -> ComparisonResult:
    """Classify one reference/candidate pair under ``policy``.

    Output-contract violations (a different set of request ids, or a different logits shape
    for the same request) are ``FAIL``: the candidate did not produce the output it was
    required to produce, and that is a regression regardless of the numbers involved.
    """
    if reference.case_id != candidate.case_id:
        raise ValueError(
            f"cannot compare results for different cases: {reference.case_id} vs "
            f"{candidate.case_id}"
        )

    def build(
        verdict: Verdict,
        diffs: Sequence[TensorDiff],
        failing: Sequence[str],
        detail: str,
    ) -> ComparisonResult:
        return ComparisonResult(
            verdict=verdict,
            case_id=reference.case_id,
            policy=policy,
            reference_adapter=reference.adapter_id,
            candidate_adapter=candidate.adapter_id,
            diffs=tuple(diffs),
            failing_request_ids=tuple(failing),
            detail=detail,
            reference_wall_time_ns=reference.wall_time_ns,
            candidate_wall_time_ns=candidate.wall_time_ns,
        )

    reference_ids = set(reference.outputs)
    candidate_ids = set(candidate.outputs)
    if reference_ids != candidate_ids:
        missing = sorted(reference_ids - candidate_ids)
        extra = sorted(candidate_ids - reference_ids)
        return build(
            Verdict.FAIL,
            (),
            missing or extra,
            f"output contract violated: candidate missing {missing}, unexpected {extra}",
        )

    diffs: list[TensorDiff] = []
    failing: list[str] = []
    contract_notes: list[str] = []

    for request_id in sorted(reference_ids):
        ref_array = np.asarray(reference.outputs[request_id])
        cand_array = np.asarray(candidate.outputs[request_id])
        if ref_array.shape != cand_array.shape:
            failing.append(request_id)
            contract_notes.append(
                f"{request_id}: shape {cand_array.shape} != reference {ref_array.shape}"
            )
            continue
        diff = compare_arrays(request_id, ref_array, cand_array, policy)
        diffs.append(diff)
        if diff.violated:
            failing.append(request_id)

    if contract_notes:
        return build(
            Verdict.FAIL,
            diffs,
            failing,
            "output contract violated: " + "; ".join(contract_notes),
        )

    if not failing:
        return build(Verdict.PASS, diffs, (), "within tolerance policy")

    worst = max((d for d in diffs if d.violated), key=lambda d: d.max_abs_err, default=None)
    detail = f"{len(failing)} request(s) violate the policy"
    if worst is not None:
        detail += (
            f"; worst {worst.key}: max|Δ|={worst.max_abs_err:.3e}, "
            f"relL2={worst.rel_l2_err:.3e}, "
            f"{worst.n_violations}/{worst.n_elements} entries"
        )
    nonfinite = [d for d in diffs if d.candidate_nonfinite > d.reference_nonfinite]
    if nonfinite:
        detail += (
            f"; nonfinite candidate output on valid positions in "
            f"{', '.join(d.key for d in nonfinite)}"
        )
    return build(Verdict.FAIL, diffs, failing, detail)


def within_policy(reference: np.ndarray, candidate: np.ndarray, policy: TolerancePolicy) -> bool:
    """Convenience predicate for a single tensor pair."""
    return not compare_arrays("tensor", reference, candidate, policy).violated


__all__ = ["compare", "compare_arrays", "within_policy"]
