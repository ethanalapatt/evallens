"""The injected-fault corpus.

**Everything in this package is a fault this project wrote on purpose.** Nothing here is a
discovered bug in PyTorch or any other library, and no report, viewer, or summary may
describe it as one.

This package belongs to the *scoring* layer. It knows fault labels, trigger fixtures, and
expected answers. `evallens.generate` and `evallens.reduce` must never import it — there is a
test that enforces that, because a search policy that can see the answer key measures nothing.

Qualification
-------------
A variant is included in a frozen benchmark manifest only once it is **qualified**: its own
independently written trigger fixture must produce a *stable* FAIL against the reference.
The triggers below are handwritten, with explicit token values chosen to expose each
mechanism — they are not produced by either generator, so a generator can never be credited
for finding a case that was built for it.

A variant that does not qualify stays visible in the manifest as incomplete. It is never
quietly deleted to improve a headline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evallens.adapters.native import CandidateAdapter, ReferenceAdapter
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.replay import ReplayBudget, RunCounters, stable_comparison
from evallens.types import Case, ExecutionMode, Request, TolerancePolicy, Verdict


@dataclass(frozen=True, slots=True)
class FaultFamily:
    number: int
    key: str
    title: str
    description: str


FAMILIES: tuple[FaultFamily, ...] = (
    FaultFamily(
        1,
        "causal_mask",
        "Causal mask boundary",
        "The attention mask admits a position that lies in the future, so a token can read "
        "information it has not been given yet.",
    ),
    FaultFamily(
        2,
        "padding_mask",
        "Padding-mask application",
        "Padding columns are treated as real keys, so a padded row attends to positions that "
        "carry no token at all.",
    ),
    FaultFamily(
        3,
        "decode_position",
        "Positional offset during incremental decoding",
        "A decode step uses the wrong position id, so the cached path embeds tokens at "
        "positions the full-prefix path never used.",
    ),
    FaultFamily(
        4,
        "cache_indexing",
        "Cached key/value indexing",
        "The cache writes to or reads from the wrong slot, so a query attends to a history "
        "that is silently truncated or overwritten.",
    ),
    FaultFamily(
        5,
        "request_reset",
        "Between-request cache reset",
        "Per-request cache state survives into the next request of a session, so a request's "
        "output depends on whatever ran before it.",
    ),
    FaultFamily(
        6,
        "normalization",
        "Normalization epsilon or axis",
        "Normalization uses the wrong epsilon or reduces over the wrong axis, mixing "
        "statistics across tokens that should be independent.",
    ),
    FaultFamily(
        7,
        "attention_scaling",
        "Attention scaling",
        "Scores are scaled by the wrong factor, sharpening or flattening every attention "
        "distribution in the model.",
    ),
    FaultFamily(
        8,
        "batch_indexing",
        "Batch row indexing",
        "One row's activations are used for another, so results depend on what else happened "
        "to be in the batch.",
    ),
)

FAMILY_BY_KEY: dict[str, FaultFamily] = {family.key: family for family in FAMILIES}


@dataclass(frozen=True, slots=True)
class TriggerFixture:
    """A handwritten case that demonstrates one fault's intended mechanism.

    ``rationale`` records *why* this particular shape exposes the fault, so a reader can
    check the reasoning rather than trusting that the case happens to fail.
    """

    requests: tuple[Request, ...]
    execution_mode: ExecutionMode
    rationale: str

    def build(self, config: ModelConfig, weights: WeightDict) -> Case:
        return Case.create(
            model_config_id=config.config_id,
            weights_sha256=weights_sha256(weights),
            requests=self.requests,
            execution_mode=self.execution_mode,
            input_seed=0,
            category="trigger",
            provenance=("handwritten trigger fixture",),
        )


@dataclass(frozen=True, slots=True)
class MutantSpec:
    """One declared, injected fault variant."""

    mutant_id: str
    family_key: str
    variant: str
    behavior: Behavior
    description: str
    trigger: TriggerFixture
    injected: bool = field(default=True, init=False)

    @property
    def family(self) -> FaultFamily:
        return FAMILY_BY_KEY[self.family_key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "family_number": self.family.number,
            "family_key": self.family_key,
            "family_title": self.family.title,
            "variant": self.variant,
            "description": self.description,
            "behavior": self.behavior.to_dict(),
            "injected_fault": True,
            "trigger_mode": self.trigger.execution_mode.value,
            "trigger_rationale": self.trigger.rationale,
        }


def _r(name: str, tokens: Sequence[int], prefix: int | None = None, pad_left: int = 0) -> Request:
    return Request(
        name,
        tuple(tokens),
        prefix_length=len(tokens) if prefix is None else prefix,
        pad_left=pad_left,
    )


# --- the corpus ------------------------------------------------------------------------------
#
# Trigger token values are handwritten. Where a pattern matters it is chosen for a reason
# recorded in the rationale, not drawn at random.

MUTANTS: tuple[MutantSpec, ...] = (
    # Family 1 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="causal_mask.off_by_one",
        family_key="causal_mask",
        variant="off_by_one",
        behavior=Behavior(causal="off_by_one"),
        description="The causal boundary is `k <= q + 1`, admitting exactly one future key.",
        trigger=TriggerFixture(
            requests=(_r("t0", [11, 12, 13, 14, 15]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Five distinct ascending tokens in one full-prefix row. Every position except "
                "the last gains one future key, so the leak shows at position 0 rather than "
                "only at the sequence end."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="causal_mask.leak_last",
        family_key="causal_mask",
        variant="leak_last",
        behavior=Behavior(causal="leak_last"),
        description="The final key column is always visible, regardless of query position.",
        trigger=TriggerFixture(
            requests=(_r("t0", [21, 22, 23, 24, 25, 26]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Six distinct tokens so the last column carries information no earlier "
                "position should see; a repeated pattern could mask the leak by coincidence."
            ),
        ),
    ),
    # Family 2 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="padding_mask.ignore",
        family_key="padding_mask",
        variant="ignore",
        behavior=Behavior(pad_mask="ignore"),
        description="Padding columns are never masked, so pad embeddings enter attention.",
        trigger=TriggerFixture(
            requests=(_r("t0", [31, 32, 33], pad_left=4),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Four left-padding columns before three real tokens. Padding must exist for a "
                "padding fault to be reachable at all; four columns make the contribution "
                "large enough to be unambiguous."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="padding_mask.right_only",
        family_key="padding_mask",
        variant="right_only",
        behavior=Behavior(pad_mask="right_only"),
        description="Only trailing padding is masked, so leading padding leaks into attention.",
        trigger=TriggerFixture(
            requests=(
                _r("t0", [41, 42], pad_left=3),
                _r("t1", [43, 44, 45, 46, 47]),
            ),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "A short left-padded row beside a long unpadded row. The second row sets the "
                "batch width so the first genuinely has leading padding to leak."
            ),
        ),
    ),
    # Family 3 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="decode_position.minus_one",
        family_key="decode_position",
        variant="minus_one",
        behavior=Behavior(decode_pos="minus_one"),
        description="Each decode step embeds its token one position too early.",
        trigger=TriggerFixture(
            requests=(_r("t0", [51, 52, 53, 54, 55], prefix=1),),
            execution_mode=ExecutionMode.CACHED_DECODE,
            rationale=(
                "Prefill of one token leaves four decode steps, the maximum available for "
                "this length, so the offset applies to as many positions as possible."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="decode_position.restart",
        family_key="decode_position",
        variant="restart",
        behavior=Behavior(decode_pos="restart"),
        description="Decode positions restart from zero instead of continuing the prefix.",
        trigger=TriggerFixture(
            requests=(_r("t0", [61, 62, 63, 64, 65, 66], prefix=3),),
            execution_mode=ExecutionMode.CACHED_DECODE,
            rationale=(
                "A prefill of three makes the restart offset three positions wide. With a "
                "prefill of one the restart would coincide with the correct position at the "
                "first decode step and the fault would be partly invisible."
            ),
        ),
    ),
    # Family 4 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="cache_indexing.write_overwrite_last",
        family_key="cache_indexing",
        variant="write_overwrite_last",
        behavior=Behavior(cache_index="write_overwrite_last"),
        description="A new key/value overwrites the most recent slot instead of extending.",
        trigger=TriggerFixture(
            requests=(_r("t0", [71, 72, 73, 74, 75], prefix=2),),
            execution_mode=ExecutionMode.CACHED_DECODE,
            rationale=(
                "Three decode steps after a prefill of two, so at least two writes land on an "
                "already-populated slot and permanently lose an earlier position."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="cache_indexing.read_drop_oldest",
        family_key="cache_indexing",
        variant="read_drop_oldest",
        behavior=Behavior(cache_index="read_drop_oldest"),
        description="Decode reads a cache view missing its oldest entry; the cache itself is intact.",
        trigger=TriggerFixture(
            requests=(_r("t0", [81, 82, 83, 84, 85, 86], prefix=2),),
            execution_mode=ExecutionMode.CACHED_DECODE,
            rationale=(
                "A prefill of two gives the dropped view something to drop from the first "
                "decode step onward. Prefill itself is untouched, which is what keeps this "
                "shape-valid and silent rather than a crash."
            ),
        ),
    ),
    # Family 5 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="request_reset.none",
        family_key="request_reset",
        variant="none",
        behavior=Behavior(reset="none"),
        description="The cache is never cleared between requests of a session.",
        trigger=TriggerFixture(
            requests=(
                _r("t0", [91, 92, 93, 94], prefix=2),
                _r("t1", [95, 96, 89], prefix=1),
            ),
            execution_mode=ExecutionMode.SESSION,
            rationale=(
                "Two requests are the minimum that can expose a between-request reset, and "
                "the first is long enough that its residue meaningfully changes the second."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="request_reset.partial",
        family_key="request_reset",
        variant="partial",
        behavior=Behavior(reset="partial"),
        description="Only layer 0's cache is cleared between requests; deeper layers persist.",
        trigger=TriggerFixture(
            requests=(
                _r("t0", [55, 56, 57, 58, 59], prefix=2),
                _r("t1", [60, 61, 62, 63], prefix=1),
            ),
            execution_mode=ExecutionMode.SESSION,
            rationale=(
                "A longer first request leaves more stale state in the layers that are not "
                "cleared. Partial resets are harder to see than none at all, because the "
                "first layer looks correct."
            ),
        ),
    ),
    # Family 6 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="normalization.large_eps",
        family_key="normalization",
        variant="large_eps",
        behavior=Behavior(norm="large_eps"),
        description="Normalization epsilon is 1e-1 instead of 1e-5.",
        trigger=TriggerFixture(
            requests=(_r("t0", [7, 8, 9, 10]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Epsilon enters every normalization in the model, so any valid case triggers "
                "it; a short row keeps the trigger cheap."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="normalization.wrong_axis",
        family_key="normalization",
        variant="wrong_axis",
        behavior=Behavior(norm="wrong_axis"),
        description="Normalization reduces over the token axis instead of the feature axis.",
        trigger=TriggerFixture(
            requests=(_r("t0", [13, 14, 15, 16, 17]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "More than one token is required for a token-axis reduction to differ from a "
                "feature-axis one; five tokens make the cross-token mixing unmistakable."
            ),
        ),
    ),
    # Family 7 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="attention_scaling.no_sqrt",
        family_key="attention_scaling",
        variant="no_sqrt",
        behavior=Behavior(attn_scale="no_sqrt"),
        description="Scores are scaled by 1/d_head instead of 1/sqrt(d_head).",
        trigger=TriggerFixture(
            requests=(_r("t0", [23, 24, 25, 26, 27, 28]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Scaling only matters where a softmax has more than one candidate key, so the "
                "row is long enough for later positions to attend over several keys."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="attention_scaling.d_model",
        family_key="attention_scaling",
        variant="d_model",
        behavior=Behavior(attn_scale="d_model"),
        description="Scores are scaled by 1/sqrt(d_model) instead of 1/sqrt(d_head).",
        trigger=TriggerFixture(
            requests=(_r("t0", [29, 30, 31, 32, 33, 34]),),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "The classic confusion between model width and head width. Like the other "
                "scaling variant it needs several keys per softmax to be visible."
            ),
        ),
    ),
    # Family 8 -------------------------------------------------------------------------------
    MutantSpec(
        mutant_id="batch_indexing.row_leak_first",
        family_key="batch_indexing",
        variant="row_leak_first",
        behavior=Behavior(batch="row_leak_first"),
        description="Layer 0's attention output for row 0 is broadcast across the batch.",
        trigger=TriggerFixture(
            requests=(
                _r("t0", [35, 36, 37]),
                _r("t1", [38, 39, 40]),
            ),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Two rows with disjoint token values and equal lengths. Equal lengths remove "
                "padding as a confound, so a difference can only come from row mixing."
            ),
        ),
    ),
    MutantSpec(
        mutant_id="batch_indexing.v_roll",
        family_key="batch_indexing",
        variant="v_roll",
        behavior=Behavior(batch="v_roll"),
        description="Attention values are rolled by one along the batch axis.",
        trigger=TriggerFixture(
            requests=(
                _r("t0", [42, 43, 44, 45]),
                _r("t1", [46, 47, 48, 49]),
                _r("t2", [50, 51, 52, 53]),
            ),
            execution_mode=ExecutionMode.STATELESS_BATCH,
            rationale=(
                "Three equal-length rows with disjoint tokens. Three rows make the roll a "
                "genuine permutation rather than a swap, so no row keeps its own values."
            ),
        ),
    ),
)

MUTANT_BY_ID: dict[str, MutantSpec] = {mutant.mutant_id: mutant for mutant in MUTANTS}


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """Whether a variant earned its place in a frozen benchmark manifest."""

    mutant_id: str
    qualified: bool
    verdict: Verdict
    max_abs_err: float
    detail: str
    stable: bool
    verdict_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutant_id": self.mutant_id,
            "qualified": self.qualified,
            "verdict": self.verdict.value,
            "stable": self.stable,
            "max_abs_err": self.max_abs_err,
            "verdict_counts": self.verdict_counts,
            "detail": self.detail,
        }


def qualify_mutant(
    spec: MutantSpec,
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    *,
    replays: int = 3,
    budget: ReplayBudget | None = None,
    counters: RunCounters | None = None,
) -> QualificationResult:
    """Run a variant against its own handwritten trigger and decide whether it qualifies.

    Qualification requires a *stable* FAIL. A crash is an `ERROR` and does not qualify: a
    mutant that only ever crashes tests the exception path, not the detector.
    """
    case = spec.trigger.build(config, weights)
    reference = ReferenceAdapter(config, weights)
    candidate = CandidateAdapter(config, weights, spec.behavior, label=spec.mutant_id)
    outcome = stable_comparison(
        reference, candidate, case, policy, replays=replays, budget=budget, counters=counters
    )
    representative = outcome.representative
    return QualificationResult(
        mutant_id=spec.mutant_id,
        qualified=outcome.stable and outcome.verdict is Verdict.FAIL,
        verdict=outcome.verdict,
        max_abs_err=representative.max_abs_err,
        detail=representative.detail,
        stable=outcome.stable,
        verdict_counts=outcome.verdict_counts,
    )


def qualify_all(
    config: ModelConfig,
    weights: WeightDict,
    policy: TolerancePolicy,
    *,
    replays: int = 3,
) -> list[QualificationResult]:
    """Qualify every declared variant. Unqualified ones stay in the list, marked."""
    return [qualify_mutant(spec, config, weights, policy, replays=replays) for spec in MUTANTS]


def build_mutant_adapter(
    spec: MutantSpec, config: ModelConfig, weights: WeightDict
) -> CandidateAdapter:
    return CandidateAdapter(config, weights, spec.behavior, label=spec.mutant_id)


__all__ = [
    "FAMILIES",
    "FAMILY_BY_KEY",
    "MUTANTS",
    "MUTANT_BY_ID",
    "FaultFamily",
    "MutantSpec",
    "QualificationResult",
    "TriggerFixture",
    "build_mutant_adapter",
    "qualify_all",
    "qualify_mutant",
]
