"""Known-good comparisons.

Controls answer the question a detection rate cannot: *how often does EvalLens claim a
regression when there is none?* Every control here is a comparison that a correct
implementation must pass, so a stable FAIL on any of them is a false positive and is reported
as one.

The set deliberately includes more than "the same code twice". Exact equality is the easiest
control to pass and the least informative, so the corpus also covers the correct cached path
against full-prefix execution, padding transformations, batch permutations, fresh-request
isolation, and one benign perturbation that produces a real but sub-tolerance difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evallens.adapters.native import CandidateAdapter, ReferenceAdapter
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.types import Case, ExecutionMode, Request

BENIGN_PERTURBATION = 1e-7
"""Multiplicative perturbation of the output projection for the benign control.

Chosen to be genuinely nonzero yet far below the frozen tolerance band, so the control
exercises the *band* rather than exact bitwise equality. It is a control, never a mutant.
"""


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """One known-good comparison."""

    control_id: str
    description: str
    requests: tuple[Request, ...]
    execution_mode: ExecutionMode
    behavior: Behavior
    exercises: str

    def build_case(self, config: ModelConfig, weights: WeightDict, seed: int = 0) -> Case:
        return Case.create(
            model_config_id=config.config_id,
            weights_sha256=weights_sha256(weights),
            requests=self.requests,
            execution_mode=self.execution_mode,
            input_seed=seed,
            category="control",
            provenance=(f"control:{self.control_id}",),
        )

    def build_adapters(
        self, config: ModelConfig, weights: WeightDict
    ) -> tuple[ReferenceAdapter, CandidateAdapter]:
        return (
            ReferenceAdapter(config, weights),
            CandidateAdapter(config, weights, self.behavior, label=self.control_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "description": self.description,
            "execution_mode": self.execution_mode.value,
            "behavior": self.behavior.to_dict(),
            "exercises": self.exercises,
            "injected_fault": False,
        }


def _r(name: str, tokens: Sequence[int], prefix: int | None = None, pad_left: int = 0) -> Request:
    return Request(
        name,
        tuple(tokens),
        prefix_length=len(tokens) if prefix is None else prefix,
        pad_left=pad_left,
    )


CONTROLS: tuple[ControlSpec, ...] = (
    ControlSpec(
        control_id="identical_stateless",
        description="The same full-prefix implementation compared against itself.",
        requests=(_r("c0", [5, 6, 7, 8, 9]),),
        execution_mode=ExecutionMode.STATELESS_BATCH,
        behavior=Behavior(),
        exercises="Exact equality. The floor: any failure here is a bug in EvalLens itself.",
    ),
    ControlSpec(
        control_id="cached_vs_full_prefix",
        description="Correct incremental cached decoding against a correct full-prefix pass.",
        requests=(_r("c0", [10, 11, 12, 13, 14, 15, 16], prefix=3),),
        execution_mode=ExecutionMode.CACHED_DECODE,
        exercises=(
            "Two genuinely different algorithms on the same weights. Floating-point "
            "accumulation order differs, so this must pass on the tolerance band rather than "
            "on equality."
        ),
        behavior=Behavior(),
    ),
    ControlSpec(
        control_id="cached_prefill_only",
        description="Cached execution whose prefill covers the whole request.",
        requests=(_r("c0", [17, 18, 19, 20]),),
        execution_mode=ExecutionMode.CACHED_DECODE,
        behavior=Behavior(),
        exercises="The zero-decode-step boundary of the cached path.",
    ),
    ControlSpec(
        control_id="padded_alignment",
        description="A heavily left-padded row beside an unpadded one, both correct.",
        requests=(_r("c0", [21, 22, 23], pad_left=4), _r("c1", [24, 25, 26, 27, 28, 29, 30])),
        execution_mode=ExecutionMode.STATELESS_BATCH,
        behavior=Behavior(),
        exercises=(
            "Correctly aligned padding transformations. If the position-ID convention were "
            "wrong, this control would fail instead of the padding mutants."
        ),
    ),
    ControlSpec(
        control_id="batch_permutation",
        description="Several stateless rows of differing lengths in one batch.",
        requests=(
            _r("c0", [31, 32]),
            _r("c1", [33, 34, 35, 36, 37]),
            _r("c2", [38, 39, 40]),
        ),
        execution_mode=ExecutionMode.STATELESS_BATCH,
        behavior=Behavior(),
        exercises="Row independence in a ragged batch.",
    ),
    ControlSpec(
        control_id="fresh_request_isolation",
        description="A three-request session with correct per-request cache reset.",
        requests=(
            _r("c0", [41, 42, 43, 44], prefix=2),
            _r("c1", [45, 46], prefix=1),
            _r("c2", [47, 48, 49, 50, 51], prefix=3),
        ),
        execution_mode=ExecutionMode.SESSION,
        behavior=Behavior(),
        exercises="Session state that is intentional within a request and cleared between them.",
    ),
    ControlSpec(
        control_id="benign_subtolerance_perturbation",
        description=(
            "A correct implementation whose output projection is scaled by 1+1e-7 — a real "
            "difference, deliberately below the frozen tolerance."
        ),
        requests=(_r("c0", [52, 53, 54, 55, 56, 57], prefix=2),),
        execution_mode=ExecutionMode.CACHED_DECODE,
        behavior=Behavior(perturb_scale=BENIGN_PERTURBATION),
        exercises=(
            "The tolerance band itself. Without this, every control could pass by exact "
            "equality and the band would never be tested in the passing direction."
        ),
    ),
)

CONTROL_BY_ID: dict[str, ControlSpec] = {control.control_id: control for control in CONTROLS}

__all__ = ["BENIGN_PERTURBATION", "CONTROLS", "CONTROL_BY_ID", "ControlSpec"]
