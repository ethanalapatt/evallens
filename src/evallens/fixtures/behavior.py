"""The injected-fault switchboard.

Read this before concluding that EvalLens tests itself unfairly.

Every knob here defaults to the correct behavior. ``Behavior()`` with no arguments *is* the
reference implementation; the reference adapter constructs exactly that and nothing else.
Faults are introduced only by the corpus in ``bench/mutants/``, which builds non-default
``Behavior`` values and pairs each with an independently written trigger fixture.

Why the switchboard lives in ``src`` rather than in the mutant package: a fault such as
"the causal mask permits one future position" is a one-line change in the middle of
attention. Expressing it as a separate forked copy of the model per variant would give
sixteen near-duplicate transformers that drift apart, and a mutant that diverges from the
reference for accidental reasons is worse than no mutant at all. Keeping one transformer
and one explicit switch means a variant differs from the reference in exactly the declared
way, and ``tests/unit/test_behavior_isolation.py`` asserts that the default is untouched.

What still has to hold, and is enforced by tests:

* ``evallens.generate`` and ``evallens.reduce`` never import ``bench.mutants`` and never
  read a ``Behavior``. They reach models only through the ``Adapter`` protocol.
* Fault labels, trigger fixtures, and expected answers live in the scoring layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AttnScaleMode = Literal["correct", "no_sqrt", "d_model"]
CausalMode = Literal["correct", "off_by_one", "leak_last"]
PadMaskMode = Literal["correct", "ignore", "right_only"]
NormMode = Literal["correct", "large_eps", "wrong_axis"]
BatchMode = Literal["correct", "row_leak_first", "v_roll"]
CacheIndexMode = Literal["correct", "write_overwrite_last", "read_drop_oldest"]
DecodePosMode = Literal["correct", "minus_one", "restart"]
ResetMode = Literal["correct", "none", "partial"]


@dataclass(frozen=True, slots=True)
class Behavior:
    """Behavioral knobs of the native fixture. All defaults are correct.

    Model-level knobs are consumed inside the transformer; ``decode_pos`` and ``reset`` are
    consumed by the cached adapter, because a wrong decode position and a missing
    between-request cache reset are properties of the *serving loop*, not of the layer math.
    """

    attn_scale: AttnScaleMode = "correct"
    causal: CausalMode = "correct"
    pad_mask: PadMaskMode = "correct"
    norm: NormMode = "correct"
    batch: BatchMode = "correct"
    cache_index: CacheIndexMode = "correct"
    decode_pos: DecodePosMode = "correct"
    reset: ResetMode = "correct"
    perturb_scale: float = 0.0
    """Uniform multiplicative perturbation of the output projection, for benign controls.

    Used to build a known-good control whose difference is real but below the frozen
    tolerance, so that exact bitwise equality is not the only control ever exercised.
    """

    @property
    def is_reference(self) -> bool:
        return self == Behavior()

    def describe(self) -> str:
        if self.is_reference:
            return "reference (no injected fault)"
        changed = [f"{k}={v!r}" for k, v in asdict(self).items() if v != getattr(Behavior(), k)]
        return "injected: " + ", ".join(changed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> Behavior:
        known = set(Behavior.__dataclass_fields__)
        return Behavior(**{k: v for k, v in payload.items() if k in known})


REFERENCE_BEHAVIOR = Behavior()

__all__ = [
    "REFERENCE_BEHAVIOR",
    "AttnScaleMode",
    "BatchMode",
    "Behavior",
    "CacheIndexMode",
    "CausalMode",
    "DecodePosMode",
    "NormMode",
    "PadMaskMode",
    "ResetMode",
]
