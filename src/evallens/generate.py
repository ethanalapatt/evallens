"""Valid-case generators.

Two generators over the *same* declared valid-case space, given the *same* capability
information and the *same* per-trial budget. The only difference between them is how they
choose within that space.

``UniformValidGenerator``
    The named baseline. Uniform random lengths, token ids, padding, batch sizes, and session
    structures.

``BoundaryAwareGenerator``
    Emphasizes the places bugs actually live: length-1 and near-maximum sequences, prefill
    boundaries one token from each end, repeated token patterns, unequal padded lengths, and
    request transitions.

Both emit every declared category. That is deliberate and it is what makes the comparison
fair — a generator that only ever produced single-request unpadded cases could not reach the
padding, batch-indexing, or request-reset fault families at all, and its lower detection rate
would say nothing about the quality of its *choices*.

Blindness
---------
Nothing in this module may import ``bench``, read a ``Behavior``, or otherwise learn which
candidate it is generating for. It knows only the shape of the valid space. Producing invalid
random tensors and calling the resulting crashes "improved bug detection" is exactly the
failure mode this separation exists to prevent, so validity is a property test, not a hope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import numpy as np

from evallens.adapters.encoding import MAX_BATCH_ROWS, MAX_SESSION_REQUESTS
from evallens.fixtures.config import ModelConfig
from evallens.types import Case, ExecutionMode, Request

MAX_PAD_LEFT = 4
DEFAULT_MAX_TOKENS = 128


class CaseCategory(StrEnum):
    """The declared structural classes of the valid-case space.

    Every generator must be able to emit each of these, and a reduction may not move a case
    from one to another — shrinking a two-row batch to one row would silently change what is
    being tested.
    """

    STATELESS_SINGLE = "stateless_single"
    STATELESS_BATCH = "stateless_batch"
    STATELESS_PADDED = "stateless_padded"
    CACHED_PREFILL_ONLY = "cached_prefill_only"
    CACHED_DECODE = "cached_decode"
    SESSION = "session"


ALL_CATEGORIES: tuple[CaseCategory, ...] = tuple(CaseCategory)


def classify_case(case: Case) -> CaseCategory:
    """Derive a case's category from its structure alone.

    Priority order matters and is fixed: padding dominates row count, because a padded batch
    exercises the padding machinery whether or not it also has multiple rows.
    """
    if case.execution_mode is ExecutionMode.SESSION:
        return CaseCategory.SESSION
    if case.execution_mode is ExecutionMode.CACHED_DECODE:
        request = case.requests[0]
        if request.prefix_length >= request.n_valid:
            return CaseCategory.CACHED_PREFILL_ONLY
        return CaseCategory.CACHED_DECODE
    if any(request.pad_left > 0 for request in case.requests):
        return CaseCategory.STATELESS_PADDED
    if len(case.requests) > 1:
        return CaseCategory.STATELESS_BATCH
    return CaseCategory.STATELESS_SINGLE


@dataclass(frozen=True, slots=True)
class GeneratorCapability:
    """Everything a generator is told about the target, and nothing more.

    Conspicuously absent: which candidate is under test, what faults exist, and what the
    expected answer is.
    """

    model_config_id: str
    weights_sha256: str
    vocab_size: int
    max_tokens_per_request: int = DEFAULT_MAX_TOKENS
    max_batch_rows: int = MAX_BATCH_ROWS
    max_session_requests: int = MAX_SESSION_REQUESTS
    max_pad_left: int = MAX_PAD_LEFT
    categories: tuple[CaseCategory, ...] = ALL_CATEGORIES

    @staticmethod
    def for_fixture(
        config: ModelConfig, weights_sha256: str, *, max_tokens: int | None = None
    ) -> GeneratorCapability:
        limit = min(config.max_position, max_tokens or DEFAULT_MAX_TOKENS)
        return GeneratorCapability(
            model_config_id=config.config_id,
            weights_sha256=weights_sha256,
            vocab_size=config.vocab_size,
            max_tokens_per_request=limit,
        )

    def padded_token_limit(self) -> int:
        """Longest request that still fits once maximum left padding is applied."""
        return max(1, self.max_tokens_per_request - self.max_pad_left)


class CaseGenerator(ABC):
    """Deterministic generator of valid cases."""

    name: ClassVar[str] = "abstract"
    version: ClassVar[str] = "0"

    def __init__(self, capability: GeneratorCapability) -> None:
        self.capability = capability

    def generate(self, count: int, seed: int) -> list[Case]:
        """Produce ``count`` valid cases, cycling through the declared categories.

        Deterministic in ``(count, seed)``: two calls with the same arguments return
        identical cases, which is what makes a recorded trial replayable.
        """
        if count < 1:
            raise ValueError("count must be at least 1")
        rng = np.random.default_rng(seed)
        categories = self.capability.categories
        cases: list[Case] = []
        for index in range(count):
            category = categories[index % len(categories)]
            cases.append(self._build(category, rng, seed, index))
        return cases

    def _build(
        self, category: CaseCategory, rng: np.random.Generator, seed: int, index: int
    ) -> Case:
        requests, mode = self._requests_for(category, rng)
        case = Case.create(
            model_config_id=self.capability.model_config_id,
            weights_sha256=self.capability.weights_sha256,
            requests=requests,
            execution_mode=mode,
            input_seed=seed,
            category=category.value,
            provenance=(f"{self.name}@{self.version}", f"seed={seed}", f"index={index}"),
        )
        derived = classify_case(case)
        if derived is not category:
            raise AssertionError(
                f"{self.name} built a {derived.value} case while targeting {category.value}"
            )
        return case

    @abstractmethod
    def _requests_for(
        self, category: CaseCategory, rng: np.random.Generator
    ) -> tuple[list[Request], ExecutionMode]: ...

    def _tokens(self, rng: np.random.Generator, length: int) -> tuple[int, ...]:
        """Valid tokens only: drawn from ``1 .. vocab-1``, never the reserved padding id."""
        return tuple(int(t) for t in rng.integers(1, self.capability.vocab_size, size=length))


class UniformValidGenerator(CaseGenerator):
    """The named baseline: uniform random choices across the declared valid space."""

    name: ClassVar[str] = "uniform-valid"
    version: ClassVar[str] = "1"

    def _length(self, rng: np.random.Generator, limit: int | None = None) -> int:
        return int(rng.integers(1, (limit or self.capability.max_tokens_per_request) + 1))

    def _requests_for(
        self, category: CaseCategory, rng: np.random.Generator
    ) -> tuple[list[Request], ExecutionMode]:
        if category is CaseCategory.STATELESS_SINGLE:
            length = self._length(rng)
            return [Request("r0", self._tokens(rng, length), length)], ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.STATELESS_BATCH:
            rows = int(rng.integers(2, self.capability.max_batch_rows + 1))
            requests = []
            for row in range(rows):
                length = self._length(rng)
                requests.append(Request(f"r{row}", self._tokens(rng, length), length))
            return requests, ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.STATELESS_PADDED:
            rows = int(rng.integers(1, self.capability.max_batch_rows + 1))
            limit = self.capability.padded_token_limit()
            requests = []
            for row in range(rows):
                length = self._length(rng, limit)
                pad = int(rng.integers(0, self.capability.max_pad_left + 1))
                requests.append(Request(f"r{row}", self._tokens(rng, length), length, pad))
            if all(request.pad_left == 0 for request in requests):
                first = requests[0]
                requests[0] = Request(first.request_id, first.token_ids, first.prefix_length, 1)
            return requests, ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.CACHED_PREFILL_ONLY:
            length = self._length(rng)
            return [Request("r0", self._tokens(rng, length), length)], ExecutionMode.CACHED_DECODE

        if category is CaseCategory.CACHED_DECODE:
            length = max(2, self._length(rng))
            prefix = int(rng.integers(1, length))
            return [Request("r0", self._tokens(rng, length), prefix)], ExecutionMode.CACHED_DECODE

        count = int(rng.integers(2, self.capability.max_session_requests + 1))
        requests = []
        for index in range(count):
            length = self._length(rng)
            prefix = int(rng.integers(1, length + 1))
            requests.append(Request(f"r{index}", self._tokens(rng, length), prefix))
        return requests, ExecutionMode.SESSION


class BoundaryAwareGenerator(CaseGenerator):
    """Emphasizes edges: sequence ends, prefill boundaries, repeats, and ragged padding."""

    name: ClassVar[str] = "boundary-aware"
    version: ClassVar[str] = "1"

    def _boundary_length(self, rng: np.random.Generator, limit: int | None = None) -> int:
        ceiling = limit or self.capability.max_tokens_per_request
        candidates = [1, 2, 3, 4, ceiling // 2, ceiling - 1, ceiling]
        valid = sorted({value for value in candidates if 1 <= value <= ceiling})
        return int(valid[int(rng.integers(0, len(valid)))])

    def _boundary_prefix(self, rng: np.random.Generator, length: int) -> int:
        """Prefill boundaries at both ends and adjacent to them, not uniformly in between.

        Every candidate is clamped into ``[1, length]``: for a length-1 request the
        "boundary" values collapse onto 1, and an unclamped 2 would be an invalid prefix.
        """
        candidates = sorted({min(max(1, value), length) for value in (1, 2, length - 1, length)})
        return int(candidates[int(rng.integers(0, len(candidates)))])

    def _patterned_tokens(self, rng: np.random.Generator, length: int) -> tuple[int, ...]:
        """Repeated and structured token patterns, which stress position-dependent logic.

        A uniformly random sequence almost never repeats a token. A cache that reuses the
        wrong slot, or a position offset that is off by one, can be invisible under random
        tokens and obvious under `A B A B` — the wrong slot happens to hold a plausible value.
        """
        style = int(rng.integers(0, 4))
        vocab = self.capability.vocab_size
        if style == 0:
            return tuple(int(rng.integers(1, vocab)) for _ in range(length))
        if style == 1:
            value = int(rng.integers(1, vocab))
            return (value,) * length
        if style == 2:
            a, b = int(rng.integers(1, vocab)), int(rng.integers(1, vocab))
            return tuple(a if i % 2 == 0 else b for i in range(length))
        block = [int(rng.integers(1, vocab)) for _ in range(max(1, min(3, length)))]
        return tuple(block[i % len(block)] for i in range(length))

    def _requests_for(
        self, category: CaseCategory, rng: np.random.Generator
    ) -> tuple[list[Request], ExecutionMode]:
        if category is CaseCategory.STATELESS_SINGLE:
            length = self._boundary_length(rng)
            tokens = self._patterned_tokens(rng, length)
            return [Request("r0", tokens, length)], ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.STATELESS_BATCH:
            rows = int(rng.integers(2, self.capability.max_batch_rows + 1))
            # Deliberately ragged: one row at each extreme so the batch width is dominated by
            # a single long row and the short rows are heavily padded.
            lengths = [self._boundary_length(rng) for _ in range(rows)]
            lengths[0] = 1
            lengths[-1] = self.capability.max_tokens_per_request
            requests = [
                Request(f"r{row}", self._patterned_tokens(rng, n), n)
                for row, n in enumerate(lengths)
            ]
            return requests, ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.STATELESS_PADDED:
            rows = int(rng.integers(1, self.capability.max_batch_rows + 1))
            limit = self.capability.padded_token_limit()
            requests = []
            for row in range(rows):
                length = self._boundary_length(rng, limit)
                pad = self.capability.max_pad_left if row % 2 == 0 else 1
                requests.append(
                    Request(f"r{row}", self._patterned_tokens(rng, length), length, pad)
                )
            return requests, ExecutionMode.STATELESS_BATCH

        if category is CaseCategory.CACHED_PREFILL_ONLY:
            length = self._boundary_length(rng)
            tokens = self._patterned_tokens(rng, length)
            return [Request("r0", tokens, length)], ExecutionMode.CACHED_DECODE

        if category is CaseCategory.CACHED_DECODE:
            length = max(2, self._boundary_length(rng))
            prefix = min(self._boundary_prefix(rng, length), length - 1)
            tokens = self._patterned_tokens(rng, length)
            return [Request("r0", tokens, prefix)], ExecutionMode.CACHED_DECODE

        count = int(rng.integers(2, self.capability.max_session_requests + 1))
        requests = []
        for index in range(count):
            # Alternate long and short requests so every session contains a real transition
            # between differently shaped cache states.
            length = (
                self.capability.max_tokens_per_request // 2
                if index % 2 == 0
                else self._boundary_length(rng, 4)
            )
            length = max(1, length)
            prefix = self._boundary_prefix(rng, length)
            requests.append(Request(f"r{index}", self._patterned_tokens(rng, length), prefix))
        return requests, ExecutionMode.SESSION


GENERATORS: dict[str, type[CaseGenerator]] = {
    UniformValidGenerator.name: UniformValidGenerator,
    BoundaryAwareGenerator.name: BoundaryAwareGenerator,
}


def build_generator(name: str, capability: GeneratorCapability) -> CaseGenerator:
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; available: {sorted(GENERATORS)}")
    return GENERATORS[name](capability)


__all__ = [
    "ALL_CATEGORIES",
    "DEFAULT_MAX_TOKENS",
    "GENERATORS",
    "MAX_PAD_LEFT",
    "BoundaryAwareGenerator",
    "CaseCategory",
    "CaseGenerator",
    "GeneratorCapability",
    "UniformValidGenerator",
    "build_generator",
    "classify_case",
]
