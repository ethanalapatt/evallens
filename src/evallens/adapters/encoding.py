"""Case validation and the canonical case-to-tensor encoding.

This module owns the two conventions that make every downstream comparison meaningful, and
it owns them in exactly one place so that the reference and candidate paths cannot drift:

Position IDs
    A request's logical positions are ``0 .. n_valid - 1``, assigned by valid-token index.
    Left padding moves a token's *column* but never its *position ID*. This is what makes
    padding invariance a property that can be tested rather than a coincidence.

Padded batch layout
    Row ``i`` of a stateless batch is ``[PAD] * pad_left_i + tokens_i + [PAD] * rest``, and
    the batch width is ``max_i(pad_left_i + n_i)``. ``key_valid`` is True exactly on the
    valid columns, so a padding column is never a legal attention key and never appears in
    an ``ExecutionResult``.

Validation is intentionally strict. Silently accepting an all-padding row, an out-of-range
token, or a prefix longer than its request would turn a contract violation into a fake
numerical regression somewhere much further downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evallens.fixtures.config import ModelConfig
from evallens.types import PAD_TOKEN_ID, Case, ExecutionMode, InvalidCaseError, Request

MAX_SESSION_REQUESTS = 3
MAX_BATCH_ROWS = 4


def validate_case(case: Case, config: ModelConfig, *, weights_sha256: str | None = None) -> None:
    """Raise :class:`InvalidCaseError` unless ``case`` satisfies the full input contract."""
    if case.schema_version != 1:
        raise InvalidCaseError(f"unsupported case schema_version {case.schema_version}")
    if case.model_config_id != config.config_id:
        raise InvalidCaseError(
            f"case targets model config {case.model_config_id!r}, adapter holds "
            f"{config.config_id!r}"
        )
    if weights_sha256 is not None and case.weights_sha256 != weights_sha256:
        raise InvalidCaseError(
            f"case targets weights {case.weights_sha256[:16]}..., adapter holds "
            f"{weights_sha256[:16]}..."
        )
    if not case.requests:
        raise InvalidCaseError("a case must contain at least one request")

    seen: set[str] = set()
    for request in case.requests:
        if request.request_id in seen:
            raise InvalidCaseError(f"duplicate request_id {request.request_id!r}")
        seen.add(request.request_id)
        _validate_request(request, case, config)

    mode = case.execution_mode
    if mode is ExecutionMode.STATELESS_BATCH:
        if len(case.requests) > MAX_BATCH_ROWS:
            raise InvalidCaseError(
                f"stateless batch has {len(case.requests)} rows, limit is {MAX_BATCH_ROWS}"
            )
        if case.batch_width > config.max_position:
            raise InvalidCaseError(
                f"padded batch width {case.batch_width} exceeds context limit {config.max_position}"
            )
    elif mode is ExecutionMode.CACHED_DECODE:
        if len(case.requests) != 1:
            raise InvalidCaseError(
                f"cached decode is batch size one; got {len(case.requests)} requests"
            )
    elif mode is ExecutionMode.SESSION:
        if len(case.requests) > MAX_SESSION_REQUESTS:
            raise InvalidCaseError(
                f"session has {len(case.requests)} requests, limit is {MAX_SESSION_REQUESTS}"
            )
    else:  # pragma: no cover - ExecutionMode is closed
        raise InvalidCaseError(f"unknown execution mode {mode!r}")


def _validate_request(request: Request, case: Case, config: ModelConfig) -> None:
    if not request.request_id:
        raise InvalidCaseError("request_id must be a nonempty string")
    if request.n_valid == 0:
        raise InvalidCaseError(
            f"request {request.request_id!r} has no valid tokens; an all-padding request is "
            "never a legal input"
        )
    if request.n_valid > config.max_position:
        raise InvalidCaseError(
            f"request {request.request_id!r} has {request.n_valid} tokens, context limit is "
            f"{config.max_position}"
        )
    for token in request.token_ids:
        if not isinstance(token, int) or isinstance(token, bool):
            raise InvalidCaseError(f"token {token!r} is not an int")
        if token == PAD_TOKEN_ID:
            raise InvalidCaseError(
                f"request {request.request_id!r} contains the reserved padding id "
                f"{PAD_TOKEN_ID}; padding is expressed by pad_left, not by token values"
            )
        if not 0 <= token < config.vocab_size:
            raise InvalidCaseError(
                f"token {token} in request {request.request_id!r} is outside vocabulary "
                f"[0, {config.vocab_size})"
            )
    if request.pad_left < 0:
        raise InvalidCaseError(f"pad_left {request.pad_left} is negative")
    if not 1 <= request.prefix_length <= request.n_valid:
        raise InvalidCaseError(
            f"request {request.request_id!r} prefix_length {request.prefix_length} must lie in "
            f"[1, {request.n_valid}]"
        )
    if case.execution_mode is ExecutionMode.STATELESS_BATCH:
        if request.prefix_length != request.n_valid:
            raise InvalidCaseError(
                f"stateless execution consumes the whole request in one pass, so "
                f"prefix_length must equal {request.n_valid}; got {request.prefix_length}"
            )
    elif request.pad_left != 0:
        raise InvalidCaseError(
            f"cached execution is unpadded batch size one, so pad_left must be 0; got "
            f"{request.pad_left} on request {request.request_id!r}"
        )


@dataclass(frozen=True, slots=True)
class RowLayout:
    """Where one request's valid tokens live inside the padded batch tensor."""

    request_id: str
    row: int
    col_start: int
    n_valid: int

    @property
    def col_stop(self) -> int:
        return self.col_start + self.n_valid


@dataclass(frozen=True, slots=True)
class BatchEncoding:
    """Materialized tensors for a stateless batch, plus the row/column map back to requests.

    Shapes are ``[B, W]`` for all three arrays, where ``W`` is the padded batch width.
    """

    token_ids: np.ndarray
    position_ids: np.ndarray
    key_valid: np.ndarray
    layouts: tuple[RowLayout, ...]

    @property
    def batch_size(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def width(self) -> int:
        return int(self.token_ids.shape[1])


def encode_stateless_batch(case: Case) -> BatchEncoding:
    """Materialize a STATELESS_BATCH case as padded ``[B, W]`` tensors."""
    if case.execution_mode is not ExecutionMode.STATELESS_BATCH:
        raise InvalidCaseError(
            f"encode_stateless_batch requires STATELESS_BATCH, got {case.execution_mode.value}"
        )
    batch = len(case.requests)
    width = case.batch_width
    token_ids = np.full((batch, width), PAD_TOKEN_ID, dtype=np.int64)
    position_ids = np.zeros((batch, width), dtype=np.int64)
    key_valid = np.zeros((batch, width), dtype=bool)
    layouts: list[RowLayout] = []

    for row, request in enumerate(case.requests):
        start = request.pad_left
        stop = start + request.n_valid
        token_ids[row, start:stop] = np.asarray(request.token_ids, dtype=np.int64)
        # Position IDs run 0..n-1 over the valid tokens regardless of where they sit.
        position_ids[row, start:stop] = np.arange(request.n_valid, dtype=np.int64)
        key_valid[row, start:stop] = True
        layouts.append(RowLayout(request.request_id, row, start, request.n_valid))

    return BatchEncoding(token_ids, position_ids, key_valid, tuple(layouts))


def encode_cached_steps(request: Request) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Split a request into its prefill call and its per-token decode calls.

    Returns a list of ``(token_ids[1, T], position_ids[1, T], logical_positions)``. The
    tokens fed at every step are the case's own canonical tokens: this is *teacher forcing*.
    Feeding each implementation its own argmax instead would let the two runs diverge onto
    different inputs, after which comparing their activations would measure nothing.
    """
    steps: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    prefix = request.prefix_length
    prefill_positions = list(range(prefix))
    steps.append(
        (
            np.asarray([request.token_ids[:prefix]], dtype=np.int64),
            np.asarray([prefill_positions], dtype=np.int64),
            prefill_positions,
        )
    )
    for position in range(prefix, request.n_valid):
        steps.append(
            (
                np.asarray([[request.token_ids[position]]], dtype=np.int64),
                np.asarray([[position]], dtype=np.int64),
                [position],
            )
        )
    return steps


def encode_unpadded_single(request: Request) -> BatchEncoding:
    """Materialize one request as an unpadded batch-of-one full-prefix input."""
    tokens = np.asarray([request.token_ids], dtype=np.int64)
    positions = np.asarray([list(range(request.n_valid))], dtype=np.int64)
    valid = np.ones((1, request.n_valid), dtype=bool)
    return BatchEncoding(
        tokens, positions, valid, (RowLayout(request.request_id, 0, 0, request.n_valid),)
    )


__all__ = [
    "MAX_BATCH_ROWS",
    "MAX_SESSION_REQUESTS",
    "BatchEncoding",
    "RowLayout",
    "encode_cached_steps",
    "encode_stateless_batch",
    "encode_unpadded_single",
    "validate_case",
]
