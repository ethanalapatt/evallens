"""Structural invariances the reference fixture must satisfy.

These are properties of the *intended semantics*, not of any particular implementation, so
each one is stated in terms of what a caller is entitled to assume. Every injected fault
family in ``bench/mutants`` breaks at least one of them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evallens.adapters.encoding import encode_stateless_batch
from evallens.fixtures.config import ModelConfig
from evallens.fixtures.tiny_transformer import TinyTransformer
from evallens.types import Case, ExecutionMode, Request

# Two runs of the same FP32 kernels on the same machine agree far more tightly than the
# FP32-vs-FP64 oracle bound; these invariances only involve reordering identical work.
INVARIANCE_TOL = 1e-5


def _logits(model: TinyTransformer, encoding) -> np.ndarray:
    with torch.no_grad():
        out = model(
            torch.from_numpy(encoding.token_ids),
            torch.from_numpy(encoding.position_ids),
            torch.from_numpy(encoding.key_valid),
        )
    return out.numpy().astype(np.float64)


def _stateless_case(config: ModelConfig, requests: list[Request]) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256="0" * 64,
        requests=requests,
        execution_mode=ExecutionMode.STATELESS_BATCH,
        input_seed=0,
    )


def _request(name: str, tokens: list[int], pad_left: int = 0) -> Request:
    return Request(name, tuple(tokens), prefix_length=len(tokens), pad_left=pad_left)


@pytest.mark.parametrize("prefix_len,extra", [(1, 5), (3, 4), (6, 2), (8, 8)])
def test_causal_prefix_invariance(
    config: ModelConfig, model: TinyTransformer, prefix_len: int, extra: int
) -> None:
    """Appending future tokens must not change the logits at earlier valid positions.

    This is the single most load-bearing invariance in the project: without it, comparing a
    full-prefix reference against an incremental cached candidate would be meaningless.
    """
    rng = np.random.default_rng(900 + prefix_len * 17 + extra)
    prefix = [int(t) for t in rng.integers(1, config.vocab_size, size=prefix_len)]
    suffix = [int(t) for t in rng.integers(1, config.vocab_size, size=extra)]

    short = _logits(model, encode_stateless_batch(_stateless_case(config, [_request("r", prefix)])))
    long = _logits(
        model, encode_stateless_batch(_stateless_case(config, [_request("r", prefix + suffix)]))
    )

    assert np.abs(short[0] - long[0, :prefix_len]).max() < INVARIANCE_TOL


def test_causal_prefix_invariance_holds_for_every_split(
    config: ModelConfig, model: TinyTransformer
) -> None:
    """Every proper prefix of one sequence reproduces that sequence's earlier logits."""
    rng = np.random.default_rng(77)
    tokens = [int(t) for t in rng.integers(1, config.vocab_size, size=9)]
    full = _logits(model, encode_stateless_batch(_stateless_case(config, [_request("r", tokens)])))
    for cut in range(1, len(tokens)):
        partial = _logits(
            model, encode_stateless_batch(_stateless_case(config, [_request("r", tokens[:cut])]))
        )
        assert np.abs(partial[0] - full[0, :cut]).max() < INVARIANCE_TOL, f"cut={cut}"


@pytest.mark.parametrize("pad_left", [1, 2, 5, 11])
def test_padding_invariance(config: ModelConfig, model: TinyTransformer, pad_left: int) -> None:
    """Left padding shifts columns but not position IDs, so valid logits must not move."""
    rng = np.random.default_rng(300 + pad_left)
    tokens = [int(t) for t in rng.integers(1, config.vocab_size, size=6)]

    unpadded_enc = encode_stateless_batch(_stateless_case(config, [_request("r", tokens)]))
    padded_enc = encode_stateless_batch(
        _stateless_case(config, [_request("r", tokens, pad_left=pad_left)])
    )
    unpadded = _logits(model, unpadded_enc)[0]
    padded = _logits(model, padded_enc)[0, pad_left : pad_left + len(tokens)]

    assert np.abs(unpadded - padded).max() < INVARIANCE_TOL


def test_padding_invariance_within_a_ragged_batch(
    config: ModelConfig, model: TinyTransformer
) -> None:
    """A short row padded up to a long row's width keeps the logits it had alone."""
    rng = np.random.default_rng(31337)
    short_tokens = [int(t) for t in rng.integers(1, config.vocab_size, size=3)]
    long_tokens = [int(t) for t in rng.integers(1, config.vocab_size, size=10)]

    alone = _logits(
        model, encode_stateless_batch(_stateless_case(config, [_request("s", short_tokens)]))
    )[0]

    ragged_case = _stateless_case(
        config,
        [_request("s", short_tokens, pad_left=7), _request("l", long_tokens)],
    )
    encoding = encode_stateless_batch(ragged_case)
    assert encoding.width == 10
    together = _logits(model, encoding)[0, 7:10]

    assert np.abs(alone - together).max() < INVARIANCE_TOL


def test_batch_permutation_invariance(config: ModelConfig, model: TinyTransformer) -> None:
    """Stateless rows are independent: permuting them permutes the outputs and nothing else."""
    rng = np.random.default_rng(2024)
    lengths = [4, 7, 2, 5]
    requests = [
        _request(f"r{i}", [int(t) for t in rng.integers(1, config.vocab_size, size=n)])
        for i, n in enumerate(lengths)
    ]

    base_case = _stateless_case(config, requests)
    base_encoding = encode_stateless_batch(base_case)
    base = _logits(model, base_encoding)

    order = [2, 0, 3, 1]
    permuted = [requests[i] for i in order]
    # Preserve each row's padding so the only difference is row order.
    width = base_case.batch_width
    permuted = [
        _request(r.request_id, list(r.token_ids), pad_left=width - r.n_valid) for r in permuted
    ]
    original_padded = [
        _request(r.request_id, list(r.token_ids), pad_left=width - r.n_valid) for r in requests
    ]

    original = _logits(model, encode_stateless_batch(_stateless_case(config, original_padded)))
    shuffled = _logits(model, encode_stateless_batch(_stateless_case(config, permuted)))

    for new_row, old_row in enumerate(order):
        start = width - lengths[old_row]
        assert (
            np.abs(original[old_row, start:] - shuffled[new_row, start:]).max() < INVARIANCE_TOL
        ), f"row {old_row} moved to {new_row}"
    assert base.shape == original.shape


def test_stateless_rows_do_not_influence_each_other(
    config: ModelConfig, model: TinyTransformer
) -> None:
    """Changing one row's tokens must not perturb any other row."""
    rng = np.random.default_rng(64)
    a = [int(t) for t in rng.integers(1, config.vocab_size, size=5)]
    b = [int(t) for t in rng.integers(1, config.vocab_size, size=5)]
    c = [int(t) for t in rng.integers(1, config.vocab_size, size=5)]

    first = _logits(
        model, encode_stateless_batch(_stateless_case(config, [_request("a", a), _request("b", b)]))
    )
    second = _logits(
        model, encode_stateless_batch(_stateless_case(config, [_request("a", a), _request("b", c)]))
    )
    assert np.abs(first[0] - second[0]).max() < INVARIANCE_TOL


def test_repeated_execution_is_bitwise_stable(config: ModelConfig, model: TinyTransformer) -> None:
    """With pinned threads, re-running an identical case reproduces identical bits.

    Stability here is what lets a genuine mismatch be distinguished from run-to-run noise.
    """
    rng = np.random.default_rng(8)
    tokens = [int(t) for t in rng.integers(1, config.vocab_size, size=12)]
    encoding = encode_stateless_batch(_stateless_case(config, [_request("r", tokens)]))
    first = _logits(model, encoding)
    for _ in range(3):
        np.testing.assert_array_equal(first, _logits(model, encoding))
