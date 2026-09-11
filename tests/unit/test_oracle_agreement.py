"""The torch fixture against the independent FP64 NumPy oracle.

What agreement here establishes: the torch fixture computes the intended pre-normalized
decoder-only transformer, to FP32 rounding, for the shapes exercised below.

What it does not establish: that the intended architecture is a good language model (the
weights are random), or that agreement holds for shapes, dtypes, or devices never checked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from evallens.fixtures.config import ModelConfig, WeightDict
from evallens.fixtures.numpy_oracle import (
    oracle_attention_single_head,
    oracle_forward,
)
from evallens.fixtures.tiny_transformer import TinyTransformer, build_model

# FP32 accumulation against an FP64 recomputation. Logits here are O(1e-1..1e0), so this
# bound is roughly 1e3 x float32 epsilon and is not a claim about any other configuration.
ORACLE_TOL = 2e-4


def _run(model: TinyTransformer, tokens: np.ndarray, positions: np.ndarray, valid: np.ndarray):
    with torch.no_grad():
        out = model(
            torch.from_numpy(np.ascontiguousarray(tokens)),
            torch.from_numpy(np.ascontiguousarray(positions)),
            torch.from_numpy(np.ascontiguousarray(valid)),
        )
    return out.numpy().astype(np.float64)


@pytest.mark.parametrize("batch,length", [(1, 1), (1, 5), (2, 3), (3, 8), (1, 16)])
def test_full_prefix_matches_oracle(
    config: ModelConfig, weights: WeightDict, model: TinyTransformer, batch: int, length: int
) -> None:
    rng = np.random.default_rng(1000 + batch * 31 + length)
    tokens = rng.integers(1, config.vocab_size, size=(batch, length)).astype(np.int64)
    positions = np.tile(np.arange(length, dtype=np.int64), (batch, 1))
    valid = np.ones((batch, length), dtype=bool)

    actual = _run(model, tokens, positions, valid)
    expected = oracle_forward(config, weights, tokens, positions, valid)
    assert np.abs(actual - expected).max() < ORACLE_TOL


def test_tiny_single_layer_fixture_matches_oracle(
    tiny_config: ModelConfig, tiny_weights: WeightDict
) -> None:
    """An end-to-end check on a fixture small enough to recompute exhaustively."""
    model = build_model(tiny_config, tiny_weights)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    positions = np.array([[0, 1, 2, 3]], dtype=np.int64)
    valid = np.ones((1, 4), dtype=bool)

    actual = _run(model, tokens, positions, valid)
    expected = oracle_forward(tiny_config, tiny_weights, tokens, positions, valid)
    assert np.abs(actual - expected).max() < ORACLE_TOL


def test_padded_batch_matches_oracle_on_valid_columns(
    config: ModelConfig, weights: WeightDict, model: TinyTransformer
) -> None:
    """Left-padded rows, with position IDs assigned by valid-token index."""
    rng = np.random.default_rng(4242)
    width = 7
    lengths = [7, 4, 2]
    tokens = np.zeros((3, width), dtype=np.int64)
    positions = np.zeros((3, width), dtype=np.int64)
    valid = np.zeros((3, width), dtype=bool)
    for row, length in enumerate(lengths):
        start = width - length
        tokens[row, start:] = rng.integers(1, config.vocab_size, size=length)
        positions[row, start:] = np.arange(length)
        valid[row, start:] = True

    actual = _run(model, tokens, positions, valid)
    expected = oracle_forward(config, weights, tokens, positions, valid)
    for row, length in enumerate(lengths):
        start = width - length
        assert np.abs(actual[row, start:] - expected[row, start:]).max() < ORACLE_TOL


def test_single_head_attention_against_hand_written_oracle() -> None:
    """Causal single-head attention, checked against an independently looped computation."""
    rng = np.random.default_rng(5)
    length, d_head = 5, 4
    q = rng.standard_normal((length, d_head)).astype(np.float32)
    k = rng.standard_normal((length, d_head)).astype(np.float32)
    v = rng.standard_normal((length, d_head)).astype(np.float32)
    valid = np.array([False, True, True, True, True])

    expected = oracle_attention_single_head(q, k, v, valid)

    scores = torch.from_numpy(q) @ torch.from_numpy(k).T / math.sqrt(d_head)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    allowed = causal & torch.from_numpy(valid).view(1, length)
    scores = scores.masked_fill(~allowed, float("-inf"))
    fully_masked = ~allowed.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(fully_masked, 0.0)
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.where(fully_masked, torch.zeros_like(probabilities), probabilities)
    actual = (probabilities @ torch.from_numpy(v)).numpy().astype(np.float64)

    assert np.abs(actual - expected).max() < 1e-6


def test_oracle_attention_is_exactly_causal() -> None:
    """Changing a strictly-future value must not move an earlier attention output."""
    rng = np.random.default_rng(11)
    length, d_head = 6, 3
    q = rng.standard_normal((length, d_head))
    k = rng.standard_normal((length, d_head))
    v = rng.standard_normal((length, d_head))
    valid = np.ones(length, dtype=bool)

    base = oracle_attention_single_head(q, k, v, valid)
    v_changed = v.copy()
    v_changed[4:] += 100.0
    changed = oracle_attention_single_head(q, k, v_changed, valid)

    np.testing.assert_allclose(base[:4], changed[:4], atol=0, rtol=0)
    assert np.abs(base[4:] - changed[4:]).max() > 1.0


def test_fully_masked_query_row_stays_finite() -> None:
    """An all-padding row must produce zeros, not NaN, so a real NaN stays a real signal."""
    rng = np.random.default_rng(13)
    q = rng.standard_normal((3, 4))
    k = rng.standard_normal((3, 4))
    v = rng.standard_normal((3, 4))
    out = oracle_attention_single_head(q, k, v, np.zeros(3, dtype=bool))
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, np.zeros((3, 4)))
