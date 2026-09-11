"""An independent FP64 NumPy recomputation of the fixture's forward pass.

Independence is the whole point, so this module deliberately does *not* reuse anything from
``tiny_transformer``: it builds its attention mask with explicit Python loops over index
pairs rather than broadcast comparisons, splits QKV by hand, and evaluates GELU from the
``math.erf`` definition. If both files shared a masking helper, the oracle would agree with
the implementation precisely on the class of bug the oracle exists to catch.

What this establishes, and what it does not
-------------------------------------------
Agreement here is evidence that the torch fixture computes the *intended* pre-norm
decoder-only transformer to FP32 rounding, for the shapes actually exercised. It is not
evidence that the intended architecture is a good language model (the weights are random),
nor that agreement extends to shapes, dtypes, or devices that were never checked.

Axes: ``token_ids``/``position_ids``/``valid`` are ``[B, T]``; the return value is
``[B, T, vocab_size]`` in float64.
"""

from __future__ import annotations

import math

import numpy as np

from evallens.fixtures.config import ModelConfig, WeightDict, effective_lm_head

_ERF = np.vectorize(math.erf, otypes=[np.float64])


def _gelu_exact(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0))), dtype=np.float64)


def _layernorm(x: np.ndarray, gain: np.ndarray, offset: np.ndarray, eps: float) -> np.ndarray:
    """Normalize the last axis. Written with explicit moments, not a library call."""
    mu = x.sum(axis=-1, keepdims=True) / x.shape[-1]
    centered = x - mu
    sigma2 = (centered * centered).sum(axis=-1, keepdims=True) / x.shape[-1]
    return np.asarray(centered / np.sqrt(sigma2 + eps) * gain + offset, dtype=np.float64)


def _affine(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """``y = x @ W^T + b``, matching ``torch.nn.Linear``'s stored orientation."""
    return np.asarray(x @ weight.T + bias, dtype=np.float64)


def _attention_mask(valid_row: np.ndarray) -> np.ndarray:
    """Build one row's ``[T, T]`` boolean mask with explicit index loops.

    ``mask[q, k]`` is True when query column ``q`` may attend to key column ``k``: the key
    must not be padding, and it must not lie in the future.
    """
    length = valid_row.shape[0]
    mask = np.zeros((length, length), dtype=bool)
    for q in range(length):
        for k in range(length):
            if k > q:
                continue
            if not bool(valid_row[k]):
                continue
            mask[q, k] = True
    return mask


def _softmax_rows(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise masked softmax over the last axis, computed one row at a time."""
    out = np.zeros_like(scores)
    rows = scores.reshape(-1, scores.shape[-1])
    mask_rows = np.broadcast_to(mask, scores.shape).reshape(-1, scores.shape[-1])
    flat = out.reshape(-1, scores.shape[-1])
    for i in range(rows.shape[0]):
        allowed = mask_rows[i]
        if not allowed.any():
            continue
        selected = rows[i][allowed]
        shifted = selected - selected.max()
        exponentiated = np.exp(shifted)
        flat[i][allowed] = exponentiated / exponentiated.sum()
    return flat.reshape(scores.shape)


def oracle_forward(
    config: ModelConfig,
    weights: WeightDict,
    token_ids: np.ndarray,
    position_ids: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Recompute logits in float64. Returns ``[B, T, vocab_size]``."""
    tokens = np.asarray(token_ids, dtype=np.int64)
    positions = np.asarray(position_ids, dtype=np.int64)
    valid_mask = np.asarray(valid, dtype=bool)
    batch, length = tokens.shape
    heads, d_head, eps = config.n_heads, config.d_head, config.norm_eps

    w = {name: np.asarray(array, dtype=np.float64) for name, array in weights.items()}

    hidden = w["embed_tokens.weight"][tokens] + w["embed_positions.weight"][positions]

    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}."
        normed = _layernorm(hidden, w[prefix + "ln1.weight"], w[prefix + "ln1.bias"], eps)
        fused = _affine(normed, w[prefix + "attn.qkv.weight"], w[prefix + "attn.qkv.bias"])

        width = config.d_model
        queries = fused[:, :, 0:width]
        keys = fused[:, :, width : 2 * width]
        values = fused[:, :, 2 * width : 3 * width]

        context = np.zeros((batch, length, width), dtype=np.float64)
        scaling = 1.0 / math.sqrt(d_head)
        for b in range(batch):
            row_mask = _attention_mask(valid_mask[b])
            for h in range(heads):
                lo, hi = h * d_head, (h + 1) * d_head
                q_h, k_h, v_h = queries[b, :, lo:hi], keys[b, :, lo:hi], values[b, :, lo:hi]
                scores = (q_h @ k_h.T) * scaling
                probabilities = _softmax_rows(scores, row_mask)
                context[b, :, lo:hi] = probabilities @ v_h

        attn_out = _affine(context, w[prefix + "attn.out.weight"], w[prefix + "attn.out.bias"])
        hidden = hidden + attn_out

        normed2 = _layernorm(hidden, w[prefix + "ln2.weight"], w[prefix + "ln2.bias"], eps)
        inner = _gelu_exact(
            _affine(normed2, w[prefix + "mlp.fc1.weight"], w[prefix + "mlp.fc1.bias"])
        )
        hidden = hidden + _affine(inner, w[prefix + "mlp.fc2.weight"], w[prefix + "mlp.fc2.bias"])

    hidden = _layernorm(hidden, w["ln_f.weight"], w["ln_f.bias"], eps)
    head = np.asarray(effective_lm_head(weights, config), dtype=np.float64)
    return _affine(hidden, head, w["lm_head.bias"])


def oracle_attention_single_head(
    queries: np.ndarray, keys: np.ndarray, values: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Standalone causal attention for one head, for the tiny hand-checkable unit tests."""
    scaling = 1.0 / math.sqrt(queries.shape[-1])
    mask = _attention_mask(np.asarray(valid, dtype=bool))
    scores = (
        np.asarray(queries, dtype=np.float64) @ np.asarray(keys, dtype=np.float64).T
    ) * scaling
    probabilities = _softmax_rows(scores, mask)
    return probabilities @ np.asarray(values, dtype=np.float64)


__all__ = [
    "oracle_attention_single_head",
    "oracle_forward",
]
