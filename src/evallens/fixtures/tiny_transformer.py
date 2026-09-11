"""A small, deliberately transparent decoder-only transformer.

Design notes
------------
* Attention is written out as explicit matmuls rather than delegating to
  ``F.scaled_dot_product_attention``. A fused kernel would hide exactly the masking and
  scaling decisions this project exists to probe, and would make several injected faults
  impossible to express as shape-valid silent changes.
* Checkpoints are published through an explicit ``recorder`` callback instead of
  ``register_forward_hook``. Hooks fire in call order, and call order is not an alignment
  rule here: one full-prefix reference call fires a hook once for a whole sequence while a
  cached candidate fires it once per decode step. The callback hands the adapter the raw
  ``[B, T, D]`` tensor; only the adapter knows how to map ``(row, column)`` to a semantic
  ``(request_id, logical position)`` address, so only the adapter does that mapping.
* Weights arrive from one canonical NumPy state dictionary (see ``config.make_weights``).

Tensor axes
-----------
``token_ids`` / ``position_ids`` / ``key_valid``: ``[B, T]``.
Hidden states: ``[B, T, d_model]``. Attention internals: ``[B, n_heads, T, d_head]``.
Logits: ``[B, T, vocab_size]``.

With a KV cache of length ``L``, the ``T`` incoming tokens occupy absolute key columns
``L .. L+T-1`` and attend over all ``L+T`` columns.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn

from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict
from evallens.types import CheckpointKind

Recorder = Callable[[str, CheckpointKind, Tensor], None]
"""``recorder(layer_name, kind, tensor[B, T, D])`` — called at each semantic checkpoint."""


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Holds ``[B, n_heads, L, d_head]`` tensors per layer. ``clear_layer`` exists so that a
    mutant can model the realistic bug of resetting only part of the cache.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: list[Tensor | None] = [None] * n_layers
        self.values: list[Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        first = self.keys[0]
        return 0 if first is None else int(first.shape[2])

    def clear(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    def clear_layer(self, index: int) -> None:
        self.keys[index] = None
        self.values[index] = None

    def append(self, layer: int, key: Tensor, value: Tensor, mode: str) -> tuple[Tensor, Tensor]:
        """Append ``key``/``value`` for one layer and return the full key/value history.

        ``mode`` selects the cache-indexing behavior; ``"correct"`` is a plain concatenation.
        """
        past_k, past_v = self.keys[layer], self.values[layer]
        if past_k is None or past_v is None:
            full_k, full_v = key, value
        elif mode == "write_overwrite_last" and past_k.shape[2] > 0:
            # Injected fault: the new entry overwrites the most recent slot instead of
            # extending the cache, so one earlier key/value pair is permanently lost.
            full_k = torch.cat([past_k[:, :, :-1], key], dim=2)
            full_v = torch.cat([past_v[:, :, :-1], value], dim=2)
        else:
            full_k = torch.cat([past_k, key], dim=2)
            full_v = torch.cat([past_v, value], dim=2)
        self.keys[layer] = full_k
        self.values[layer] = full_v
        if mode == "read_drop_oldest" and full_k.shape[2] > 1:
            # Injected fault: the read view drops the oldest cached position, so the query
            # silently attends to a truncated history while the cache itself stays intact.
            return full_k[:, :, 1:], full_v[:, :, 1:]
        return full_k, full_v


def _layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float, mode: str) -> Tensor:
    """Layer normalization with the normalization axis and epsilon under explicit control."""
    if mode == "large_eps":
        eps = 1e-1
    axis = 1 if mode == "wrong_axis" else -1
    mean = x.mean(dim=axis, keepdim=True)
    var = x.var(dim=axis, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    result: Tensor = normalized * weight + bias
    return result


def _gelu(x: Tensor) -> Tensor:
    """Exact (erf) GELU. The NumPy oracle recomputes this from the erf definition."""
    result: Tensor = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return result


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, behavior: Behavior, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.behavior = behavior
        self.layer_index = layer_index
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def scale(self) -> float:
        mode = self.behavior.attn_scale
        if mode == "no_sqrt":
            return 1.0 / self.config.d_head
        if mode == "d_model":
            return 1.0 / math.sqrt(self.config.d_model)
        return 1.0 / math.sqrt(self.config.d_head)

    def build_mask(self, n_query: int, past_len: int, key_valid: Tensor) -> Tensor:
        """Boolean ``[B, 1, T, L+T]`` mask: ``True`` where a query may attend to a key.

        Causality is expressed over *column* indices. Because a request's valid tokens
        occupy a contiguous ascending column range, column causality and logical-position
        causality agree; padding is excluded separately by ``key_valid``. Keeping the two
        concerns separate is what lets the mask fault and the padding fault be injected
        independently.
        """
        device = key_valid.device
        total_keys = int(key_valid.shape[1])
        q_abs = torch.arange(past_len, past_len + n_query, device=device).view(n_query, 1)
        k_abs = torch.arange(total_keys, device=device).view(1, total_keys)

        causal_mode = self.behavior.causal
        if causal_mode == "off_by_one":
            # Injected fault: the boundary admits exactly one future position.
            causal = k_abs <= q_abs + 1
        else:
            causal = k_abs <= q_abs
            if causal_mode == "leak_last":
                # Injected fault: the final key column is always visible.
                causal = causal | (k_abs == total_keys - 1)

        allowed = causal.view(1, 1, n_query, total_keys)

        pad_mode = self.behavior.pad_mask
        if pad_mode == "ignore":
            # Injected fault: padding columns are attended to as if they were real tokens.
            return allowed.expand(key_valid.shape[0], 1, n_query, total_keys)
        valid = key_valid
        if pad_mode == "right_only":
            # Injected fault: only trailing padding is masked, so left padding leaks in.
            first_valid = valid.float().argmax(dim=1, keepdim=True)
            columns = torch.arange(total_keys, device=device).view(1, total_keys)
            valid = valid | (columns < first_valid)
        return allowed & valid.view(key_valid.shape[0], 1, 1, total_keys)

    def forward(
        self,
        x: Tensor,
        key_valid: Tensor,
        cache: KVCache | None,
    ) -> Tensor:
        batch, n_query, _ = x.shape
        heads, d_head = self.config.n_heads, self.config.d_head

        qkv = self.qkv(x).view(batch, n_query, 3, heads, d_head).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        past_len = 0
        if cache is not None:
            past_len = cache.length
            key, value = cache.append(self.layer_index, key, value, self.behavior.cache_index)

        if self.behavior.batch == "v_roll" and batch > 1:
            # Injected fault: every row reads the previous row's values.
            value = torch.roll(value, shifts=1, dims=0)

        n_keys = int(key.shape[2])
        if int(key_valid.shape[1]) != n_keys:
            # Cached execution is batch size one and unpadded; earlier cached columns are
            # valid by construction.
            pad = torch.ones(batch, n_keys - int(key_valid.shape[1]), dtype=torch.bool)
            key_valid = torch.cat([pad.to(key_valid.device), key_valid], dim=1)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale()
        mask = self.build_mask(n_query, past_len, key_valid)
        scores = scores.masked_fill(~mask, float("-inf"))
        # A fully masked query row (an all-padding row) would produce NaN here. Such rows
        # are never compared, but we keep them finite so that a genuine nonfinite output on
        # a valid position remains an unambiguous signal.
        fully_masked = ~mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(fully_masked, torch.zeros_like(weights), weights)

        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, n_query, self.config.d_model)

        if self.behavior.batch == "row_leak_first" and batch > 1 and self.layer_index == 0:
            # Injected fault: one row's attention output is broadcast over the whole batch.
            context = context[0:1].expand_as(context).clone()

        projected: Tensor = self.out(context)
        return projected


class MLP(nn.Module):
    """Two-layer position-wise feed-forward network with exact GELU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.fc2(_gelu(self.fc1(x)))
        return result


class Block(nn.Module):
    def __init__(self, config: ModelConfig, behavior: Behavior, index: int) -> None:
        super().__init__()
        self.config = config
        self.behavior = behavior
        self.index = index
        self.name = f"block{index}"
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config, behavior, index)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.mlp = MLP(config)

    def _norm(self, x: Tensor, layer: nn.LayerNorm) -> Tensor:
        return _layer_norm(x, layer.weight, layer.bias, self.config.norm_eps, self.behavior.norm)

    def forward(
        self,
        x: Tensor,
        key_valid: Tensor,
        cache: KVCache | None,
        recorder: Recorder | None,
    ) -> Tensor:
        attn_out = self.attn(self._norm(x, self.ln1), key_valid, cache)
        if recorder is not None:
            recorder(self.name, CheckpointKind.ATTN_OUT, attn_out)
        x = x + attn_out

        mlp_out = self.mlp(self._norm(x, self.ln2))
        if recorder is not None:
            recorder(self.name, CheckpointKind.MLP_OUT, mlp_out)
        x = x + mlp_out

        if recorder is not None:
            recorder(self.name, CheckpointKind.BLOCK_OUT, x)
        return x


class TinyTransformer(nn.Module):
    """Pre-normalized decoder-only transformer with learned position embeddings."""

    def __init__(self, config: ModelConfig, behavior: Behavior | None = None) -> None:
        super().__init__()
        self.config = config
        self.behavior = behavior or Behavior()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position, config.d_model)
        self.blocks = nn.ModuleList(
            [Block(config, self.behavior, i) for i in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.eval()

    def new_cache(self) -> KVCache:
        return KVCache(self.config.n_layers)

    @torch.no_grad()
    def load_canonical_weights(self, weights: WeightDict) -> None:
        """Copy the canonical NumPy state dictionary into this module.

        Raises on any missing or mis-shaped entry rather than partially loading, so a
        weights-hash mismatch can never masquerade as a numerical regression.
        """
        expected = dict(self.named_parameters())
        wanted = set(weights)
        if self.config.tie_embeddings:
            wanted.discard("lm_head.weight")
        for name in sorted(wanted):
            if name not in expected:
                raise KeyError(f"canonical weights contain unknown parameter {name!r}")
            target = expected[name]
            source = np.asarray(weights[name], dtype=np.float32)
            if tuple(target.shape) != source.shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: module {tuple(target.shape)} vs "
                    f"canonical {source.shape}"
                )
            target.copy_(torch.from_numpy(source.copy()))
        missing = set(expected) - wanted
        if self.config.tie_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            raise KeyError(f"canonical weights are missing {sorted(missing)}")
        if self.behavior.perturb_scale:
            # Benign control: a real but deliberately sub-tolerance difference.
            self.lm_head.weight.mul_(1.0 + self.behavior.perturb_scale)
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        token_ids: Tensor,
        position_ids: Tensor,
        key_valid: Tensor,
        cache: KVCache | None = None,
        recorder: Recorder | None = None,
    ) -> Tensor:
        """Run the model over ``[B, T]`` tokens and return ``[B, T, vocab_size]`` logits."""
        hidden = self.embed_tokens(token_ids) + self.embed_positions(position_ids)
        if recorder is not None:
            recorder("embed", CheckpointKind.EMBEDDING, hidden)

        for block in self.blocks:
            hidden = block(hidden, key_valid, cache, recorder)

        hidden = _layer_norm(
            hidden, self.ln_f.weight, self.ln_f.bias, self.config.norm_eps, self.behavior.norm
        )
        if recorder is not None:
            recorder("final", CheckpointKind.FINAL_NORM, hidden)

        logits: Tensor = self.lm_head(hidden)
        if recorder is not None:
            recorder("final", CheckpointKind.LOGITS, logits)
        return logits


def build_model(
    config: ModelConfig, weights: WeightDict, behavior: Behavior | None = None
) -> TinyTransformer:
    """Construct a model in inference mode with the canonical weights already loaded."""
    model = TinyTransformer(config, behavior)
    model.load_canonical_weights(weights)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "Attention",
    "Block",
    "KVCache",
    "Recorder",
    "TinyTransformer",
    "build_model",
]
