"""Fixture configuration and the canonical weight initializer.

One canonical state dictionary is built here as plain ``numpy`` arrays and then copied into
every implementation. Sharing a *seed* between two implementations is not enough: if their
initialization paths differ at all (parameter creation order, an extra RNG draw, a different
fan-in convention) the weights silently diverge and every downstream comparison becomes
meaningless. Copying one hashed dictionary removes that whole failure mode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

WeightDict = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Geometry of the native decoder-only fixture.

    The unit fixture is deliberately tiny so that thousands of CPU evaluations fit inside a
    benchmark budget on a laptop, and so that an FP64 NumPy oracle can recompute an entire
    forward pass by hand in milliseconds.
    """

    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    vocab_size: int = 97
    max_position: int = 128
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    init_seed: int = 20240917
    name: str = "tiny-2L-64d-4h"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must leave room for at least one non-padding token")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def config_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"{self.name}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_id"] = self.config_id
        payload["d_head"] = self.d_head
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ModelConfig:
        known = set(ModelConfig.__dataclass_fields__)
        fields = {k: v for k, v in payload.items() if k in known}
        return ModelConfig(**fields)


UNIT_FIXTURE = ModelConfig()
"""The fixture used by correctness tests, the demo, and the primary benchmark."""

SCALE_FIXTURE = ModelConfig(
    n_layers=6,
    d_model=384,
    n_heads=6,
    d_ff=1536,
    vocab_size=4096,
    max_position=128,
    name="scale-6L-384d-6h",
)
"""Optional ~11M-parameter scale check. Never used by the primary correctness suite."""


def parameter_count(config: ModelConfig) -> int:
    per_block = (
        2 * config.d_model  # ln1
        + 3 * config.d_model * config.d_model
        + 3 * config.d_model  # qkv
        + config.d_model * config.d_model
        + config.d_model  # attn out
        + 2 * config.d_model  # ln2
        + config.d_ff * config.d_model
        + config.d_ff  # fc1
        + config.d_model * config.d_ff
        + config.d_model  # fc2
    )
    head = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    return (
        config.vocab_size * config.d_model
        + config.max_position * config.d_model
        + config.n_layers * per_block
        + 2 * config.d_model
        + head
        + config.vocab_size
    )


def make_weights(config: ModelConfig, seed: int | None = None) -> WeightDict:
    """Build the canonical FP32 state dictionary for ``config``.

    Draw order is fixed and documented by the code below; changing it changes
    ``weights_sha256`` and therefore invalidates every previously recorded case, which is
    the intended behavior.
    """
    rng = np.random.default_rng(config.init_seed if seed is None else seed)
    d, f, v, p = config.d_model, config.d_ff, config.vocab_size, config.max_position

    def normal(*shape: int, scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights: WeightDict = {
        "embed_tokens.weight": normal(v, d),
        "embed_positions.weight": normal(p, d, scale=0.01),
    }
    for i in range(config.n_layers):
        weights[f"blocks.{i}.ln1.weight"] = np.ones(d, dtype=np.float32)
        weights[f"blocks.{i}.ln1.bias"] = np.zeros(d, dtype=np.float32)
        weights[f"blocks.{i}.attn.qkv.weight"] = normal(3 * d, d)
        weights[f"blocks.{i}.attn.qkv.bias"] = normal(3 * d, scale=0.005)
        weights[f"blocks.{i}.attn.out.weight"] = normal(d, d)
        weights[f"blocks.{i}.attn.out.bias"] = normal(d, scale=0.005)
        weights[f"blocks.{i}.ln2.weight"] = np.ones(d, dtype=np.float32)
        weights[f"blocks.{i}.ln2.bias"] = np.zeros(d, dtype=np.float32)
        weights[f"blocks.{i}.mlp.fc1.weight"] = normal(f, d)
        weights[f"blocks.{i}.mlp.fc1.bias"] = normal(f, scale=0.005)
        weights[f"blocks.{i}.mlp.fc2.weight"] = normal(d, f)
        weights[f"blocks.{i}.mlp.fc2.bias"] = normal(d, scale=0.005)
    weights["ln_f.weight"] = np.ones(d, dtype=np.float32)
    weights["ln_f.bias"] = np.zeros(d, dtype=np.float32)
    if not config.tie_embeddings:
        weights["lm_head.weight"] = normal(v, d)
    weights["lm_head.bias"] = normal(v, scale=0.005)
    return weights


def weights_sha256(weights: WeightDict) -> str:
    """Order-independent content hash over names, shapes, dtypes, and raw bytes."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        array = np.ascontiguousarray(weights[name], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(b"float32")
        digest.update(array.tobytes())
    return digest.hexdigest()


def effective_lm_head(weights: WeightDict, config: ModelConfig) -> np.ndarray:
    """Resolve the output projection, honoring the explicit tie/untie decision."""
    if config.tie_embeddings:
        return weights["embed_tokens.weight"]
    return weights["lm_head.weight"]


__all__ = [
    "SCALE_FIXTURE",
    "UNIT_FIXTURE",
    "ModelConfig",
    "WeightDict",
    "effective_lm_head",
    "make_weights",
    "parameter_count",
    "weights_sha256",
]
