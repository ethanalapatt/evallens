"""Native model fixtures and the independent oracle used to check them."""

from evallens.fixtures.behavior import REFERENCE_BEHAVIOR, Behavior
from evallens.fixtures.config import (
    SCALE_FIXTURE,
    UNIT_FIXTURE,
    ModelConfig,
    make_weights,
    parameter_count,
    weights_sha256,
)
from evallens.fixtures.tiny_transformer import KVCache, TinyTransformer, build_model

__all__ = [
    "REFERENCE_BEHAVIOR",
    "SCALE_FIXTURE",
    "UNIT_FIXTURE",
    "Behavior",
    "KVCache",
    "ModelConfig",
    "TinyTransformer",
    "build_model",
    "make_weights",
    "parameter_count",
    "weights_sha256",
]
