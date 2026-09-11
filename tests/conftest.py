"""Shared pytest fixtures.

Thread counts are pinned once for the whole session so that numerical results and timings
do not depend on whatever the machine happened to be doing.
"""

from __future__ import annotations

import numpy as np
import pytest

from evallens.fixtures.config import UNIT_FIXTURE, ModelConfig, WeightDict, make_weights
from evallens.fixtures.tiny_transformer import TinyTransformer, build_model
from evallens.resources import set_deterministic_threads

set_deterministic_threads(2)


@pytest.fixture(scope="session")
def config() -> ModelConfig:
    return UNIT_FIXTURE


@pytest.fixture(scope="session")
def weights(config: ModelConfig) -> WeightDict:
    return make_weights(config)


@pytest.fixture(scope="session")
def model(config: ModelConfig, weights: WeightDict) -> TinyTransformer:
    return build_model(config, weights)


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    """A one-layer, one-head fixture small enough for fully hand-checkable oracle runs."""
    return ModelConfig(
        n_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        vocab_size=11,
        max_position=16,
        name="oracle-1L-8d-2h",
    )


@pytest.fixture(scope="session")
def tiny_weights(tiny_config: ModelConfig) -> WeightDict:
    return make_weights(tiny_config)


def random_tokens(rng: np.random.Generator, config: ModelConfig, length: int) -> list[int]:
    """Valid tokens only: never the reserved padding id."""
    return [int(t) for t in rng.integers(1, config.vocab_size, size=length)]
