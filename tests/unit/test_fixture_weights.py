"""Canonical weights: determinism, hashing, loading, and the tie/untie decision."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evallens.fixtures.config import (
    UNIT_FIXTURE,
    ModelConfig,
    make_weights,
    parameter_count,
    weights_sha256,
)
from evallens.fixtures.tiny_transformer import TinyTransformer, build_model


def test_weights_are_deterministic_across_calls(config: ModelConfig) -> None:
    first, second = make_weights(config), make_weights(config)
    assert weights_sha256(first) == weights_sha256(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_weights_hash_changes_with_seed(config: ModelConfig) -> None:
    assert weights_sha256(make_weights(config, seed=1)) != weights_sha256(
        make_weights(config, seed=2)
    )


def test_weights_hash_is_order_independent(config: ModelConfig) -> None:
    weights = make_weights(config)
    shuffled = dict(reversed(list(weights.items())))
    assert weights_sha256(shuffled) == weights_sha256(weights)


def test_weights_hash_detects_a_single_element_change(config: ModelConfig) -> None:
    weights = make_weights(config)
    baseline = weights_sha256(weights)
    perturbed = {k: v.copy() for k, v in weights.items()}
    perturbed["ln_f.weight"][0] += np.float32(1e-6)
    assert weights_sha256(perturbed) != baseline


def test_all_weights_are_float32_and_finite(config: ModelConfig) -> None:
    for name, array in make_weights(config).items():
        assert array.dtype == np.float32, name
        assert np.isfinite(array).all(), name


def test_parameter_count_matches_the_built_module(config: ModelConfig) -> None:
    model = build_model(config, make_weights(config))
    actual = sum(p.numel() for p in model.parameters())
    assert actual == parameter_count(config)


def test_loading_copies_every_canonical_array(config: ModelConfig) -> None:
    weights = make_weights(config)
    model = build_model(config, weights)
    named = dict(model.named_parameters())
    for name, array in weights.items():
        np.testing.assert_array_equal(named[name].detach().numpy(), array)


def test_loading_rejects_a_missing_parameter(config: ModelConfig) -> None:
    weights = make_weights(config)
    del weights["ln_f.bias"]
    with pytest.raises(KeyError, match="missing"):
        TinyTransformer(config).load_canonical_weights(weights)


def test_loading_rejects_an_unknown_parameter(config: ModelConfig) -> None:
    weights = make_weights(config)
    weights["blocks.0.attn.nonexistent"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(KeyError, match="unknown parameter"):
        TinyTransformer(config).load_canonical_weights(weights)


def test_loading_rejects_a_shape_mismatch(config: ModelConfig) -> None:
    weights = make_weights(config)
    weights["ln_f.weight"] = np.ones(config.d_model + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        TinyTransformer(config).load_canonical_weights(weights)


def test_tied_embeddings_share_storage_and_untied_do_not() -> None:
    tied = ModelConfig(tie_embeddings=True, name="tied")
    untied = ModelConfig(tie_embeddings=False, name="untied")

    tied_model = build_model(tied, make_weights(tied))
    assert tied_model.lm_head.weight.data_ptr() == tied_model.embed_tokens.weight.data_ptr()

    untied_model = build_model(untied, make_weights(untied))
    assert untied_model.lm_head.weight.data_ptr() != untied_model.embed_tokens.weight.data_ptr()


def test_tied_config_ignores_a_supplied_lm_head() -> None:
    """A tied model must not silently load a separate head and then ignore it."""
    tied = ModelConfig(tie_embeddings=True, name="tied2")
    weights = make_weights(tied)
    weights["lm_head.weight"] = np.zeros((tied.vocab_size, tied.d_model), dtype=np.float32)
    model = build_model(tied, weights)
    np.testing.assert_array_equal(
        model.lm_head.weight.detach().numpy(), weights["embed_tokens.weight"]
    )


def test_model_is_in_eval_mode_with_grad_disabled(model: TinyTransformer) -> None:
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_default_config_ids_are_stable_and_distinct() -> None:
    assert UNIT_FIXTURE.config_id == ModelConfig().config_id
    assert UNIT_FIXTURE.config_id != ModelConfig(n_layers=3).config_id


def test_config_rejects_indivisible_head_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=64, n_heads=5)


def test_forward_produces_finite_float32_logits(
    config: ModelConfig, model: TinyTransformer
) -> None:
    tokens = torch.tensor([[3, 5, 7]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    valid = torch.ones(1, 3, dtype=torch.bool)
    logits = model(tokens, positions, valid)
    assert logits.shape == (1, 3, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
