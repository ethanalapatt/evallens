"""Native adapters against the real fixture.

These are the M2 known-good controls. They establish that the *correct* candidate agrees
with the reference before any fault is ever injected — without that, every later "detection"
would just be measuring a bug in EvalLens itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from evallens.adapters.native import (
    CandidateAdapter,
    CaptureBudget,
    ReferenceAdapter,
    build_adapter_pair,
)
from evallens.compare import compare
from evallens.fixtures.behavior import Behavior
from evallens.fixtures.config import ModelConfig, WeightDict, weights_sha256
from evallens.types import (
    Case,
    CheckpointKind,
    ExecutionMode,
    InvalidCaseError,
    Request,
    TolerancePolicy,
    Verdict,
)

POLICY = TolerancePolicy()


@pytest.fixture(scope="module")
def sha(weights: WeightDict) -> str:
    return weights_sha256(weights)


def _case(config: ModelConfig, sha: str, requests, mode, seed: int = 1) -> Case:
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=sha,
        requests=requests,
        execution_mode=mode,
        input_seed=seed,
    )


def _tokens(rng: np.random.Generator, config: ModelConfig, n: int) -> tuple[int, ...]:
    return tuple(int(t) for t in rng.integers(1, config.vocab_size, size=n))


# --- known-good controls -------------------------------------------------------------------


@pytest.mark.parametrize("length,prefix", [(1, 1), (5, 1), (8, 3), (12, 11), (16, 8)])
def test_cached_decode_matches_full_prefix(
    config: ModelConfig, weights: WeightDict, sha: str, length: int, prefix: int
) -> None:
    """The central control: incremental cached decoding equals a full-prefix forward pass."""
    rng = np.random.default_rng(100 + length * 7 + prefix)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, length), prefix_length=prefix)],
        ExecutionMode.CACHED_DECODE,
    )
    reference, candidate = build_adapter_pair(config, weights)
    result = compare(reference.run(case), candidate.run(case), POLICY)
    assert result.verdict is Verdict.PASS, result.detail
    assert result.max_abs_err < POLICY.atol


def test_identical_stateless_implementations_agree_bitwise(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    rng = np.random.default_rng(11)
    case = _case(
        config,
        sha,
        [
            Request("a", _tokens(rng, config, 6), prefix_length=6, pad_left=3),
            Request("b", _tokens(rng, config, 9), prefix_length=9),
        ],
        ExecutionMode.STATELESS_BATCH,
    )
    reference, candidate = build_adapter_pair(config, weights)
    result = compare(reference.run(case), candidate.run(case), POLICY)
    assert result.verdict is Verdict.PASS
    assert result.max_abs_err == 0.0


def test_session_requests_are_isolated(config: ModelConfig, weights: WeightDict, sha: str) -> None:
    """Later requests in a session must match requests run entirely alone."""
    rng = np.random.default_rng(22)
    requests = [
        Request("r0", _tokens(rng, config, 5), prefix_length=2),
        Request("r1", _tokens(rng, config, 7), prefix_length=3),
        Request("r2", _tokens(rng, config, 4), prefix_length=1),
    ]
    case = _case(config, sha, requests, ExecutionMode.SESSION)
    reference, candidate = build_adapter_pair(config, weights)
    assert compare(reference.run(case), candidate.run(case), POLICY).verdict is Verdict.PASS

    alone = _case(config, sha, [requests[2]], ExecutionMode.CACHED_DECODE, seed=2)
    solo = candidate.run(alone).outputs["r2"]
    in_session = candidate.run(case).outputs["r2"]
    assert np.abs(solo - in_session).max() < POLICY.atol


def test_benign_sub_tolerance_perturbation_still_passes(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """A control with a real but sub-tolerance difference, so equality is not the only one."""
    rng = np.random.default_rng(33)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 8), prefix_length=3)],
        ExecutionMode.CACHED_DECODE,
    )
    reference = ReferenceAdapter(config, weights)
    perturbed = CandidateAdapter(config, weights, Behavior(perturb_scale=1e-7))
    result = compare(reference.run(case), perturbed.run(case), POLICY)
    assert result.verdict is Verdict.PASS
    assert result.max_abs_err > 0.0, "the perturbation must be real, not a no-op"


def test_padding_transformations_do_not_change_results(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    rng = np.random.default_rng(44)
    tokens = _tokens(rng, config, 7)
    reference = ReferenceAdapter(config, weights)
    unpadded = reference.run(
        _case(config, sha, [Request("r0", tokens, 7)], ExecutionMode.STATELESS_BATCH)
    ).outputs["r0"]
    padded = reference.run(
        _case(config, sha, [Request("r0", tokens, 7, pad_left=9)], ExecutionMode.STATELESS_BATCH)
    ).outputs["r0"]
    assert np.abs(unpadded - padded).max() < POLICY.atol


def test_batch_permutation_does_not_change_results(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    rng = np.random.default_rng(55)
    a = Request("a", _tokens(rng, config, 5), 5, pad_left=3)
    b = Request("b", _tokens(rng, config, 8), 8)
    reference = ReferenceAdapter(config, weights)
    forward = reference.run(_case(config, sha, [a, b], ExecutionMode.STATELESS_BATCH))
    backward = reference.run(_case(config, sha, [b, a], ExecutionMode.STATELESS_BATCH, seed=2))
    for request_id in ("a", "b"):
        assert np.abs(forward.outputs[request_id] - backward.outputs[request_id]).max() == 0.0


# --- output contract -----------------------------------------------------------------------


def test_outputs_cover_exactly_the_valid_positions(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """Padding must never reach an ExecutionResult."""
    rng = np.random.default_rng(66)
    case = _case(
        config,
        sha,
        [
            Request("a", _tokens(rng, config, 3), 3, pad_left=5),
            Request("b", _tokens(rng, config, 11), 11),
        ],
        ExecutionMode.STATELESS_BATCH,
    )
    outputs = ReferenceAdapter(config, weights).run(case).outputs
    assert outputs["a"].shape == (3, config.vocab_size)
    assert outputs["b"].shape == (11, config.vocab_size)
    assert all(array.dtype == np.float32 for array in outputs.values())
    assert all(np.isfinite(array).all() for array in outputs.values())


def test_adapter_ids_distinguish_reference_candidate_and_behavior(
    config: ModelConfig, weights: WeightDict
) -> None:
    reference = ReferenceAdapter(config, weights)
    good = CandidateAdapter(config, weights)
    faulty = CandidateAdapter(config, weights, Behavior(attn_scale="no_sqrt"))

    assert reference.adapter_id != good.adapter_id
    assert good.adapter_id != faulty.adapter_id
    assert "attn_scale=no_sqrt" in faulty.adapter_id
    assert good.describe()["behavior_description"] == "reference (no injected fault)"
    assert "attn_scale" in faulty.describe()["behavior_description"]


# --- state isolation -----------------------------------------------------------------------


def test_repeated_runs_of_one_case_are_identical(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    rng = np.random.default_rng(77)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 9), prefix_length=4)],
        ExecutionMode.CACHED_DECODE,
    )
    candidate = CandidateAdapter(config, weights)
    first = candidate.run(case).outputs["r0"]
    for _ in range(3):
        np.testing.assert_array_equal(first, candidate.run(case).outputs["r0"])


def test_state_does_not_leak_between_separate_cases(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """Running a long session first must not change a later independent case."""
    rng = np.random.default_rng(88)
    target = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 6), prefix_length=2)],
        ExecutionMode.CACHED_DECODE,
    )
    noisy = _case(
        config,
        sha,
        [
            Request("s0", _tokens(rng, config, 12), prefix_length=5),
            Request("s1", _tokens(rng, config, 10), prefix_length=1),
        ],
        ExecutionMode.SESSION,
        seed=99,
    )
    candidate = CandidateAdapter(config, weights)
    clean = candidate.run(target).outputs["r0"]
    candidate.run(noisy)
    after = candidate.run(target).outputs["r0"]
    np.testing.assert_array_equal(clean, after)


def test_a_faulty_adapter_that_leaks_state_still_cannot_leak_across_cases(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """Even the no-reset mutant starts each *case* clean; its fault is within-session only."""
    rng = np.random.default_rng(101)
    target = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 5), prefix_length=2)],
        ExecutionMode.CACHED_DECODE,
    )
    noisy = _case(
        config,
        sha,
        [Request("s0", _tokens(rng, config, 9), prefix_length=3)],
        ExecutionMode.CACHED_DECODE,
        seed=7,
    )
    leaky = CandidateAdapter(config, weights, Behavior(reset="none"))
    clean = leaky.run(target).outputs["r0"]
    leaky.run(noisy)
    np.testing.assert_array_equal(clean, leaky.run(target).outputs["r0"])


# --- invalid inputs --------------------------------------------------------------------------


def test_adapters_reject_a_weights_hash_mismatch(config: ModelConfig, weights: WeightDict) -> None:
    case = _case(config, "b" * 64, [Request("r0", (1, 2), 2)], ExecutionMode.STATELESS_BATCH)
    with pytest.raises(InvalidCaseError, match="weights"):
        ReferenceAdapter(config, weights).run(case)


def test_adapters_reject_an_all_padding_request(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    case = _case(
        config,
        sha,
        [Request("r0", (), prefix_length=1, pad_left=4)],
        ExecutionMode.STATELESS_BATCH,
    )
    with pytest.raises(InvalidCaseError, match="no valid tokens"):
        CandidateAdapter(config, weights).run(case)


def test_adapters_reject_an_out_of_vocabulary_token(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    case = _case(
        config,
        sha,
        [Request("r0", (1, config.vocab_size + 5), 2)],
        ExecutionMode.STATELESS_BATCH,
    )
    with pytest.raises(InvalidCaseError, match="outside vocabulary"):
        ReferenceAdapter(config, weights).run(case)


# --- capture ---------------------------------------------------------------------------------


def test_capture_is_off_by_default(config: ModelConfig, weights: WeightDict, sha: str) -> None:
    case = _case(config, sha, [Request("r0", (1, 2, 3), 2)], ExecutionMode.CACHED_DECODE)
    result = CandidateAdapter(config, weights).run(case)
    assert result.checkpoints == ()
    assert result.capture_enabled is False


def test_capture_produces_aligned_semantic_addresses(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """The reference fires hooks once per sequence; the candidate once per decode step.

    Both must nonetheless produce the same set of checkpoint addresses.
    """
    rng = np.random.default_rng(202)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 6), prefix_length=2)],
        ExecutionMode.CACHED_DECODE,
    )
    reference, candidate = build_adapter_pair(config, weights)
    reference_result = reference.run(case, capture=True)
    candidate_result = candidate.run(case, capture=True)

    assert reference_result.meta["model_calls"] == 1
    assert candidate_result.meta["model_calls"] == 5  # prefill + 4 decode steps

    reference_keys = set(reference_result.checkpoint_index())
    candidate_keys = set(candidate_result.checkpoint_index())
    assert reference_keys == candidate_keys
    # 6 positions x (1 embedding + 2 blocks x 3 kinds + final_norm + logits)
    assert len(reference_keys) == 6 * (1 + 2 * 3 + 2)

    kinds = {key[3] for key in reference_keys}
    assert kinds == {k.value for k in CheckpointKind}
    layers = {key[1] for key in reference_keys}
    assert layers == {"embed", "block0", "block1", "final"}


def test_capture_values_match_between_correct_implementations(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    rng = np.random.default_rng(303)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 5), prefix_length=2)],
        ExecutionMode.CACHED_DECODE,
    )
    reference, candidate = build_adapter_pair(config, weights)
    reference_index = reference.run(case, capture=True).checkpoint_index()
    candidate_index = candidate.run(case, capture=True).checkpoint_index()
    for key, reference_checkpoint in reference_index.items():
        assert reference_checkpoint.values is not None
        assert candidate_index[key].values is not None
        assert np.abs(reference_checkpoint.values - candidate_index[key].values).max() < 1e-4


def test_capture_budget_truncates_rather_than_growing_without_bound(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    case = _case(
        config,
        sha,
        [Request("r0", (1, 2, 3, 4, 5, 6), prefix_length=2)],
        ExecutionMode.CACHED_DECODE,
    )
    adapter = CandidateAdapter(config, weights, capture_budget=CaptureBudget(max_checkpoints=7))
    result = adapter.run(case, capture=True)
    assert len(result.checkpoints) == 7
    assert result.meta["capture_truncated"] is True


def test_capture_drops_values_over_the_element_budget_but_keeps_summaries(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    case = _case(config, sha, [Request("r0", (1, 2), prefix_length=1)], ExecutionMode.CACHED_DECODE)
    adapter = CandidateAdapter(
        config, weights, capture_budget=CaptureBudget(max_values_per_checkpoint=70)
    )
    index = adapter.run(case, capture=True).checkpoint_index()

    hidden = [cp for key, cp in index.items() if key[3] == "block_out"]
    logits = [cp for key, cp in index.items() if key[3] == "logits"]
    assert all(cp.values is not None for cp in hidden)  # d_model 64 fits
    assert all(cp.values is None for cp in logits)  # vocab 97 does not
    assert all(cp.summary["count"] == config.vocab_size for cp in logits)


def test_capture_does_not_leak_between_runs(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    case = _case(
        config, sha, [Request("r0", (1, 2, 3), prefix_length=1)], ExecutionMode.CACHED_DECODE
    )
    adapter = CandidateAdapter(config, weights)
    first = adapter.run(case, capture=True)
    second = adapter.run(case, capture=True)
    assert len(first.checkpoints) == len(second.checkpoints)
    assert adapter.run(case, capture=False).checkpoints == ()


def test_capture_does_not_change_the_outputs(
    config: ModelConfig, weights: WeightDict, sha: str
) -> None:
    """Diagnostic capture must be observation only."""
    rng = np.random.default_rng(404)
    case = _case(
        config,
        sha,
        [Request("r0", _tokens(rng, config, 7), prefix_length=3)],
        ExecutionMode.CACHED_DECODE,
    )
    adapter = CandidateAdapter(config, weights)
    np.testing.assert_array_equal(
        adapter.run(case, capture=False).outputs["r0"],
        adapter.run(case, capture=True).outputs["r0"],
    )
