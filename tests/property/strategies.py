"""Hypothesis strategies for valid cases.

These build cases from the *declared* valid space directly. They are a test-support tool and
are deliberately separate from ``evallens.generate``: a generator that is only ever checked
against strategies written from the same mental model will happily share its blind spots.
"""

from __future__ import annotations

from hypothesis import strategies as st

from evallens.adapters.encoding import MAX_BATCH_ROWS, MAX_SESSION_REQUESTS
from evallens.fixtures.config import UNIT_FIXTURE, ModelConfig
from evallens.types import Case, ExecutionMode, Request

MAX_TOKENS = 24
WEIGHTS_SHA = "f" * 64


def tokens(config: ModelConfig = UNIT_FIXTURE, max_size: int = MAX_TOKENS):
    return st.lists(
        st.integers(min_value=1, max_value=config.vocab_size - 1), min_size=1, max_size=max_size
    )


@st.composite
def stateless_requests(draw, config: ModelConfig = UNIT_FIXTURE, max_rows: int = MAX_BATCH_ROWS):
    count = draw(st.integers(min_value=1, max_value=max_rows))
    rows = []
    for index in range(count):
        values = draw(tokens(config))
        pad_left = draw(st.integers(min_value=0, max_value=4))
        rows.append(
            Request(f"r{index}", tuple(values), prefix_length=len(values), pad_left=pad_left)
        )
    return rows


@st.composite
def cached_request(draw, config: ModelConfig = UNIT_FIXTURE):
    values = draw(tokens(config))
    prefix = draw(st.integers(min_value=1, max_value=len(values)))
    return Request("r0", tuple(values), prefix_length=prefix)


@st.composite
def session_requests(draw, config: ModelConfig = UNIT_FIXTURE):
    count = draw(st.integers(min_value=1, max_value=MAX_SESSION_REQUESTS))
    rows = []
    for index in range(count):
        values = draw(tokens(config, max_size=12))
        prefix = draw(st.integers(min_value=1, max_value=len(values)))
        rows.append(Request(f"r{index}", tuple(values), prefix_length=prefix))
    return rows


@st.composite
def valid_cases(draw, config: ModelConfig = UNIT_FIXTURE):
    mode = draw(st.sampled_from(list(ExecutionMode)))
    if mode is ExecutionMode.STATELESS_BATCH:
        requests = draw(stateless_requests(config))
    elif mode is ExecutionMode.CACHED_DECODE:
        requests = [draw(cached_request(config))]
    else:
        requests = draw(session_requests(config))
    return Case.create(
        model_config_id=config.config_id,
        weights_sha256=WEIGHTS_SHA,
        requests=requests,
        execution_mode=mode,
        input_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        category=draw(st.sampled_from(["uniform", "boundary"])),
    )
