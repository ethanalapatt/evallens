# EvalLens

A local differential debugger for neural-network inference.

You replace a correct inference path with an optimized one — a KV cache, a batched kernel, a
fused attention. Nothing crashes. The outputs are quietly different, and you find out days
later when a quality metric slips. EvalLens is built for that situation: it compares a
reference and a candidate implementation on identical weights and inputs, tells you whether
the difference is stable, shows you the earliest checkpoint where it becomes visible,
shrinks the failing input to something you can read, and exports a reproduction that runs on
its own.

> **Results not yet measured.** No benchmark has been run. Detection rates, false-positive
> rates, reduction ratios, and timings will appear only when `bench/report.py` generates them
> from raw run records. See `PROGRESS.md` for the honest current state.

> **Every fault in this project's corpus is one this project injected on purpose.** They are
> deliberate test material for the debugger, not discovered bugs in PyTorch or any other
> library.

## Status

M1 and M2 are complete: the reference fixture and its independent FP64 oracle, the case
schema, the reference and candidate adapters, the numerical comparator, and stability replay
all work and are tested on the target machine. Every known-good control passes — incremental
cached decode matches a full-prefix forward pass at `max|Δ| = 2.4e-07`, and identical
stateless implementations agree bitwise. M3 (the fault corpus and input generators) is in
progress. `PROGRESS.md` tracks each milestone with the actual commands and their output.

## Install and check

Requires Python 3.11+ on CPU. No GPU, no downloads, no services.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/evallens doctor
```

`doctor` prints the real environment and then rebuilds the unit fixture and checks it against
an independent float64 NumPy recomputation of the same architecture. On the development
machine (Apple M3, macOS 15.6, Python 3.13.7, PyTorch 2.14.0, NumPy 2.5.3) it reports
`max |Δ| = 2.388e-07` against a `2e-4` bound.

## Run the tests

```bash
.venv/bin/python -m pytest -m 'not mps and not download'
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src/evallens
```

199 tests pass at M2.

## How it works

**The fixture** is a small pre-normalized decoder-only transformer: 2 layers, width 64, 4
heads, vocabulary 97, 120,801 parameters. Its weights are random and deterministic, built
once as a canonical NumPy state dictionary and copied into every implementation with a
content hash. Sharing a random seed between two implementations is not enough — if their
initialization paths differ at all, the weights silently diverge and every comparison
downstream becomes meaningless.

The fixture is untrained. It demonstrates execution correctness, not language capability.

**The oracle** recomputes the same forward pass in float64 and shares no code with the
implementation it checks — it builds its attention mask with explicit index loops rather than
broadcast comparisons, splits QKV by hand, and evaluates GELU from the `erf` definition. A
shared masking helper would make the oracle agree with the implementation precisely on the
class of bug the oracle exists to catch.

**The conventions that make comparison meaningful** live in one place, `adapters/encoding.py`:

- Logical positions run `0 .. n-1` by valid-token index, so left padding shifts a token's
  column but never its position ID. That turns padding invariance into a property you can
  test instead of a coincidence.
- Padding is never a token value. It is `pad_left` plus the batch width, and the reserved
  padding id is rejected inside `token_ids`, so a padding column is always distinguishable
  from a real token.
- Cached execution is teacher-forced: both implementations receive the same canonical tokens
  at every step. Feeding each model its own argmax would let them diverge onto different
  inputs, after which comparing activations measures nothing.

**Verdicts stay disjoint.** `PASS`, `FAIL`, `INVALID`, `ERROR`, `TIMEOUT`, `RESOURCE_LIMIT`,
`UNSTABLE`. A crash is never counted as a detected regression. A classification that changes
on replay is `UNSTABLE` and stays visible in the report.

## What is and is not supported

Supported: the native decoder-only fixture, CPU float32, stateless batched execution,
incremental cached decode, and short sequential sessions.

Not supported and not claimed: arbitrary `torch.compile` graph debugging, quantized-runtime
equivalence, automatic source repair, distributed execution, or production serving. MPS and
pretrained-model adapters are extensions; the primary result does not depend on either.

## Documents

- [`SPEC.md`](SPEC.md) — the resolved technical contract, and why each design choice was made
- [`PROGRESS.md`](PROGRESS.md) — actual milestone state, commands, output, and limitations
- [`CLAUDE.md`](CLAUDE.md) — working conventions and the project's non-negotiables
