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

M1 through M3 are complete: the reference fixture and its independent FP64 oracle, the case
schema, the reference and candidate adapters, the numerical comparator, stability replay, the
injected-fault corpus, and the two input generators all work and are tested on the target
machine.

All 7 known-good controls are stable passes — incremental cached decode matches a full-prefix
forward pass at `max|Δ| = 2.4e-07`, and identical stateless implementations agree bitwise. All
16 declared fault variants qualify: each produces a stable, shape-valid, silent mismatch
against its own handwritten trigger. Those are qualification checks, not detection rates —
whether a *generator* finds them under budget is a separate question, measured in M7.

M4 (localization), M5 (reduction), and M6 (reproduction export and the demo) are complete too.
M7 (the benchmark and its generated report) is in progress. `PROGRESS.md` tracks each milestone
with the actual commands and their output.

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

501 tests pass at M6.

## See it work

One command runs the whole pipeline against a **deliberately injected** cache fault and writes
real artifacts:

```bash
.venv/bin/evallens demo --out artifacts/demo --config configs/cpu.toml
```

An actual run on the development machine:

```
  [  0.07s] search                 stable failure on case 5 of 64 (cached_decode), max |Δ| = 7.263e-02
  [  0.01s] localize               earliest observed divergence at r0/block0/pos12/attn_out;
                                   16/126 aligned checkpoints diverge (discrepancy reconverges
                                   later; not monotone)
  [  0.06s] reduce                 14 -> 2 valid tokens (7.0x) in 4 predicate queries;
                                   one_minimal_wrt_declared_operations
  [  0.61s] verify_reduced_case    reproduced in a fresh process
  [  0.05s] export                 wrote a self-contained package to repro/
  [  0.59s] verify_export          ran from a fresh temporary directory without importing EvalLens
```

Then run the exported reproduction — it needs Python, PyTorch, and NumPy, and nothing else:

```bash
cd artifacts/demo/repro
python repro.py --expect-mismatch   # exits 0 only if the recorded mismatch reproduces
python repro.py                     # exits 1 because the mismatch is present
```

A committed example lives in [`examples/reproductions/`](examples/reproductions/).

Inspect the run in the offline viewer (loopback only, no backend):

```bash
.venv/bin/evallens view artifacts/demo
```

### Other commands

```bash
evallens compare --case artifacts/demo/failure.json --localize
evallens reduce  --failure artifacts/demo/failure.json --out artifacts/reduced
evallens export  --failure artifacts/reduced/failure.json --out artifacts/repro
```

`compare` exits 0 on PASS, 1 on a stable FAIL, and 2 for anything else — a crash, an invalid
input, or an unstable result never gets counted as a detection.

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

**Localization compares every aligned checkpoint — no binary search.** Bisection needs the
"diverged" predicate to be monotone along the traversal, and measurably it is not: 10 of the
16 injected fault variants have a checkpoint that returns inside tolerance after an earlier
one left it. The result is called the *earliest observed divergence*, which is evidence about
where a difference becomes visible at the exposed checkpoints — not proof of root cause.

**Reduction is signature-preserving and blind.** A transformation is accepted only if the
result is valid, strictly smaller in a lexicographic order, in the same declared category, and
still stably fails at the same target request. The reducer reaches models only through the
adapter protocol; it cannot see which fault it is reducing, and a test enforces that three
different ways.

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
