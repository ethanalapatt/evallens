# PROGRESS

Actual state of the build. Status values are `not started`, `in progress`, `blocked`,
`complete`, used literally. A file existing is not a completion gate.

| Milestone | Status |
|---|---|
| M1 — repository, environment, reference fixture | **complete** |
| M2 — adapters and numerical comparator | **complete** |
| M3 — fault corpus and generators | **complete** |
| M4 — checkpoint alignment and localization | **in progress** |
| M5 — input reduction and replay | not started |
| M6 — portable reproduction and demo | not started |
| M7 — benchmark and generated report | not started |
| M8 — reviewer polish and ownership | not started |

**Measured results: none yet.** No benchmark has been run. `RESULTS.md` does not exist and
must not exist until `bench/report.py` generates it from raw records.

**Remote:** `origin` → `https://github.com/ethanalapatt/evallens` (public).

---

## M1 — repository, environment, and reference fixture — complete (2026-09-11)

### What was built

- `src/evallens/types.py` — the resolved case/result/checkpoint/verdict schema, the `Adapter`
  protocol, `TolerancePolicy`, `FailureSignature`, and the lexicographic `CaseSize`.
- `src/evallens/fixtures/config.py` — `ModelConfig`, the canonical NumPy weight initializer,
  and content hashing over names/shapes/dtypes/bytes.
- `src/evallens/fixtures/behavior.py` — the injected-fault switchboard, all defaults correct.
- `src/evallens/fixtures/tiny_transformer.py` — the pre-norm decoder-only fixture with a KV
  cache and an explicit checkpoint `recorder` callback.
- `src/evallens/fixtures/numpy_oracle.py` — an independent FP64 recomputation sharing no
  helper with the implementation it checks.
- `src/evallens/adapters/encoding.py` — strict case validation plus the canonical
  case-to-tensor encoding (position-ID and padding conventions).
- `src/evallens/env.py`, `src/evallens/resources.py` — environment manifest, thread pinning,
  RSS guard, time budgets.
- `src/evallens/cli.py` — `evallens doctor`.
- `SPEC.md`, `CLAUDE.md`, `configs/cpu.toml`, `requirements-lock.txt`, CI workflow.

### Commands actually run, and their output

```
$ .venv/bin/python -m pytest -q
122 passed in 2.39s

$ .venv/bin/python -m ruff check .
All checks passed!

$ .venv/bin/python -m ruff format --check .
25 files already formatted

$ .venv/bin/python -m mypy src/evallens
Success: no issues found in 12 source files

$ .venv/bin/evallens doctor
EvalLens 0.1.0
  python            3.13.7
  torch / numpy     2.14.0 / 2.5.3
  platform          macOS-15.6-arm64-arm-64bit-Mach-O
  chip / cpus       Apple M3 / 8
  physical memory   16.0 GiB
  torch threads     4 (interop 1)
  default dtype     torch.float32
  mps / cuda        available=True / available=False
  process rss       206.0 MiB

  fixture           tiny-2L-64d-4h-89a0eb72a852
  parameters        120,801
  weights sha256    e954a43b36aaeb31595503cc0c0b1aec...

  self-test         [PASS] fixture vs FP64 NumPy oracle
                    max |Δ| = 2.388e-07  (tolerance 2.0e-04)
```

Environment is this MacBook Air M3 (16 GB, macOS 15.6). No separate-machine caveat applies.

### Memory pilot

`evallens doctor` reports ~206 MiB process RSS with torch imported and the unit fixture
built. The 4 GiB design ceiling has ample headroom at this fixture size; `ResourceGuard`
enforces it and is unit-tested with an impossible ceiling so the check is provably live.

### Acceptance gate

| Gate | Evidence |
|---|---|
| Installation and CPU tests work on the actual environment | 122 tests pass in the project venv |
| Weights and seeds reproduce | `test_weights_are_deterministic_across_calls`, order-independent hashing, single-element change detection |
| Independent attention/mask checks pass | `tests/unit/test_oracle_agreement.py` — 9 tests, `max |Δ| = 2.39e-07` |
| Causal invariance passes | `test_causal_prefix_invariance` (4 params) and `test_causal_prefix_invariance_holds_for_every_split` |
| No GPU required | CPU-only throughout; MPS visible but unused |

### Evidence paths

- `tests/unit/test_oracle_agreement.py` — fixture vs independent FP64 oracle
- `tests/unit/test_fixture_invariances.py` — causal prefix, padding, batch permutation, row
  isolation, repeat stability
- `tests/unit/test_encoding.py` — 30 validation and encoding tests
- `tests/property/test_case_properties.py` — 10 Hypothesis properties
- `requirements-lock.txt` — the environment these numbers came from

### Teach-back

**What was built.** A tiny, deliberately transparent decoder-only transformer; one canonical
hashed weight dictionary; an independent FP64 NumPy recomputation of the same math; a strict
case schema with explicit padding and position conventions; and a `doctor` command that
proves all of it on the machine in front of you.

**Why this design.** Two choices carry most of the weight. First, attention is written as
explicit matmuls rather than `F.scaled_dot_product_attention` — a fused kernel hides exactly
the masking and scaling decisions the project exists to probe, and would make several
planned faults impossible to express as shape-valid silent changes. Second, checkpoints are
published through a `recorder` callback instead of `register_forward_hook`: hooks fire in
call order, and call order is not an alignment rule when a full-prefix reference fires once
per sequence and a cached candidate fires once per decode step.

**One tricky failure.** The property test `canonicalizing token values never increases size`
failed on a one-token case: rewriting token 1 to token 2 *raised* `token_value_complexity`,
because the ordering ranks by distinct-value count and then by value sum. The test premise
was wrong, not the ordering — "simplify toward something simpler" is not a converging
operation. The fix was to make the target explicit: `CANONICAL_TOKEN_ID = 1`, the smallest
legal non-padding id, so token simplification can only move a case down the order. That
matters directly for M5, where a non-converging simplification step would let `ddmin` cycle
forever inside its budget.

**How it was tested.** Agreement against an oracle that shares no code with the
implementation; five structural invariances stated as properties of the intended semantics;
30 negative validation tests (the most important being that an all-padding request is
rejected rather than silently accepted); and Hypothesis properties over the declared valid
case space.

### Limitations at M1

- Agreement with the oracle covers the shapes exercised (batch 1–3, length 1–16). It says
  nothing about untested shapes, dtypes, or devices.
- The fixture is untrained. Its outputs demonstrate execution correctness and nothing about
  language capability.
- Repeat-run bitwise stability is observed with pinned threads on this machine. It is not a
  general claim about CPU float32 reproducibility.
- No adapters, comparator, mutants, generators, reducer, export, or benchmark exist yet.

---

## M2 — adapters and numerical comparator — complete (2026-09-11)

### What was built

- `src/evallens/adapters/native.py` — `ReferenceAdapter` (always correct, always full-prefix;
  has no `Behavior` parameter and cannot be made faulty) and `CandidateAdapter` (the optimized
  path: prefill plus per-token incremental decode against a KV cache). Plus bounded
  checkpoint capture via `CaptureBudget`.
- `src/evallens/compare.py` — the tolerance policy, `TensorDiff` evidence, explicit nonfinite
  rules, the zero-norm denominator rule, and output-contract checking.
- `src/evallens/replay.py` — clean-state execution with disjoint verdict routing, and
  `stable_comparison`, which requires an unchanging verdict across three replays.

### Commands actually run, and their output

```
$ .venv/bin/python -m pytest -q
199 passed in 2.89s

$ .venv/bin/python -m ruff check .
All checks passed!

$ .venv/bin/python -m mypy src/evallens
Success: no issues found in 15 source files
```

### Known-good controls (measured)

| Control | Verdict | max abs error |
|---|---|---|
| Incremental cached decode vs full prefix | PASS | 2.384e-07 |
| Identical stateless implementations | PASS | 0.0 (bitwise) |
| Session with per-request cache reset | PASS | 2.533e-07 |
| Benign sub-tolerance perturbation (`perturb_scale=1e-7`) | PASS | > 0, below the band |

Policy in force: `atol=1e-5`, `rtol=1e-4`. No tolerance was changed to make any of these pass.

### Pre-M3 observation: which modes each fault family can reach

Not a benchmark metric — a single-seed manual smoke check, to be replaced in M3 by the
independently constructed trigger fixtures that formally qualify each variant. `pass` here
means the fault is semantically unreachable in that mode, which is correct behavior.

| Family | stateless | cached | session |
|---|---|---|---|
| 1 causal mask (both variants) | fail | fail | fail |
| 2 padding mask (both variants) | fail | pass (no padding) | pass (no padding) |
| 3 decode position (both variants) | pass (no decode) | fail | fail |
| 4 cache indexing (both variants) | pass (no cache) | fail | fail |
| 5 request reset (both variants) | pass | pass (single request) | fail |
| 6 normalization (both variants) | fail | fail | fail |
| 7 attention scaling (both variants) | fail | fail | fail |
| 8 batch indexing (both variants) | fail | pass (batch 1) | pass (batch 1) |

All sixteen declared variants produce a stable, shape-valid, silent mismatch in at least one
mode. That mode-sensitivity is what makes the generators' job non-trivial: a generator that
only ever emits single-request unpadded cases cannot reach families 2, 5, or 8 at all.

### Acceptance gate

| Gate | Evidence |
|---|---|
| Native cached/reference comparisons pass | `test_cached_decode_matches_full_prefix` (5 shapes) and the Hypothesis property over the whole valid space |
| Malformed inputs are INVALID | `test_adapters_reject_*`, `test_an_invalid_case_is_invalid_not_a_failure` |
| Finite/nonfinite handling is explicit | 6 tests in `test_compare.py`, including matching NaNs as agreement |
| Repeated cases do not leak state | `test_state_does_not_leak_between_separate_cases`, and the same for the no-reset mutant |
| Tolerance tests distinguish within-policy from stable violations | band-edge tests, rtol scaling, benign-perturbation control |

### Evidence paths

- `tests/integration/test_native_adapters.py` — 24 tests: controls, output contract, state
  isolation, invalid inputs, capture
- `tests/unit/test_compare.py` — 25 tests on the numerical policy
- `tests/unit/test_replay.py` — 19 tests on verdict routing and stability
- `tests/property/test_adapter_properties.py` — 7 Hypothesis properties over real model runs

### Teach-back

**What was built.** Two adapters that run the same weights on the same inputs by two
different routes, a comparator that turns their disagreement into one explicit verdict, and a
replay layer that refuses to call anything a failure until it has reproduced three times from
clean state.

**Why this design.** The reference deliberately has no fault parameter at all. For cached and
session cases it executes each request independently, unpadded, at batch size one — so
request isolation is a property of how the reference is *constructed*, not something assumed
about it. A candidate that leaks state between requests then has something trustworthy to
fail against. The comparator's nonfinite rules are spelled out rather than left to emerge
from arithmetic, because `NaN != NaN` would otherwise make two matching NaNs look like a
discrepancy while `abs(nan - nan) > band` would make a genuine NaN regression invisible.

**One tricky failure.** The first full run showed the known-good cached control *failing* at
`max|Δ| = 1.08e-01` — the control, not a mutant. Every "cached" column in the fault matrix
showed the same 1.1e-01, which was the giveaway: that was not sixteen faults being detected,
it was one baseline bug being reported sixteen times. The cause was in `KVCache.length`,
which returned layer 0's cached length. A forward pass walks layers in order, so by the time
layer 1 asked for "the" cache length, layer 0 had already appended the current step — layer 1
saw `L + T` instead of `L` and shifted every query's absolute position by one step. The fix
was `length_of(layer)`, read per layer before that layer appends. Two things made this
catchable: the control existing at all, and the failure magnitude being identical across
unrelated mutants.

**How it was tested.** Controls first, faults second — deliberately in that order. The
strongest control is a Hypothesis property rather than a list of shapes: *for any valid case
in the declared space*, the known-good candidate agrees with the reference. A control that
only held on hand-picked lengths would be worthless as a detection baseline.

### Limitations at M2

- Checkpoint capture produces correctly aligned addresses and matching values between correct
  implementations, but the localization logic that consumes them (earliest observed
  divergence) is M4.
- `TIMEOUT` is detected after a call returns, not enforced preemptively — execution is
  in-process. The verdict means "this case exceeded its budget", not "this case was killed at
  its budget". Hard limits come with the fresh-subprocess replay in M5/M6.
- The fault-mode table above is a single-seed smoke observation, not measured evidence. M3
  replaces it with independently constructed trigger fixtures.
- No generators, reducer, export, or benchmark yet.

---

## M3 — fault corpus and generators — complete (2026-09-11)

### What was built

- `bench/mutants/` — 8 families x 2 variants = 16 declared injected faults, each with a
  **handwritten** trigger fixture and a recorded rationale for why that shape exposes the
  mechanism. `qualify_mutant` decides inclusion; `qualify_all` reports every variant.
- `bench/controls/` — 7 known-good comparisons covering identical implementations, correct
  cached vs full-prefix execution, the prefill-only boundary, padded alignment, batch
  permutation, fresh-request isolation, and one benign sub-tolerance perturbation.
- `bench/seeds.py` — disjoint calibration / development / evaluation seed sets, with
  disjointness asserted at import time.
- `src/evallens/generate.py` — `UniformValidGenerator` (the named baseline) and
  `BoundaryAwareGenerator`, over the same declared space, same capability, same budget, plus
  six declared case categories and `classify_case`.

### Commands actually run, and their output

```
$ .venv/bin/python -m pytest -q
326 passed in 4.28s

$ .venv/bin/python -m ruff check .
All checks passed!

$ .venv/bin/python -m ruff format --check .
45 files already formatted

$ .venv/bin/python -m mypy src/evallens
Success: no issues found in 16 source files

$ .venv/bin/python -m mypy bench
Success: no issues found in 4 source files
```

### Qualification: 16 of 16 declared variants qualified

Each variant produced a **stable FAIL** across 3 replays against its own handwritten trigger,
under the frozen policy `atol=1e-5, rtol=1e-4`. No tolerance was adjusted to obtain any of
these. Max absolute error on the trigger case:

| Family | Variant | max abs error |
|---|---|---|
| 1 causal mask | `off_by_one` | 2.772e-01 |
| 1 causal mask | `leak_last` | 3.222e-01 |
| 2 padding mask | `ignore` | 4.493e-01 |
| 2 padding mask | `right_only` | 4.739e-01 |
| 3 decode position | `minus_one` | 3.274e-01 |
| 3 decode position | `restart` | 2.813e-01 |
| 4 cache indexing | `write_overwrite_last` | 2.763e-01 |
| 4 cache indexing | `read_drop_oldest` | 1.955e-01 |
| 5 request reset | `none` | 4.008e-01 |
| 5 request reset | `partial` | 1.964e-01 |
| 6 normalization | `large_eps` | 6.332e-01 |
| 6 normalization | `wrong_axis` | 7.659e-01 |
| 7 attention scaling | `no_sqrt` | 7.668e-03 |
| 7 attention scaling | `d_model` | 3.690e-03 |
| 8 batch indexing | `row_leak_first` | 2.942e-01 |
| 8 batch indexing | `v_roll` | 5.377e-01 |

None of these is a detection rate. They are qualification checks on handwritten triggers: the
variants do what they claim. Whether either *generator* can find them under budget is M7.

The two attention-scaling variants are two orders of magnitude subtler than the rest, which
is expected — a scale change perturbs every softmax slightly rather than corrupting a
specific position — and makes them the most interesting variants for the detection study.

### Controls: 7 of 7 are stable passes

All under the same frozen policy, so the false-positive denominator is real.

### Acceptance gate

| Gate | Evidence |
|---|---|
| Every included fault is independently qualified by a valid trigger | 16 parametrized qualification tests, plus tests that each trigger case is itself valid and that the reference never fails one |
| Generated cases are valid and repeatable | 4 unit tests x 2 generators over 120 cases each, plus 5 Hypothesis properties over arbitrary seeds and budgets |
| Search policies cannot access fault labels | `tests/unit/test_search_policy_blindness.py` — AST scan for forbidden imports and names, a source scan for mutant identity strings, and a clean-subprocess check that importing the generator loads no `bench` module |

### Evidence paths

- `tests/integration/test_fault_corpus.py` — 58 tests: corpus shape, qualification, controls, seeds
- `tests/unit/test_generators.py` — 35 tests: validity, determinism, coverage, diversity
- `tests/unit/test_search_policy_blindness.py` — 7 blindness tests
- `tests/property/test_generator_properties.py` — 9 Hypothesis properties

### Teach-back

**What was built.** Sixteen deliberate faults, each paired with a handwritten case proving it
does what it claims; seven known-good comparisons that a correct implementation must pass;
and two generators that explore the same valid space by different strategies under identical
budgets.

**Why this design.** The generators share a *category schedule* and differ only in their
choices within it. That is the fair comparison: if the uniform baseline could not emit padded
batches or sessions at all, it could never reach fault families 2, 5, or 8, and the
boundary-aware generator's higher score would measure coverage rather than judgment. The
blindness tests are structural rather than a convention in a docstring, because a search
policy that can read the answer key is not being measured — it is being told. They check
three independent ways: no forbidden import in the AST, no mutant identity in the source text,
and no `bench` module in `sys.modules` after a clean-subprocess import.

**One tricky failure.** Two failures, both caught by the tests rather than by inspection.
The first: `request_reset.none` came back `INVALID` instead of `FAIL`. Its handwritten trigger
used token 97, and the vocabulary is 97 tokens — valid ids run 0..96. Validation was right and
the trigger was wrong, which is exactly the failure mode the "every trigger case is itself
valid" test now guards: an invalid trigger would have scored as `INVALID` forever and quietly
dropped a whole fault family from the corpus. The second: `BoundaryAwareGenerator` emitted a
prefix length of 2 for a single-token request, because its boundary set `{1, 2, n-1, n}` is
only sensible for `n >= 2`. Clamping every candidate into `[1, n]` fixed it. Both are the same
underlying lesson — boundary values need to be clamped to the structure they describe, not
assumed to fit it.

**How it was tested.** Qualification is a test, not a report: all sixteen variants are
parametrized, so a variant that stops failing breaks the build rather than silently dropping
out of a manifest. Controls are tested with the same `stable_comparison` path the benchmark
uses, so a false positive shows up as a failing test. The `benign_subtolerance_perturbation`
control has a guard asserting the perturbation is genuinely nonzero *and* genuinely below
tolerance, so it can never decay into a no-op that passes for the wrong reason.

### Limitations at M3

- Qualification uses handwritten triggers. It establishes that each variant is a real,
  stable, shape-valid silent fault; it says nothing about whether a generator can find it
  under budget. That is M7, and it is a separate question.
- These are held-out executions of *known* fault families. They are not evidence of
  generalization to unknown real-world bugs, and no report may present them as such.
- Eight families is a narrow corpus. Uncertainty on any aggregate rate will be substantial.
- Localization, reduction, export, and the benchmark harness do not exist yet.

---

## M4 — checkpoint alignment and localization — in progress

### Next exact action

Implement `src/evallens/trace.py`: take a stable output failure, run a separate traced pass on
both adapters, align checkpoints by `(request_id, layer_name, token_position, kind)`, and
compare **all** aligned checkpoints in execution order — no binary search, because
monotonicity cannot be assumed. Report the earliest *observed* divergence, and report
`localization unavailable` when no checkpoints align.

Then add the crafted reconvergence case: an intermediate discrepancy that disappears before
the output, which a first-divergence search assuming monotonicity would mis-handle.

M4 acceptance gate: known injected examples localize to the expected exposed checkpoint;
unavailable alignment is reported accurately; and capture neither leaks across runs nor
silently enters only one timing baseline.
