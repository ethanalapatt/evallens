# PROGRESS

Actual state of the build. Status values are `not started`, `in progress`, `blocked`,
`complete`, used literally. A file existing is not a completion gate.

| Milestone | Status |
|---|---|
| M1 — repository, environment, reference fixture | **complete** |
| M2 — adapters and numerical comparator | **complete** |
| M3 — fault corpus and generators | **in progress** |
| M4 — checkpoint alignment and localization | not started |
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

## M3 — fault corpus and generators — in progress

### Next exact action

Build `bench/mutants/` (the sixteen declared variants, each with an independently written
trigger fixture that demonstrates its intended fault), `bench/controls/` (the known-good
comparisons), and `src/evallens/generate.py` with both generators — uniform-valid and
boundary-aware — over the same declared valid-case space, plus disjoint calibration,
development, and evaluation seed manifests.

M3 acceptance gate: every included fault is independently qualified by a valid trigger;
generated cases are valid and repeatable; and search policies cannot access fault labels.
Neither generator is required to detect every fault under budget.
