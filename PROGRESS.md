# PROGRESS

Actual state of the build. Status values are `not started`, `in progress`, `blocked`,
`complete`, used literally. A file existing is not a completion gate.

| Milestone | Status |
|---|---|
| M1 — repository, environment, reference fixture | **complete** |
| M2 — adapters and numerical comparator | **in progress** |
| M3 — fault corpus and generators | not started |
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

## M2 — adapters and numerical comparator — in progress

### Next exact action

Implement `src/evallens/adapters/native.py` with two adapters over the shared encoding:

- `ReferenceAdapter` — correct full-prefix execution. For `stateless_batch` it runs the
  padded batch in one pass; for `cached_decode` and `session` it runs each request
  independently, unpadded, batch 1, so the reference is correct and isolated by construction.
- `CachedAdapter` — prefill plus per-token incremental decode, batch 1, honoring
  `Behavior.decode_pos` and `Behavior.reset`. `Behavior()` makes it the known-good cached
  path; the mutant corpus supplies the faulty variants in M3.

Both strip padding and return `[n_valid, vocab]` float32 per request. Then
`src/evallens/compare.py` (tolerance policy, `TensorDiff`, verdict classification, nonfinite
handling) and `src/evallens/replay.py` (clean-state stability replay, three runs).

M2 acceptance gate: native cached/reference comparisons pass; malformed inputs are `INVALID`;
finite/nonfinite handling is explicit; repeated cases do not leak state; tolerance tests
distinguish within-policy differences from stable violations.
