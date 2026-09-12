# EvalLens technical specification

Resolved from the project brief. This is the contract the implementation is held to; where
the brief left a choice open, the decision and its reason are recorded here.

**Status:** M1–M8 complete; measured results in `RESULTS.md`. Every section now describes
behavior that exists and is tested. Nothing in this document is a claim that unmeasured behavior
has been measured; the one acceptance item never satisfied — a screen recording of the demo — is
recorded as incomplete in `PROGRESS.md` and `docs/RECORDING.md`.

---

## 1. Problem

A developer replaces a correct inference path with an optimized one — a KV cache, a batched
kernel, a fused attention. Nothing crashes. The outputs are subtly, or catastrophically,
different, and the difference shows up as a quality regression days later.

EvalLens is a differential debugger for that situation. Given a reference adapter, a
candidate adapter, and a generator of valid inputs, it:

1. Runs both on identical weights, inputs, and execution conditions.
2. Compares the intended output tensors under an explicit numerical policy.
3. Re-runs candidate failures from clean state to separate stable mismatches from noise.
4. Captures aligned intermediate checkpoints in a separate diagnostic pass and reports the
   **earliest observed divergence**.
5. Minimizes the failing input while preserving its failure signature.
6. Exports a standalone reproduction that runs without EvalLens installed.

### Scope boundaries

Supported: the native decoder-only fixture, CPU float32, stateless batched execution,
incremental cached decode, and short sequential sessions.

Not supported, and not claimed: arbitrary `torch.compile` graph debugging, quantized-runtime
equivalence, automatic source repair, distributed execution, production serving, or model
training. MPS and pretrained-model adapters are extensions; the primary result does not
depend on either.

---

## 2. Platform

CPU execution is the required reference platform. The development, correctness, demo, and
primary benchmark paths all run on a MacBook Air M3 with 16 GB unified memory. CUDA is never
assumed.

Measured environment for M1 (`evallens doctor`, 2026-09-11):

| Field | Value |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.14.0 |
| NumPy | 2.5.3 |
| OS | macOS 15.6 (24G84), arm64 |
| Chip | Apple M3, 8 CPUs |
| Physical memory | 16.0 GiB |
| torch threads | 4 (interop 1) |
| Default dtype | torch.float32 |
| MPS | available (unused by the CPU release) |
| CUDA | unavailable |

`requirements-lock.txt` is a `pip freeze` of the environment these numbers came from.

### Resource budget

A 4 GiB combined worker RSS ceiling with short sequences and one benchmark worker. This is
a **design budget that is enforced**, not a measurement: `ResourceGuard` samples real RSS at
explicit checkpoints and marks a run `RESOURCE_LIMIT` when it is crossed, rather than
letting the machine swap and corrupt every timing in the study.

---

## 3. Case schema (version 1)

### Execution modes

| Mode | Shape | `prefix_length` | Padding | Purpose |
|---|---|---|---|---|
| `stateless_batch` | 1–4 rows, one full-prefix forward | must equal request length | per-row `pad_left` | batching, padding, and masking faults |
| `cached_decode` | exactly 1 request, batch 1, unpadded | prefill boundary | must be 0 | cache indexing and decode-position faults |
| `session` | 1–3 sequential requests, each like `cached_decode` | per request | must be 0 | between-request cache reset |

### Request

```python
Request(request_id: str, token_ids: tuple[int, ...], prefix_length: int, pad_left: int = 0)
```

`token_ids` holds *valid tokens only*. Padding is never expressed as a token value; it is
expressed as `pad_left` plus the batch width. Token id `0` (`PAD_TOKEN_ID`) is reserved and
rejected inside `token_ids`, so a padding column is always distinguishable from a real token
during validation.

`request_id` is stable across reduction. When the reducer deletes earlier requests from a
session, the target request keeps its id, so a failure signature recorded before reduction
still addresses the same request afterwards.

### Case

```python
Case(
    schema_version,
    case_id,
    model_config_id,
    weights_sha256,
    requests,
    execution_mode,
    input_seed,
    category,
    provenance,
)
```

`case_id` is derived from the case's own canonical content hash. `provenance` is excluded
from that hash on purpose: it is bookkeeping about where a case came from, and two cases
that execute identically must share a predicate-cache key.

A case names the fixture configuration and weights it is valid against *by hash*, so
replaying it in another process or checkout either reproduces the same execution or fails
loudly on a mismatch.

### Position-ID convention

A request's logical positions are `0 .. n_valid - 1`, assigned by **valid-token index**,
never by column index in the padded tensor. Left padding moves a token's column but not its
position ID. This is the decision that makes padding invariance a testable property rather
than a coincidence of implementation.

### Padded batch layout

Row `i` is `[PAD] * pad_left_i + tokens_i + [PAD] * rest`; batch width is
`max_i(pad_left_i + n_i)`. `key_valid` is True exactly on valid columns, so a padding column
is never a legal attention key and never reaches an `ExecutionResult`.

### Validation

`validate_case` rejects: unknown schema versions, model-config or weights-hash mismatches,
empty request lists, duplicate or empty request ids, requests with zero valid tokens (an
all-padding row is never a legal input), the reserved padding id inside `token_ids`,
out-of-vocabulary and non-integer tokens (`bool` included — it is an `int` subclass and
would otherwise become token 1), negative padding, prefix lengths outside `[1, n_valid]`,
stateless prefixes that do not cover the whole request, padding on cached execution, and
batch/session/context-limit overruns.

### Output contract

`ExecutionResult.outputs[request_id]` is float32 `[n_valid_tokens, vocab_size]`. Row `p` is
the next-token logits produced after consuming logical token `p`. Padding is stripped by the
adapter, so comparison code cannot accidentally treat a padded position as meaningful.

All tensors crossing the adapter boundary are NumPy arrays, never torch tensors, so
comparison, hashing, and serialization never depend on a framework version.

### Teacher forcing

Cached execution feeds the case's own canonical tokens at every step, never a model
prediction. Feeding each implementation its own argmax would let the two runs diverge onto
*different inputs*, after which comparing their activations would measure nothing.

---

## 4. Fixture

Pre-normalized decoder-only transformer: token + learned position embeddings, causal
self-attention, residuals, GELU MLP, final norm, vocabulary projection.

**Unit fixture** (`tiny-2L-64d-4h`): 2 layers, `d_model` 64, 4 heads, `d_ff` 256, vocab 97,
context 128, **untied** embeddings, `eps` 1e-5. 120,801 parameters. Used by the correctness
suite, the demo, and the primary benchmark.

**Scale fixture** (`scale-6L-384d-6h`): ~11M parameters, optional, never used by the primary
correctness suite.

Weights are random and deterministic. Training is unnecessary for execution-correctness
testing, and these fixtures demonstrate nothing about language capability.

### Canonical weights

One canonical NumPy state dictionary is built by `make_weights` and copied into every
implementation, with a content hash over names, shapes, dtypes, and raw bytes. Sharing a
*seed* between two implementations is not sufficient: if their initialization paths differ at
all, the weights silently diverge and every downstream comparison becomes meaningless.

### Design decisions and their reasons

**Explicit matmuls, not `F.scaled_dot_product_attention`.** A fused kernel hides exactly the
masking and scaling decisions this project probes, and makes several injected faults
impossible to express as shape-valid silent changes.

**A `recorder` callback, not `register_forward_hook`.** Hooks fire in call order, and call
order is not an alignment rule: a full-prefix reference call fires a hook once for a whole
sequence while a cached candidate fires it once per decode step. The callback hands the
adapter a raw `[B, T, D]` tensor; only the adapter knows how to map `(row, column)` to a
semantic `(request_id, logical position)` address, so only the adapter does that mapping.
*(This deviates from the brief's suggestion of hook APIs; the reason is recorded here.)*

**Causality over columns, padding over `key_valid`.** A request's valid tokens occupy a
contiguous ascending column range, so column causality and logical-position causality agree.
Keeping the two concerns separate is what lets the causal-mask fault and the padding-mask
fault be injected independently.

**Fully masked query rows produce zeros, not NaN.** An all-padding row would otherwise
softmax over `-inf` everywhere. Such rows are never compared, but keeping them finite means
a genuine nonfinite output at a valid position stays an unambiguous signal.

---

## 5. Reference correctness

### Independent oracle

`fixtures/numpy_oracle.py` recomputes the forward pass in float64. It deliberately shares
**nothing** with the torch fixture: it builds its attention mask with explicit Python loops
over index pairs rather than broadcast comparisons, splits QKV by hand, and evaluates GELU
from the `math.erf` definition. If both files shared a masking helper, the oracle would agree
with the implementation precisely on the class of bug it exists to catch.

Measured agreement (M1): `max |Δ| = 2.39e-07` on the unit fixture at batch 2 × length 6,
against a `2e-4` bound (roughly 1e3 × float32 epsilon for logits of order 1e-1..1e0).

**What this establishes:** the torch fixture computes the intended architecture to float32
rounding, for the shapes actually exercised.
**What it does not establish:** that the architecture is a good language model (the weights
are random), or that agreement extends to shapes, dtypes, or devices never checked.

### Structural invariances (all passing at M1)

| Invariance | Statement |
|---|---|
| Causal prefix | Appending future tokens cannot change logits at earlier valid positions. Checked at every split point of a 9-token sequence. |
| Padding | Left padding shifts columns, not position IDs, so valid logits do not move. Checked standalone and inside a ragged batch. |
| Batch permutation | Stateless rows are independent; permuting rows permutes outputs and nothing else. |
| Row isolation | Changing one stateless row cannot perturb another. |
| Repeat stability | With pinned threads, re-running an identical case reproduces identical bits. |

Causal prefix invariance is the most load-bearing: without it, comparing a full-prefix
reference against an incremental cached candidate would be meaningless.

*(Full-prefix versus incremental execution and fresh-request isolation are M2 gates.)*

---

## 6. Numerical policy

```text
violation = abs(candidate - reference) > atol + rtol * abs(reference)
```

Starting CPU float32 policy: `atol = 1e-5`, `rtol = 1e-4`, `zero_norm_eps = 1e-12`. These are
**policy choices, not mathematical truths**. They may be calibrated on known-good comparisons
only, and are frozen before held-out fault evaluation. Threshold changes and their rationale
are recorded. Tolerances are never loosened to make a faulty candidate pass.

Reported per aligned tensor: max absolute error, relative L2 error, fraction of violating
entries, shapes, dtype, and finite/nonfinite counts. When the reference tensor's L2 norm is
below `zero_norm_eps`, relative error is reported as the absolute L2 difference and flagged,
rather than dividing by a near-zero norm to manufacture a headline number.

`TolerancePolicy.policy_id` participates in every predicate cache key, so a policy change can
never silently reuse a cached verdict.

### Verdicts

`PASS`, `FAIL` (stable numerical/output-contract discrepancy), `INVALID` (input contract
violated), `ERROR` (unexpected exception), `TIMEOUT`, `RESOURCE_LIMIT`, `UNSTABLE`
(classification changed on replay). These are disjoint: a crash is never counted as a
detected regression, and an unstable result is never promoted to a failure.

The primary failure predicate uses intended final output tensors. Intermediate discrepancies
are diagnostic evidence and may reconverge before the output; they do not automatically
become output-regression detections.

---

## 7. Localization

Checkpoints are addressed by `(request_id, layer_name, token_position, kind)` with
`kind ∈ {embedding, attn_out, mlp_out, block_out, final_norm, logits}`. Alignment is by
address, never by hook invocation order: one full-prefix reference call records a whole
sequence while a cached candidate records one decode step at a time, so the *n*-th recorded
tensor on one side has no correspondence to the *n*-th on the other.

**Traversal order:** `(request index, network depth, token position)`. Depth dominates because
it is the causal axis — a difference at block 0 must precede any difference it causes at
block 1. Within a depth, lower positions come first, since position `p` depends only on
positions `≤ p`. This is a *total* order imposed on a partial one, declared and stable rather
than a claim about execution sequence.

**All aligned checkpoints are compared. No binary search.** Bisection needs the "diverged"
predicate to be monotone along the traversal. Measured at M4: 10 of the 16 qualified variants
have at least one checkpoint that returns inside tolerance after an earlier one left it. The
linear pass over a bounded capture is cheap and is the only correct version.

The result is the **earliest observed divergence** — evidence about where divergence becomes
visible at the exposed checkpoints, not proof of root cause, and never an identification of an
operation inside a layer. The caveat is attached to the serialized payload so a viewer cannot
present it as a root cause. If no checkpoints align, or their values were dropped under the
capture budget, the output-level failure is retained and localization is reported as
unavailable — summaries are not element-wise evidence.

Capture is a **separate traced pass**, run only after a stable failure is established, so its
allocation and serialization cost never lands inside one side of a detection timing.

**Granularity limit:** checkpoints are layer-level. A result of `block0/attn_out` narrows a
fault to attention but cannot distinguish a mask bug from a scaling bug within it.

---

## 8. Fault corpus *(M3)*

Eight families, two variants each, in candidate-only code under `bench/mutants/`.

The injected-fault switchboard is `fixtures/behavior.py`. `Behavior()` with no arguments *is*
the reference implementation. Faults are introduced only by the mutant corpus, which builds
non-default `Behavior` values and pairs each with an independently written trigger fixture.

**Why the switchboard lives in `src`:** a fault such as "the causal mask permits one future
position" is a one-line change in the middle of attention. Expressing it as a forked copy of
the model per variant would give sixteen near-duplicate transformers that drift apart, and a
mutant that diverges for accidental reasons is worse than no mutant. One transformer plus one
explicit switch means a variant differs from the reference in exactly the declared way.

What must still hold, and is enforced by test: `evallens.generate` and `evallens.reduce`
never import `bench.mutants` and never read a `Behavior`. They reach models only through the
`Adapter` protocol. Fault labels, trigger fixtures, and expected answers live in the scoring
layer.

| # | Family | Variants |
|---|---|---|
| 1 | Causal mask boundary | `off_by_one`, `leak_last` |
| 2 | Padding-mask application | `ignore`, `right_only` |
| 3 | Decode positional offset | `minus_one`, `restart` |
| 4 | Cached K/V indexing | `write_overwrite_last`, `read_drop_oldest` |
| 5 | Between-request cache reset | `none`, `partial` |
| 6 | Normalization epsilon/axis | `large_eps`, `wrong_axis` |
| 7 | Attention scaling | `no_sqrt`, `d_model` |
| 8 | Batch row indexing | `row_leak_first`, `v_roll` |

Families 3, 4, and 5 live in the serving loop (the cached adapter) rather than the layer
math, because that is where those bugs actually occur.

Every included variant must have an independently constructed valid trigger fixture before
the benchmark manifest is frozen. Non-qualified variants stay visible as incomplete rather
than being quietly deleted.

Known-good controls include identical implementations, correct full-prefix versus cached
execution, correctly aligned padding transformations, batch permutations, fresh-request
isolation, and at least one benign sub-tolerance perturbation (`Behavior.perturb_scale`), so
that exact equality is not the only control exercised.

Calibration, development, and evaluation seed sets are disjoint. These are held-out
executions of *known* fault families, not evidence of generalization to unknown real bugs.

---

## 9. Reduction *(M5)*

Structure-aware `ddmin`, implemented directly, plus a separately implemented greedy
single-deletion baseline.

Operations, in order: remove earlier session requests (retaining the target); remove token
chunks (keeping requests nonempty and prefix/decode boundaries valid); reduce padding and
simplify batch structure for stateless cases; simplify token values toward
`CANONICAL_TOKEN_ID`; repeat until no accepted reduction or the budget is exhausted.

Size is the lexicographic tuple `(n_requests, n_valid_tokens, n_padding_tokens,
token_value_complexity)`, and every accepted change must strictly decrease it. That is what
prevents simplification cycles. Token-value simplification targets one fixed value rather
than "something simpler", for the same reason.

Weights, tolerances, implementation, and architecture are never altered while minimizing an
input.

A reduction is accepted only when the reduced case is valid, reproduces a stable output
failure, and matches the failure signature: failure class, target request, and execution
mode — plus checkpoint identity `(request, layer, kind)` when preserving a localized failure
is the task. Position is reported but not required to match, because deleting tokens
renumbers positions. A matching signature cannot prove identical root cause.

Budgets: 256 logical predicate queries or 60 s, both configurable; stability replays and
final verification are charged to a recorded budget; each baseline gets the same declared
limits; the best valid failing case is preserved on timeout.

After reduction: replay three times and re-run in a fresh subprocess. **1-minimal with
respect to the declared deletion operations** is claimed only when every remaining eligible
single deletion was checked and failed. Otherwise the label is
`reduced, minimality not established`. Never "globally smallest counterexample".

---

## 10. Reproduction export *(M6)*

A self-contained directory: `repro.py`, the minimal model/adapter code, input JSON,
deterministic weights as non-object NPZ, the tolerance policy, hashes, an environment
manifest, and a README with exact commands.

The native-fixture reproduction must not import the installed EvalLens package and must not
reference absolute paths into the original checkout. It is verified from a fresh temporary
directory and subprocess with `PYTHONPATH` cleared.

`--expect-mismatch` exits zero only when the recorded mismatch reproduces. Default mode
returns a documented nonzero code for a mismatch, distinct from setup/execution errors — an
import error is never mistaken for a successful reproduction.

Exported reproductions are outputs of the actual reducer, never handcrafted.

---

## 11. Benchmark and metric integrity *(M7)*

Smoke preset and a full declared CPU preset, run sequentially. A pilot estimates duration;
the finalized workload and budgets are then fixed before outcomes are inspected.

Every run preserves: git commit and clean/dirty state (headline runs require clean source),
hardware/software/thread configuration, fixture/weights/config/tolerance hashes, seeds, the
qualified mutant and control manifest, generator/reducer versions, the complete set of
expected trial IDs with actual outcomes including errors and timeouts, and raw JSONL records.

`bench/report.py` generates `RESULTS.md` from raw records and rejects missing required
trials, duplicate IDs, inconsistent hashes, invalid denominators, and mismatched comparison
cohorts. An interrupted study produces an explicitly incomplete diagnostic report, never a
publishable success table.

**Hard rule: never fabricate metrics.** No estimated speedups, invented detection rates,
copied numbers, or hand-edited result tables. Synthetic timing fixtures are allowed only in
tests, clearly labeled, and blocked from producing a file named `RESULTS.md`. Until real
measurements exist, README says `Results not yet measured`.

Injected faults are deliberate and are labeled as such in the README, reports, viewer, and
any resume material. An external library mismatch is an observation to investigate, never
automatically "a newly discovered framework bug".

---

## 12. Repository layout

| Path | Purpose |
|---|---|
| `src/evallens/types.py` | Case, result, checkpoint, verdict, and adapter contracts |
| `src/evallens/env.py` | Environment manifest capture |
| `src/evallens/resources.py` | Thread pinning, RSS guard, time budgets |
| `src/evallens/fixtures/` | Tiny transformer, canonical weights, behavior switchboard, FP64 oracle |
| `src/evallens/adapters/` | Validation, case-to-tensor encoding, native adapters |
| `src/evallens/cli.py` | CLI entry point |
| `bench/`, `configs/`, `tests/`, `viewer/`, `docs/`, `artifacts/`, `examples/` | As in the brief |

Modules for later milestones (`generate`, `compare`, `trace`, `reduce`, `replay`, `export`)
are created when they are implemented, not scaffolded empty.
