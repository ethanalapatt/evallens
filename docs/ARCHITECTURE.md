# How EvalLens works

A short tour of the pipeline, written for someone reading the code for the first time. It
explains the four decisions that were hard rather than describing every module.

```
          ┌──────────────┐
  case ──▶│  generate    │  valid inputs only; never sees a fault label
          └──────┬───────┘
                 ▼
          ┌──────────────┐      ┌──────────────┐
          │  reference   │      │  candidate   │   same weights, same inputs
          │   adapter    │      │   adapter    │   different implementation
          └──────┬───────┘      └──────┬───────┘
                 └────────┬────────────┘
                          ▼
                   ┌─────────────┐
                   │  compare    │  the failure oracle: PASS / FAIL / …
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │   replay    │  is the verdict stable, or UNSTABLE?
                   └──────┬──────┘
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │  localize   │         │   reduce    │  ddmin + greedy baseline
       │ (trace.py)  │         │ (reduce.py) │
       └─────────────┘         └──────┬──────┘
                                      ▼
                               ┌─────────────┐
                               │   export    │  standalone package
                               └─────────────┘
```

## 1. The failure oracle

The question "did the candidate break?" has no exact answer in floating point. Two correct
implementations of the same math disagree — incremental cached decode versus a full-prefix
forward pass differ at `max|Δ| ≈ 2.4e-07` on this fixture purely from reassociation. An oracle
that demanded bitwise equality would report that as a bug.

So `compare.py` is a *tolerance policy*, not an equality test, and it is deliberately boring:

- Both absolute and relative terms, `|a − b| ≤ atol + rtol·|b|`, elementwise.
- Non-finite values compare by **kind**, not by value: `NaN` matches `NaN`, `+inf` matches
  `+inf`, and a `NaN` opposite a finite number is a violation. Silently treating `NaN == NaN`
  as equal would hide the most interesting class of regression.
- Relative error uses a norm ratio, with an explicit rule for a zero-norm denominator rather
  than an accidental division by zero.
- The comparison reports `max_abs_err`, relative L2, and a violation count, not a boolean. The
  policy turns those into a verdict; the numbers stay visible.

**Verdicts are disjoint and never collapsed.** `PASS`, `FAIL`, `INVALID` (the input violated
the contract), `ERROR` (something raised), `TIMEOUT`, `RESOURCE_LIMIT`, `UNSTABLE`. A crash is
not a detection. This matters because the easiest way to inflate a detection rate is to count
crashes and flakes as finds.

**Calibration discipline.** `atol`/`rtol` were chosen on *known-good* comparisons only — the
seven controls in `bench/controls/` — and frozen before any held-out evaluation. Loosening a
tolerance until a faulty candidate passes would be circular, and loosening it after seeing
held-out results would be worse.

**Stability.** A single disagreeing run is not a failure. `replay.py` re-runs the comparison
from clean state and requires the verdict to repeat; a verdict that changes is `UNSTABLE`, a
category of its own. The reducer's predicate uses the stable verdict, so the whole search is
built on a repeatable signal.

## 2. Checkpoint alignment

To say *where* a difference first becomes visible, you need to compare the reference's
internal activations against the candidate's. The obvious implementation — register forward
hooks on both models and zip the two sequences in invocation order — is wrong, and quietly so.

Hook invocation order is an artifact of *how* a model executes, and the whole premise of this
tool is that the two implementations execute differently. A cached decode calls each block once
per step; a full-prefix pass calls it once for the whole sequence. Zipping those two streams
pairs unrelated tensors and produces confident nonsense.

EvalLens instead gives every captured tensor a **semantic address**:

```
(request_id, layer_name, token_position, kind)
```

`trace.py` aligns on that address and reports unmatched checkpoints on either side explicitly,
rather than dropping them. Two adapters can disagree about how many times they call a block;
they cannot disagree about which request, which layer, which token position, and which kind of
activation a tensor represents.

**Earliest *observed* divergence, not root cause.** The result is the first address in a
defined traversal order whose comparison leaves tolerance. That is evidence about where a
difference becomes visible *at the checkpoints these adapters expose*. It cannot see inside a
layer, and it is not a claim about causation.

**Why there is no binary search.** The natural optimization is to bisect: find the boundary
between "clean" and "diverged" in logarithmic time. Bisection requires the predicate to be
monotone along the traversal, and it is measurably not. In the full benchmark run, **108 of 160**
stable failures *reconverged* — an aligned checkpoint returned inside tolerance after an
earlier one had left it. Normalization is the usual reason: a large pre-norm difference can
shrink back under tolerance downstream. So `localize` compares every aligned checkpoint, and
the viewer reports reconvergence when it happens rather than hiding it.

## 3. Reduction, and what 1-minimal actually means

A failing case from the generator is typically 14–200 tokens. `reduce.py` shrinks it while
preserving the **failure signature**: the verdict, the target request, the execution mode, and
(optionally) the checkpoint identity. Preserving a signature is not proof of identical root
cause — it establishes the same failure class at the same target in the same category — and the
viewer says so on the page.

Size is a lexicographic tuple:

```
(n_requests, n_valid_tokens, n_padding_tokens, token_value_complexity)
```

Every accepted step must **strictly decrease** it. That is what guarantees termination; without
it, a transform that trades tokens for padding could cycle forever.

Four deletion operations are declared: remove requests, remove tokens, remove padding, and
simplify token values toward the canonical id. The last one is grouped by token *value* rather
than by slot, and that detail was a real bug. Canonicalizing one slot of `[2, 2, 2]` gives
`[1, 2, 2]` — two distinct values where there was one — so `token_value_complexity` goes *up*
and the step is rejected. Correct, but it burned most of the query budget on rejections. A
property test caught it. Grouping by value makes the operation monotone by construction.

> **1-minimal means 1-minimal with respect to those four declared operations.** It does not
> mean "smallest possible". A smaller counterexample may exist under operations EvalLens does
> not perform. Every place this project reports minimality says this.

**ddmin versus the baseline.** `ddmin` is structure-aware delta debugging; the greedy
single-deletion baseline was implemented independently so the comparison is real. Both ran from
identical frozen starting cases under identical budgets, on a cohort chosen *before* either
reducer was written. The measured result, in full:

| | ddmin | greedy |
|---|---|---|
| Median reduction | 35.2x | 35.2x |
| Median predicate queries | 12 | 82 |
| Cases where it produced a smaller result | 0 | 0 |

**ddmin does not produce smaller cases here.** Sixteen of sixteen paired cases tied. Its
advantage is cost: a median 6.2x fewer predicate queries, each of which is a pair of model
executions. That is the honest claim.

Predicate results are cached, keyed by the case content hash, *both* adapter identities, the
policy id, the signature, and the numeric environment identity. Keying on the case alone would
silently reuse a verdict across a different candidate or a different tolerance.

## 4. Search-policy blindness

`evallens.generate` and `evallens.reduce` must never see what they are looking for. They do not
import `bench.mutants`, do not read a `Behavior`, and never touch mutant ids, source patches,
trigger fixtures, or expected answers. They reach a model only through the `Adapter` protocol
and learn only a verdict.

This is enforced two ways. `tests/unit/test_search_policy_blindness.py` fails if those modules
acquire such an import. And `bench/` is deliberately **not** part of the installed `evallens`
package, so a shipped wheel physically cannot import the fault corpus — which is why
`evallens bench` has to locate a repository checkout and says so when it cannot.

Without this, every number in `RESULTS.md` would be meaningless: a search that can read the
answer key always wins.

## Where the code lives

| Path | What it does |
|---|---|
| `src/evallens/types.py` | Case schema, verdicts, tolerance policy, the `Adapter` protocol |
| `src/evallens/fixtures/` | The tiny transformer, its weights, and an independent FP64 NumPy oracle |
| `src/evallens/adapters/` | Reference and candidate adapters, input encoding, checkpoint capture |
| `src/evallens/compare.py` | The failure oracle |
| `src/evallens/replay.py` | Stability replay and subprocess replay |
| `src/evallens/generate.py` | The two blind input generators |
| `src/evallens/trace.py` | Checkpoint alignment and localization |
| `src/evallens/reduce.py` | ddmin, the greedy baseline, minimality certification |
| `src/evallens/export.py` | Standalone reproduction packages and their verification |
| `src/evallens/cli.py` | The command surface |
| `bench/` | Injected faults, controls, seeds, the run harness, the report generator |
| `viewer/` | The offline, loopback-only inspector |

## The fixture, and why it is tiny

`tiny-2L-64d-4h` is a 120,801-parameter decoder-only transformer: 2 layers, `d_model` 64, 4
heads, learned positions, untied embeddings. It is small on purpose. The entire test suite runs
in under a minute on a laptop CPU, the full benchmark in 80 seconds, and an exported
reproduction carries its own weights in 744 KB.

Its correctness is not assumed. `fixtures/numpy_oracle.py` recomputes the forward pass in FP64
**sharing no helper with the implementation** — its attention mask is built with explicit nested
Python loops, its GELU calls `math.erf` through `np.vectorize`, its softmax goes row by row. If
the oracle imported the same masking helper as the model, a bug in that helper would cancel out
and the oracle would certify it. They agree at `max|Δ| = 2.388e-07`, which `evallens doctor`
re-checks on every run.
