# Interview guide

A study guide for explaining EvalLens out loud, and a set of questions to answer before
claiming to.

> **This document does not certify that anyone understands this code.** It is a list of things
> a person should be able to explain and demonstrate. Understanding is shown by answering the
> questions below at a whiteboard and by making the code do something new — not by having read
> this file. Do not describe yourself as understanding EvalLens until you have done that.

## The 90-second version

> When you swap a correct inference path for a faster one — a KV cache, a batched kernel, fused
> attention — the failure mode that hurts is the silent one. Nothing raises. Shapes are right.
> The numbers are quietly wrong, and you find out a week later when a quality metric slips.
>
> EvalLens runs a reference and a candidate implementation on identical weights and identical
> inputs and asks four questions in order. Did they disagree beyond a tolerance calibrated on
> known-good comparisons? Does that disagreement repeat on replay, or was it noise? At which
> aligned internal checkpoint does the difference first become visible? And what is the smallest
> input that still triggers it? Then it exports a standalone package that reproduces the
> mismatch on its own, with no dependency on EvalLens.
>
> The measured run is 202 declared trials in 80 seconds on a laptop CPU: it detects all 160
> qualified fault trials within budget, with zero false positives on 2,560 known-good cases.
> Every fault in that corpus is one I injected on purpose — it is test material for the
> detector, not a bug found in PyTorch.

## The five things worth explaining

Each of these is a decision where the obvious implementation is wrong.

### 1. The oracle cannot be an equality test

Two *correct* implementations disagree. On this fixture, cached incremental decode differs from
a full-prefix forward pass at `max|Δ| ≈ 2.4e-07` purely from floating-point reassociation. So
the oracle is a tolerance policy, calibrated on known-good comparisons only and frozen before
held-out evaluation. Non-finite values compare by kind, so a `NaN` appearing where a number
belongs is a violation rather than a silent match.

*Be ready for:* "How do you know your tolerance isn't just wide enough to hide the bug?"
Answer: it was calibrated on the seven controls, which are correct-vs-correct comparisons, and
frozen. The counter-evidence is the false-positive rate — 0 on 2,560 known-good cases means the
tolerance is not absurdly tight, and 160/160 detection at the same tolerance means it is not
absurdly loose. Neither number alone would settle it.

### 2. Checkpoint alignment cannot use hook order

Registering forward hooks on both models and zipping the results in invocation order is the
natural implementation and it is wrong. A cached decode calls each block once per token; a
full-prefix pass calls it once for the sequence. Zipping those streams pairs unrelated tensors.
EvalLens addresses every checkpoint semantically — `(request_id, layer_name, token_position,
kind)` — and reports unmatched checkpoints instead of dropping them.

*Be ready for:* "Why not bisect to find the divergence point?" Because the predicate is not
monotone. 108 of 160 stable failures **reconverged**: a checkpoint came back inside tolerance
after an earlier one had left it, usually because a normalization shrank the difference. That is
measured, not assumed, and it is why every aligned checkpoint is compared.

### 3. Reduction needs a well-ordering, not just a shrink loop

Case size is the lexicographic tuple `(n_requests, n_valid_tokens, n_padding_tokens,
token_value_complexity)` and every accepted step must strictly decrease it. That is what
guarantees termination — otherwise a transform trading tokens for padding can cycle.

*Be ready for:* "Is the reduced case minimal?" It is 1-minimal **with respect to the four
declared deletion operations**. A smaller counterexample may exist under operations EvalLens
does not perform. The tool never says "smallest".

*Also be ready for:* "Did ddmin beat your baseline?" **No, not on size.** Sixteen paired cases,
sixteen ties. ddmin used a median 6.2x fewer predicate queries to reach the same result. That is
the honest claim, and on this corpus greedy is a perfectly adequate reducer if you don't care
about query cost.

### 4. The search must be blind

`generate` and `reduce` never see a mutant id, a source patch, a trigger fixture, or an expected
answer. They reach a model only through the `Adapter` protocol and learn only a verdict. A test
fails if those modules acquire such an import, and `bench/` is not part of the installed package
so a shipped wheel physically cannot reach the fault corpus. Without that, every detection
number would be worthless.

### 5. The report's denominator is written before the numerator exists

`bench/run.py` enumerates all 202 trial ids and freezes them into `manifest.json` — with the git
commit, clean-tree flag, weights hash, tolerance policy id, and seed cohorts — *before any model
runs*. `bench/report.py` then checks records against that frozen list. Sixteen conditions are
fatal, including a dirty working tree.

*Be ready for:* "Why does that matter?" Because if you count whatever happens to be in the
output directory, a crashed trial silently leaves the denominator and 100% detection can mean
"everything that finished, finished". That is survivorship, and it is the single easiest way to
publish a wrong number honestly.

## Questions you should be able to answer without notes

**On the oracle**
1. Why isn't `torch.allclose` enough?
2. What happens when the candidate returns `NaN` and the reference returns `0.5`? Which verdict,
   and why is it not `PASS`?
3. A comparison passes on one run and fails on the next. What verdict, and why isn't it `FAIL`?
4. What is the zero-norm denominator rule for, and what breaks without it?

**On alignment and localization**
5. Why can't you align checkpoints by hook invocation order?
6. What exactly does "earliest observed divergence" claim, and what does it not claim?
7. Show a case where the divergence reconverges. What does that rule out?

**On reduction**
8. Why is size a tuple and not a token count?
9. Why is the token-value operation grouped by value rather than by slot? What went wrong when
   it wasn't?
10. What is the predicate cache keyed on, and what breaks if you key it on the case alone?
11. In what sense is the output minimal, and in what sense is it not?

**On the benchmark**
12. Why resample fault families instead of trials?
13. Why is the bootstrap interval degenerate, and why is that reported rather than smoothed?
14. Why can't this benchmark tell you which of the two generators is better?
15. What stops a synthetic timing fixture from producing `RESULTS.md`?
16. Why must a published run come from a clean git tree?

**On honesty**
17. Which results in this project went against the hypothesis? (There are two: ddmin's tie on
    size, and the saturated corpus. Both are published.)
18. What would you have to see before calling something a bug in PyTorch rather than an
    observation to investigate?

## Demonstrate it, don't assert it

Reading this file is not evidence. These are:

- Run `evallens demo` and narrate each of the six steps as it happens.
- Open `viewer/viewer.js` and explain why `field()` renders `not recorded` instead of a default.
- Add a **ninth** fault family to `bench/mutants/`, write its handwritten trigger, and get it to
  qualify. If it doesn't qualify, explain why — that is the more interesting outcome.
- Break something on purpose: loosen `atol` by 10x and explain which number moves and which
  doesn't.
- Delete every `greedy` record from a run directory and run `bench/report.py`. Explain the exact
  integrity code it raises and why a `missing_trials` error alone wasn't enough.
- Trace one tensor from `encode_cached_steps` through `KVCache.length_of` to a checkpoint
  address, and explain why `length_of` is per-layer.

## Things not to say

- Don't call an injected fault a discovered bug.
- Don't quote a detection rate without its budget, its corpus size, and the fact that the corpus
  is saturated.
- Don't say "minimal" without "with respect to the declared operations".
- Don't claim ddmin produced smaller cases. It didn't.
- Don't present 100% as a strong result. On a corpus nothing misses, it is a statement about the
  corpus.
