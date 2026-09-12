# Supported scope

What EvalLens does, what it does not do, and where the edges are. Nothing here is aspirational;
everything listed as supported is exercised by a test or by a run in `RESULTS.md`.

## Supported

**Models.** One built-in fixture family: decoder-only pre-norm transformers with learned
position embeddings, built by `evallens.fixtures`. The unit fixture is `tiny-2L-64d-4h`
(120,801 parameters); a larger `SCALE_FIXTURE` config exists for stress runs.

**Execution modes.** Three, and the differences between them are the point:

| Mode | What it exercises |
|---|---|
| `STATELESS_BATCH` | padded batching, masking, row independence |
| `CACHED_DECODE` | KV-cache correctness under teacher-forced incremental decoding |
| `SESSION` | state isolation across up to 3 sequential requests on one adapter |

**Inputs.** Batches of up to 4 rows, sequences up to the configured `max_position` (128 by
default), left or right padding, per-request prefill boundaries. Position ids are assigned by
**valid-token index, not column index**, which is what makes padding invariance a testable
property rather than an accident of layout.

**Fault families detected.** Eight, sixteen variants, all injected by this project:
attention scaling, causal masking, padding masking, normalization, batch indexing, KV-cache
indexing, decode position, and request reset. `bench/mutants/` defines them; each qualifies by
producing a stable, shape-valid, silent mismatch on its own handwritten trigger.

**Platform.** CPU float32 is the reference platform and the only one any published number comes
from. Python 3.11+; measured on 3.13.7 with torch 2.14.0 and numpy 2.5.3.

## Not supported

- **No GPU path.** CUDA is never assumed. MPS is visible on the development machine and is
  deliberately unused; a `mps` pytest marker exists for optional extension work, and no
  published result depends on it.
- **No downloaded weights, no network, no service.** The fixture is generated locally from a
  seed. Nothing phones home.
- **No adapter plugin mechanism.** `evallens compare`, `reduce`, and `export` accept native
  fixture bundles only. Pointing EvalLens at your own model means implementing the `Adapter`
  protocol in-process; there is no configuration file that will do it for you.
- **No encoder-decoder, MoE, or state-space models.** The checkpoint kinds are attention
  output, MLP output, block output, final norm, embedding, and logits.
- **No sampling.** Everything is teacher-forced and deterministic. A sampling difference is not
  something this tool can see.
- **No training.** Inference only.
- **No root-cause analysis.** EvalLens localizes the earliest *observed* divergence at exposed
  checkpoints. It cannot see inside a layer, and it does not attribute cause.

## What the numbers do and do not mean

`RESULTS.md` states these too. They are repeated here because they are the part most likely to
be over-read.

1. **Every fault measured is one this project injected on purpose.** None is a bug discovered
   in PyTorch or any other third-party library. If EvalLens ever flags an external library, that
   is an observation to investigate, not a discovery to announce.
2. **The corpus is saturated.** Both generators detect 100% of trials at the benchmark budget,
   so the benchmark **cannot rank them** — their median cases-to-detection is identical (2.5).
   A benchmark every method passes measures nothing about the methods. The family-level
   bootstrap interval is degenerate for the same reason, and reporting it as a tight interval
   would be misleading.
3. **Trials within a mutant are not independent.** Five seeds against one variant measure the
   same fault five times. Uncertainty is resampled at the **family** level for that reason.
4. **Eight families is a narrow corpus**, on one fixture, on one machine. No generalization to
   unknown real-world regressions is claimed.
5. **"Detection within budget" is relative to this budget**, this fixture, and these generators.
   A trial that did not detect is budget-censored and stays in the denominator; it is never
   dropped.
6. **A preserved failure signature is not a proof of identical root cause.**
7. **1-minimal is with respect to the four declared deletion operations**, never globally
   smallest.
8. **ddmin does not beat the greedy baseline on size** — sixteen ties out of sixteen. It wins on
   query cost (median 6.2x fewer), and that is the only advantage claimed.

## Safety and locality

- The viewer binds loopback only and refuses any other host; it serves static files with a
  path-traversal guard and there is no backend.
- Exported reproductions are verified by running them from a fresh temporary directory
  **outside the checkout** in an isolated interpreter (`python -I`), and the script itself
  checks `sys.modules` and exits with a setup error if `evallens` leaked in.
- Nothing in this project modifies system Python, shell configuration, or anything outside the
  project directory.
