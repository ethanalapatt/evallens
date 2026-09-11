# EvalLens: Claude Code master build prompt

Copy everything below into Claude Code, or place this file in the project directory and tell Claude to read it and implement it.

---

You are my senior ML systems engineer. Build **EvalLens**, a local differential debugger for neural-network inference that detects output regressions, identifies the earliest observed divergence at aligned checkpoints, minimizes a failing input, and exports a standalone reproduction.

Implement the working project. Do not stop after proposing an architecture, writing documentation, scaffolding empty modules, or creating a dashboard with mock data. Work through the milestones in order, validate the actual behavior, make small commits, and preserve progress between sessions. Make routine implementation decisions yourself without asking me repeated clarification questions.

## 1. My constraints and the goal

- My development machine is a **MacBook Air M3 with 16 GB unified memory**. Development, correctness tests, the demo, and the primary benchmark must work entirely on this machine.
- Use **CPU execution as the required reference platform**. Apple GPU/MPS comparisons are an optional extension after the CPU project works. Never assume CUDA or access to my separate DGX Spark.
- I know Python, PyTorch, NumPy-style scientific programming, TypeScript/JavaScript, React, and testing. Prefer these. No new systems language is necessary.
- The project should be credible for SWE, ML infrastructure, evaluation infrastructure, and research-engineering internships.
- My existing portfolio already includes distributed workflows, RAG, computer vision, and full-stack applications. EvalLens needs substantive numerical and debugging algorithms.
- No paid APIs, cloud infrastructure, model training, external inference service, or mandatory Docker setup. Keep the project local. Internet access is only needed for installing dependencies and an optional small public model download.
- Budget roughly 12–16 focused implementation sessions over 2–4 weeks. Treat this as an estimate. Prioritize a correct, measured CPU release.
- Keep normal project workloads within a **4 GiB combined worker RSS budget**, with short sequences and one benchmark worker. This is a design budget, not a claim about measured usage. Check actual memory and stop/mark a run if it exceeds the configured bound. Leave room for macOS, the editor, browser, and Claude Code.

The project question is:

> Can a small, transparent debugging engine detect silent inference regressions and turn large failing cases into useful minimal reproductions, with measured detection, false-positive, and reduction behavior?

The output is an engineering tool and benchmark study. Do not claim novel research, universal support for all models, or real PyTorch bugs without actual evidence.

## 2. Concrete user experience

A developer supplies a reference adapter, candidate adapter, and valid input cases. An optimized candidate may produce a different result even though nothing crashes. EvalLens:

1. Runs both implementations on the same weights, inputs, and specified execution conditions.
2. Compares the intended output tensors using an explicit numerical policy.
3. If a stable discrepancy exists, captures aligned intermediate checkpoints in a separate diagnostic pass.
4. Locates the earliest **observed** checkpoint that exceeds tolerance. Explain that this is evidence about where divergence is visible, not proof of its root cause.
5. Removes unnecessary requests/tokens and simplifies the input while preserving the same valid failure.
6. Writes a replayable case, reduction history, environment manifest, and standalone reproduction.
7. Displays the real recorded evidence in a local viewer.

The required demo begins with a clearly labeled deliberately injected cache or position bug. It shows the original case, observed mismatch, reduction steps, reduced case, and a runnable exported reproduction. Never present an injected fault as an upstream discovery.

## 3. Keep the MVP narrow

Required:

- A small native decoder-only transformer fixture with independently checked attention/masking behavior.
- Correct full-prefix and incremental cached execution of that fixture.
- Typed adapters so the debugging engine does not depend on the native fixture's internals.
- Stateless single/batched inputs and sequential request sessions for testing cache reset. Incremental cache mode can be batch size one.
- Valid-case generation, numerical comparison, checkpoint alignment, deterministic failure replay, structure-aware delta debugging, and reproduction export.
- An explicit corpus of injected implementation faults and independently validated known-good controls.
- Benchmarks against named simple baselines, generated results, and an offline trace viewer.

Deferred:

- Arbitrary `torch.compile` graph debugging, arbitrary graph rewriting, quantized-runtime equivalence, automatic source-code repair, distributed execution, production serving, LLM judges, model training, and a general-purpose model hosting UI.
- MPS and pretrained model adapters are extensions. The primary result cannot depend on either.
- No Kafka, Kubernetes, database, agent framework, authentication, or remote deployment.

## 4. Stack and repository

Use Python with PyTorch, NumPy, pytest, Hypothesis, Ruff, a type checker, and standard-library JSON/CLI/reporting where practical. A small static HTML/CSS/JavaScript viewer is sufficient. Do not introduce a frontend build system unless a concrete feature requires it.

Inspect the current directory and git state first. Work in the current project directory. Preserve unrelated user files. If it is empty, initialize the project here. If an incompatible existing repository occupies the directory, use a clearly named `evallens/` subdirectory rather than overwriting it.

Create this layout, adapting filenames only when necessary and documenting the reason:

| Path | Purpose |
|---|---|
| `README.md` | Reviewer entry point, real demo, installation and reproduction |
| `SPEC.md` | Resolved technical specification derived from this prompt |
| `CLAUDE.md` | Persistent agent conventions and scope rules |
| `PROGRESS.md` | Actual milestone state, commands, evidence, blockers, next action |
| `pyproject.toml`, dependency lock | Package, dev dependencies, reproducible environment |
| `src/evallens/types.py` | Case, result, trace, and adapter contracts |
| `src/evallens/adapters/` | Native reference/cached adapters; optional HF/MPS adapters |
| `src/evallens/fixtures/` | Tiny transformer and independent attention checks |
| `src/evallens/generate.py` | Valid-case generators |
| `src/evallens/compare.py` | Numerical policy and classification |
| `src/evallens/trace.py` | Semantic checkpoint alignment and summaries |
| `src/evallens/reduce.py` | Structured delta debugging and linear baseline |
| `src/evallens/replay.py` | Isolated, repeatable case execution |
| `src/evallens/export.py` | Portable reproduction packages |
| `src/evallens/cli.py` | CLI entry point |
| `bench/mutants/` | Explicitly labeled injected faults; not imported by the search policy |
| `bench/controls/` | Known-good implementation comparisons |
| `bench/run.py`, `bench/report.py` | Evidence acquisition and generated reports |
| `configs/` | Smoke, benchmark, tolerance, and resource settings |
| `tests/unit/`, `tests/property/`, `tests/integration/` | Real behavior tests |
| `viewer/` | Offline result and trace inspector |
| `docs/` | Algorithm explanations, limitations, decisions, interview guide |
| `artifacts/runs/<run_id>/` | Immutable manifests, measurements, cases, and derived reports |
| `examples/reproductions/` | A few real generated reproduction packages |
| `.github/workflows/ci.yml` | Small CPU-only CI suite |

Pin versions that actually work on this machine. Record Python, PyTorch, NumPy, macOS, chip, dtype, CPU thread settings, and MPS availability. Do not invent tested versions or lock hashes. Use a project-local environment; avoid changing system Python or shell configuration.

## 5. Case and adapter contracts

Implement typed dataclasses/protocols along these lines:

```python
class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"  # Stable numerical/output-contract discrepancy
    INVALID = "invalid"  # Input/adapter contract violated
    ERROR = "error"  # Unexpected exception, tracked separately
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    UNSTABLE = "unstable"  # Classification changes on replay


@dataclass(frozen=True)
class Request:
    token_ids: tuple[int, ...]
    prefix_length: int
    # Explicit padding/batch/session fields belong in the resolved schema.


@dataclass(frozen=True)
class Case:
    schema_version: int
    case_id: str
    model_config_id: str
    weights_sha256: str
    requests: tuple[Request, ...]
    execution_mode: str
    input_seed: int


class Adapter(Protocol):
    def reset(self) -> None: ...
    def validate(self, case: Case) -> None: ...
    def run(self, case: Case, capture: bool = False) -> "ExecutionResult": ...


def compare(
    reference: "ExecutionResult", candidate: "ExecutionResult", policy: "TolerancePolicy"
) -> "ComparisonResult": ...


def reduce_case(
    case: Case, predicate: "FailurePredicate", budget: "ReductionBudget"
) -> "ReductionResult": ...
```

Complete the schema before implementing it. Document every tensor's axes. Input validation must check token ranges, nonempty requests, model context limits, mask shapes, and the existence of at least one valid token per row. Never accept all-padding inputs accidentally.

Each replay begins with clean adapter state. State within a multi-request case is intentional and serialized; state must not leak between separate cases. Preserve the original failure's target request using a stable identifier when removing earlier requests.

For cache comparisons, use **teacher-forced shared token sequences**. At step t, both adapters receive the same canonical prefix, even if their argmax predictions differ. Do not feed each model its own prediction and then compare activations on different inputs.

Full-prefix logits and incremental logits must be aligned by logical token position. Compare only intended valid positions, with masks and position IDs reconstructed consistently. Do not compare padded outputs as if they had semantic meaning.

Checkpoints are keyed by stable semantic addresses such as `(request_id, layer_name, logical_token_position, checkpoint_kind)`. A reference full-sequence call and several incremental calls have different call counts; raw hook invocation order is not a valid alignment rule.

## 6. Fixtures and reference correctness

Implement a small understandable decoder-only transformer: token and learned position embeddings, pre-normalized causal self-attention, residuals, MLP, final norm, and vocabulary projection. Tie or untie weights explicitly and consistently. Disable dropout for inference tests.

- Unit fixture: about two layers, width 64, four heads, vocabulary 97, context at most 128. Small exact dimensions may change if necessary, but record them.
- Primary benchmark fixture: remain small enough for thousands of CPU evaluations; use the unit fixture or a slightly larger locked configuration after a pilot.
- Optional scale check: a separate roughly 10–20 million parameter fixture with short sequences. Do not inflate every correctness test to this size.

Random deterministic weights are sufficient for execution-correctness testing. Training is unnecessary. Clearly identify untrained fixtures; their outputs do not demonstrate language capability.

Initialize one canonical state dictionary and copy it to implementations, checking hashes. Merely setting the same seed twice is not enough if initialization paths differ. Run models in eval/inference mode.

Check the reference itself using:

- Independent NumPy FP64 calculations for tiny attention, normalization, masks, and selected end-to-end tiny cases where practical.
- Causal prefix invariance: adding future tokens cannot alter earlier valid outputs.
- Padding invariance under a documented position-ID convention.
- Batch permutation invariance for stateless execution.
- Full-prefix versus incremental execution with identical prefixes.
- Request isolation with fresh caches.

Do not share the same masking/indexing helper between an independent oracle and the implementation it is intended to verify. Explain what independent checks do and do not establish.

## 7. Numerical comparison and localization

For each valid aligned tensor entry, calculate absolute error and use an explicit policy such as:

```text
violation = abs(candidate - reference) > atol + rtol * abs(reference)
```

Initial CPU FP32 settings can be `atol=1e-5`, `rtol=1e-4`; these are starting policy choices, not universal mathematical truths. Validate and, if justified, calibrate thresholds on **known-good calibration comparisons only**. Freeze the policy before held-out fault evaluation. Record threshold changes and rationale; never loosen tolerances to make a faulty candidate pass.

Report maximum absolute error, relative L2 error with a defined zero-norm denominator rule, fraction of violating entries, shapes, dtype, and finite/nonfinite counts. Do not divide by tiny reference elements to create a misleading headline relative-error statistic.

The primary failure predicate uses intended final output tensors. Intermediate discrepancies provide diagnostic evidence and may reconverge before the output; they must not automatically become output-regression detections. Nonfinite outputs on valid finite cases are recorded explicitly as numerical failures. Crashes, invalid inputs, timeouts, and resource limits stay in separate categories.

Re-run any candidate failure three times from clean state. Only a stable mismatch enters reduction. A result that changes classification is `UNSTABLE` and remains visible in the report.

On a stable failure, perform a separate traced pass. Compare all supported aligned checkpoints in execution order; do not binary-search for the first mismatch unless you can justify monotonicity, which should not be assumed. Numerical discrepancies may appear, disappear, and reappear.

Call the result the **earliest observed divergence**. If adapters expose only layer outputs, do not pretend to have identified an exact operation inside a layer. If no checkpoints align, retain the output-level failure and label localization unavailable.

Do not retain all large activations for every generated test. Record compact summaries by default and bounded tensor excerpts only for selected failures. Keep capture, serialization, and viewer work separate from normal comparison timing.

## 8. Fault corpus and controls

Implement eight injected fault families in candidate-only code, with two declared variants per family where practical:

1. A causal mask boundary that incorrectly permits a future position.
2. Incorrect padding-mask application on valid padded inputs.
3. Wrong positional offset during incremental decoding.
4. Incorrect cached key/value indexing or stale-cache slot reuse.
5. Failure to reset request-specific cache state between requests.
6. Incorrect normalization epsilon or normalization axis.
7. Incorrect attention scaling.
8. Incorrect batch indexing that reuses one row's data for another.

Prefer shape-valid silent faults. Track crashes separately if a mutant also crashes. Every included variant must have an independently constructed valid trigger fixture demonstrating its intended fault before the benchmark manifest is frozen. Keep non-qualified variants visible as incomplete rather than quietly deleting poor results.

Known-good comparisons must include identical implementations, correct full-prefix versus cached execution, correctly aligned padding transformations, batch permutations, and fresh-request isolation. Add at least one benign numerical perturbation below the fixed tolerance so exact equality is not the only exercised control.

Fault labels and known trigger fixtures belong to the scoring/validation layer. The case generator and reducer must not inspect mutant IDs, source patches, trigger examples, or expected answers. They operate through the adapter and verdict contracts.

Use separate calibration, development, and evaluation seed sets. These are held-out executions of known fault families, not proof of generalization to unknown real-world bugs. State that distinction.

## 9. Valid input generation

Provide two generators for the same declared valid-case space:

- **Uniform-valid baseline:** random valid lengths, token IDs, padding, batch sizes, and session structures.
- **Boundary-aware generator:** emphasizes short/long prefixes, decode boundaries, repeated token patterns, unequal padded lengths, and request transitions.

Both must support every declared case category and receive the same model/adapter capability information. Neither knows which mutant is running. Use the same per-trial case budget and record case hashes. Generate inputs before timed execution, so input preparation and model comparison time can be reported separately.

Use bounded lengths and batch sizes. Initial limits: at most 128 logical tokens per request, batch size 1–4 for stateless comparisons, and at most three sequential requests for session tests. Incremental cache comparisons remain batch size one. Respect the configured memory/time budgets.

Property-test generator validity, determinism, category coverage, and meaningful diversity. Do not benchmark invalid random tensors against a validity-aware generator and call the difference improved bug detection.

## 10. Failure-preserving input minimization

Implement structure-aware `ddmin` yourself, with a simple understandable implementation and a separately implemented greedy single-deletion baseline.

Reduction operations, in order:

1. Remove unnecessary earlier requests from a session while retaining the target request.
2. Remove chunks of tokens while keeping requests nonempty and cache prefix/decode boundaries valid.
3. Reduce padding and simplify eligible batch structure for stateless cases.
4. Simplify token values to a fixed canonical set.
5. Repeat eligible passes until no further accepted reduction or the budget is exhausted.

Reconstruct dependent masks, positions, and prefix/decode boundaries from the reduced canonical case. Reject a transformation that changes the declared test category or cannot preserve its contracts. Shrinking may change token positions; both adapters must receive the same recomputed semantics.

Accept a reduction only when it remains valid, reproduces a stable output failure, and matches the intended failure signature. The signature includes the failure class and target request. If the task is specifically to preserve a localized checkpoint failure, also preserve that checkpoint identity. Explain that even a matching signature cannot universally prove identical root cause.

Use lexicographic size `(number_of_requests, total_valid_tokens, padding_tokens, token_value_complexity)` and require strict improvement for accepted changes. This prevents simplification cycles. Do not alter weights, tolerances, implementation, or model architecture while minimizing an input.

Cache predicate results by canonical case hash plus adapter, weights, policy, and environment identities. Count logical predicate queries, actual model runs, and cache hits separately.

Stop at 256 logical predicate queries or 60 seconds per initial MVP reduction, whichever comes first; make both configurable. Charge stability replays and final verification to an explicitly recorded model-run/time budget. Each baseline gets the same declared limits. Preserve the best valid failing case on timeout.

After reduction, replay the resulting case three times and rerun it in a fresh subprocess. Only claim **1-minimal with respect to the declared deletion operations** if every remaining eligible single deletion was checked and failed to preserve the target failure. Otherwise label it `reduced, minimality not established`. Never call it a globally smallest counterexample.

## 11. Reproduction export

Export a self-contained directory containing:

- `repro.py`, with a concise explanation of the failure.
- The minimal model/adapter implementation necessary for the selected supported fixture.
- Input/session data as JSON, deterministic weights as non-object NPZ or safetensors, tolerance policy, and hashes.
- A dependency/environment manifest and a short README with exact commands.
- Source commit, original/reduced case IDs, failure signature, and whether this was an injected fault.

The native-fixture reproduction must run with its declared lightweight dependencies and **must not import the installed EvalLens package** or reference absolute paths to the original checkout. Verify it from a fresh temporary directory/subprocess with `PYTHONPATH` cleared.

Use an explicit `--expect-mismatch` mode that exits successfully only when the recorded mismatch is reproduced. A default comparison mode should return a documented nonzero code for a mismatch, distinct from setup/execution errors. Do not mistake an import error for successful reproduction.

The exported reproduction is an artifact of the actual reducer. Never handcraft a tiny input and pretend the program discovered it.

## 12. Benchmark plan and metric integrity

Implement a small smoke preset and a full declared CPU preset. Use a pilot to estimate duration, then keep the finalized workload and budgets fixed. Run sequentially to fit the Mac's memory and thermal constraints.

Suggested full detection preset:

- Eight fault families, two qualified variants per family.
- Five held-out input seeds per variant and generator.
- Up to 64 valid cases per trial, with early detection recorded and unused cases visible in the manifest.
- Both uniform-valid and boundary-aware generators under equal budgets.
- A separate held-out known-good set of 256 comparison cases for each generator. Calibration controls are disjoint.

These are workload parameters, not performance claims. If the pilot requires a smaller preset, finalize and document that change before evaluating outcomes. Never remove a slow/failing mutant after seeing the results to improve the headline.

Reduction comparison:

- Freeze a separate corpus of known valid failing starting cases, selected before running either reducer, at most two per fault family for MVP.
- Run EvalLens ddmin and greedy single-deletion from each identical initial case with the same budgets.
- Also report end-to-end reduction success on failures discovered by the generators. This is a separate population and cannot be silently substituted for the fixed reduction cohort.
- Record the cost of stability checking and optional minimality certification separately as well as in total wall time.

Generate these metrics from raw logs:

| Metric | Definition |
|---|---|
| Detection within budget | Detected mutant/seed trials divided by all declared qualified mutant/seed trials |
| Family coverage | Fault families with at least one stable detection divided by all declared families; report variant coverage too |
| Known-good false-positive rate | Stable FAIL verdicts divided by valid known-good cases; report all other verdict counts separately |
| Time/cases to first detection | Elapsed comparison time and case count; unsuccessful trials are budget-censored, not omitted |
| Reduction ratio | Original valid token count divided by reduced valid token count, with session/padding changes separately reported |
| Reduction cost | Wall time, logical predicate queries, actual model runs, and cache hits |
| Reproduction success | Exported cases reproducing the same mismatch in the clean environment divided by attempted exports |
| Localization coverage | Stable failures with valid aligned checkpoint localization divided by stable failures |
| Resource usage | Peak worker/child RSS and recorded CPU thread count; MPS telemetry is separate if available |

Report detection and false positives per family/category as well as in aggregate. Do not treat all prompts from the same mutant as independent evidence. With only eight families, uncertainty is substantial; avoid a strong generalization claim from a narrow corpus. Report paired per-case reducer comparisons and medians. If adding intervals, resample at a justified family/case level and explain the unit.

Time the model-comparison path with `perf_counter_ns`, consistent thread settings, and equivalent warmups. Report startup/import/model initialization separately. Alternate generator/reducer order with a recorded seed to reduce order bias. Record thermal/power mode when observable; do not assume a laptop sustains its initial speed indefinitely. Do not change system power settings automatically.

Primary CPU results should use wall time including the implementation work being compared. Detailed tracing/HTML serialization belongs outside those detection timings. If MPS is used later, synchronize device work around timing boundaries and report those results separately from CPU.

Every run must preserve:

- Git commit and clean/dirty state; published headline runs require clean source.
- Actual hardware/software/thread configuration.
- Fixture/weights hashes, config and tolerance hashes, seeds, qualified mutant/control manifest, generator/reducer versions.
- Complete expected trial IDs and actual outcomes, including errors/timeouts/limits.
- Raw JSONL request/trial records, reduction steps, validation outputs, and reproduction results.

Implement `bench/report.py` to create `RESULTS.md` and any plots from these raw records. It must reject missing required trials, duplicate IDs, inconsistent hashes, invalid denominators, and mismatched comparison cohorts. Interrupted studies produce an explicitly incomplete diagnostic report, not a publishable success table. Resume only under the same frozen inputs/source; otherwise start a new run.

**Hard rule: never fabricate metrics.** No estimated speedups, made-up detection rates, copied benchmark numbers, mock profiler screenshots, or manually edited result tables. Synthetic timing fixtures are allowed only in tests, clearly labeled, and blocked from generating a file named `RESULTS.md`. Before real measurements, README says `Results not yet measured`.

Fault injection is real testing evidence, but the faults are deliberate. Label that accurately in README, reports, the viewer, and resume templates. An external library mismatch is an observation to investigate; never automatically call it a newly discovered framework bug.

## 13. CLI and reviewer demo

Implement and test a coherent command interface such as:

```bash
python -m pip install -e '.[dev]'
evallens doctor
evallens demo --device cpu --out artifacts/demo
evallens compare --case examples/case.json --config configs/cpu.toml
evallens reduce --failure artifacts/demo/failure.json --out artifacts/reduced
evallens export --failure artifacts/reduced/failure.json --out artifacts/repro
python artifacts/repro/repro.py --expect-mismatch
evallens bench --preset smoke --out artifacts/runs/smoke
evallens bench --preset full --out artifacts/runs/full
evallens report artifacts/runs/full --out RESULTS.md
evallens view artifacts/demo --host 127.0.0.1
```

These are commands to implement, not claims that an existing CLI already provides them. Ensure help text and failure messages are useful.

`evallens demo` should be one command after installation: create the real injected-fault example, detect it, reduce it, export it, validate it, and write a viewer-compatible record. No downloads should be necessary for the native fixture demo.

Make the viewer clean and useful: original/reduced inputs, verdict, reference/candidate identities, earliest observed divergence, error summaries, reduction timeline, predicate counts, tolerance policy, and an export link. Clearly mark injected faults. Include color-independent labels and empty/error states. Avoid charts with invented data.

Use a static file picker for offline traces or a simple loopback-only local server. No backend service is necessary beyond serving local files. Do not expose the server publicly.

Record a short GIF or video from an actual completed demo if capture tools are available. Otherwise provide tested steps for recording it and mark that media task incomplete. Never manufacture a screenshot of a successful run.

## 14. Milestones and acceptance gates

Complete these sequentially. Each milestone is approximately one or two focused sessions; the numerical core may require more.

### M1: repository, environment, and reference fixture

Create project metadata, initial SPEC/CLAUDE/PROGRESS, CLI doctor, tiny deterministic transformer, and independent reference checks. Establish dependency locks and resource controls.

Done when installation and CPU tests work on the actual environment; weights and seeds reproduce; independent attention/mask checks and causal invariance pass; no GPU is required. Record actual environment and memory pilot. If running somewhere other than my Mac, label that environment and leave Mac verification pending.

### M2: adapters and numerical comparator

Implement correct full-prefix and cached execution, aligned output contracts, masks/positions, typed verdicts, and clean replay. Add known-good cases before mutants.

Done when native cached/reference comparisons pass, malformed inputs are INVALID, finite/nonfinite handling is explicit, repeated cases do not leak state, and tolerance tests distinguish within-policy differences from stable violations.

### M3: fault corpus and generators

Implement declared injected faults, valid trigger fixtures, the two generators, controls, and disjoint seed manifests.

Done when every included fault is independently qualified by a valid trigger, generated cases are valid/repeatable, and search policies cannot access fault labels. This does not require either generator to detect every fault under budget.

### M4: checkpoint alignment and localization

Implement bounded diagnostic capture and semantic alignment across full and cached paths. Add a crafted case where an intermediate discrepancy reconverges so first-divergence logic cannot assume monotonicity.

Done when known injected examples localize to the expected exposed checkpoint, unavailable alignment is reported accurately, and capture does not leak across runs or silently become part of only one timing baseline.

### M5: input reduction and replay

Implement ddmin, greedy baseline, valid transforms, strict size improvement, cache keys, budgets, stable signatures, and replay/minimality handling.

Done when reductions remain valid and failing, session-state cases work, timeout preserves the best valid result, no invalid case is accepted as a reduction, and property tests show determinism and termination. Confirm the reducer never reads mutant IDs.

### M6: portable reproduction and actual demo

Implement export packages and one-command end-to-end demo.

Done when an actual discovered/reduced native case reproduces from a fresh directory without importing EvalLens, execution errors are distinguished from reproduced discrepancies, and an unrelated clean case remains clean. Preserve the actual artifacts.

### M7: complete benchmark and generated report

Run the smoke pilot, freeze the full declared matrix, run it, and generate reports from raw evidence. Add report-integrity tests.

Done when the measured declared cohort is complete, all verdicts remain accounted for, clean controls have reported outcomes, reduction baselines are paired, and the report regenerates deterministically. Performance wins are not a completion gate; honest losses are valid results.

### M8: reviewer polish and ownership

Finish the viewer, README, supported-scope documentation, reproducible examples, short architecture explanation, and interview guide.

Done when a fresh CPU setup can run the demo, source/evidence links work, the viewer displays actual records, no fake measurements remain, and documentation explains the failure oracle, checkpoint alignment, ddmin limits, and benchmark limitations. Do not claim that I personally understand the code until I have demonstrated that.

## 15. Testing requirements

Use meaningful tests, not a target test count.

- Unit tests: probability-free numerical comparisons, masks, positional indices, state reset, trace alignment, case hashing, size ordering, budget arithmetic, and export paths.
- Property tests: valid inputs stay valid; reducer size strictly decreases; a claimed failure stays a failure; cache identity includes all behavioral dependencies; case serialization/replay is deterministic.
- Integration tests: real tiny transformer full-prefix/cached comparisons, injected silent faults, stateful sessions, complete demo, fresh-process exports, and report regeneration.
- Integrity tests: reject falsified coverage, duplicate trials, wrong hashes, impossible counts, synthetic metrics presented as measurements, and failures hidden as skipped rows.
- UI checks: open a generated trace, inspect labels and numerical fields against JSON, test file/error states. Use browser automation if available, otherwise record what was and was not checked.

CPU CI should be small and offline. Mark optional download/MPS tests separately. A skipped MPS test is not evidence of MPS support. Verify the commands you publish, for example:

```bash
python -m pytest -m 'not mps and not download'
python -m ruff check .
python -m ruff format --check .
python -m mypy src/evallens
evallens bench --preset smoke --out artifacts/runs/ci-smoke
```

## 16. Git, persistence, and working style

- Make small local commits after each meaningful working change and after each milestone. Include relevant test results in the progress entry, not huge generated logs in commit messages.
- If this repository already has a correctly configured and authenticated GitHub remote, push the current working branch after each completed milestone. Never force-push, delete branches, rewrite history, guess a remote URL, or push to an unrelated repository.
- If no remote is configured, keep local commits, note the missing remote in PROGRESS.md, and continue. Do not let absent GitHub setup block implementation. Do not create or publicly publish a repository automatically.
- Keep all project files inside the project folder, apart from standard local dependency caches. Do not modify unrelated projects or install background services.
- Ignore environments, downloaded weights, secrets, caches, and oversized temporary traces. Preserve compact final benchmark evidence and a few real reproduction artifacts with hashes; document where larger reproducible inputs come from.
- Update **PROGRESS.md after every milestone and before stopping mid-milestone** with status, implementation, actual commands/results, evidence paths, source commit, limitations, and the next exact action.
- Use `not started`, `in progress`, `blocked`, and `complete` truthfully. Files existing is not a completion gate.
- At the end of each milestone, write a short teach-back note: what was built, why the design was chosen, one tricky failure, and how it was tested. Continue implementation without waiting for a routine approval.
- Do not spawn sub-agents unless I explicitly ask for parallel agent work.

When a session ends, make the next session recoverable from PROGRESS.md. Do not restart completed work or silently replace the design with a different project.

## 17. Risks, fallback, and extensions

If generalized tracing grows too complex, support only the native fixture's documented checkpoints. If cache/session reduction stalls, first finish stateless token reduction and label session support deferred. If the benchmark is too expensive, finalize a smaller honest preset before inspecting final outcomes. If a new generator or reducer loses to its baseline, publish that result.

Minimum credible release: CPU reference/candidate comparisons, at least four qualified fault families, known-good controls, stable failure reduction, portable reproduction, paired baselines, and raw evidence. If reduced scope is used, state it in README and PROGRESS; do not mark the original eight-family scope complete.

After the CPU release:

1. Add MPS comparisons with a separate numerical policy, synchronization, and device metadata. Never label ordinary CPU/GPU rounding differences as bugs automatically.
2. Add an optional `distilbert/distilgpt2` adapter at an immutable revision for a small pretrained-model integration check. Start with correct cached versus full-prefix behavior; this does not imply a library bug exists. Keep CPU memory bounded and download tests optional.
3. Add a second architecture or an upstream minimal reproduction only if there is an actual observed issue worth investigating.
4. Explore compiled/quantized execution with carefully defined semantics as a new scope document.

Primary references to consult as needed, verifying APIs against the installed version:

- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch module and hook APIs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [DistilGPT2 model card](https://huggingface.co/distilbert/distilgpt2)

## Start now

Inspect the project directory and environment. Create SPEC.md, CLAUDE.md, and PROGRESS.md from this prompt, then implement M1. Continue through the milestones, validating actual behavior and committing small working changes. Deliver a working local tool, tested reproduction commands, and an honest generated report. Never substitute a polished mockup for the numerical engine or manufactured metrics for measurements.
