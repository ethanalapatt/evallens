# Working conventions for this repository

Read `SPEC.md` for the technical contract and `PROGRESS.md` for actual state. This file is
about *how* to work here.

## Environment

```bash
.venv/bin/python -m pytest -m 'not mps and not download'
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src/evallens
.venv/bin/evallens doctor
```

The project-local `.venv` is authoritative. Never modify system Python or shell config.
`requirements-lock.txt` records the environment the measured numbers came from.

CPU float32 is the reference platform. MPS is visible on this machine and is *not* used by
the primary path. CUDA is never assumed.

## Non-negotiables

**Never fabricate a measurement.** No estimated speedups, invented detection rates, copied
benchmark numbers, or hand-edited result tables. Synthetic timing fixtures are allowed only
in tests, must be labeled, and must never produce a file named `RESULTS.md`. Before real
measurements exist, the README says `Results not yet measured`.

**Injected faults are injected.** Every mutant is a deliberate fault the project wrote
itself. Say so in the README, reports, viewer, and any resume material. An external library
mismatch is an observation to investigate, never automatically "a PyTorch bug".

**The search policy is blind.** `evallens.generate` and `evallens.reduce` must never import
`bench.mutants`, read a `Behavior`, or otherwise see fault labels, source patches, trigger
fixtures, or expected answers. They reach models only through the `Adapter` protocol. There
is a test that enforces this; do not weaken it.

**Verdicts stay disjoint.** A crash is `ERROR`, not a detection. A classification that
changes on replay is `UNSTABLE`, not a failure. An input-contract violation is `INVALID`.
Never collapse these to make a number look better.

**Never loosen a tolerance to make a faulty candidate pass.** Calibration happens on
known-good comparisons only, and the policy is frozen before held-out evaluation.

**Never delete a slow or awkward result after seeing it.** Non-qualified mutant variants stay
visible as incomplete. A generator or reducer that loses to its baseline gets published as a
loss.

## Code style

- Full type annotations; `mypy --strict` over `src/evallens` must stay clean.
- Docstrings explain *why* a design choice was made, especially where a simpler-looking
  alternative would have been wrong (fused attention kernels, forward hooks, shared masking
  helpers between an oracle and the code it checks).
- Tests assert real behavior. No test whose only purpose is to raise a count.
- Don't scaffold empty modules. A module appears when it does work.

## Milestone rhythm

1. Implement.
2. Run the four commands above; all must pass.
3. Update `PROGRESS.md` with actual commands, actual output, evidence paths, limitations,
   and the exact next action. Status values are `not started`, `in progress`, `blocked`,
   `complete` — used truthfully. Files existing is not a completion gate.
4. Write the milestone teach-back note.
5. Commit, then push to `origin`.

## Git

- Small, focused commits. Test results go in `PROGRESS.md`, not in commit messages.
- **Commit messages carry no AI attribution.** No `Co-Authored-By`, no `Generated with`
  trailers. This is the repository owner's work.
- Push the working branch after each completed milestone. Never force-push, rewrite history,
  or delete branches.
- Ignore environments, downloaded weights, caches, and oversized traces. Preserve compact
  final benchmark evidence and a few real reproduction artifacts with hashes.
