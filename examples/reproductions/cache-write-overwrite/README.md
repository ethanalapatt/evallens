# Reproduction: case_7287c8c752e22ccf

> **This is a deliberately injected fault.** It was introduced by EvalLens as test
> material for its own detector. It is not a bug discovered in PyTorch or any other
> third-party library.

A reference implementation and a candidate implementation, given identical weights and
identical inputs, produce different outputs. Nothing crashes and the shapes agree.

**What differs:** During incremental decoding the candidate writes each new key/value pair over the most recent cache slot instead of extending the cache, so one earlier position is permanently lost. Shapes stay valid and nothing raises; only the numbers change.

**Recorded result:** `FAIL`, max absolute error
`2.641718e-01` against the policy `atol=1e-05,
rtol=0.0001`. Failing request(s): r0.

## Run it

Requires Python 3.11+, PyTorch, and NumPy. Nothing else — in particular, **not** EvalLens.

```bash
python repro.py --expect-mismatch   # exits 0 only if the recorded mismatch reproduces
python repro.py                     # exits 1 if the mismatch is present, 0 if it is not
python repro.py --json              # machine-readable
```

Exit codes are distinct on purpose, so a script that fails to start is never mistaken for a
mismatch that reproduced:

| Code | Meaning |
|---|---|
| 0 | As expected |
| 1 | Default mode found the mismatch |
| 2 | Setup error (missing inputs, wrong hashes, or EvalLens leaked into the run) |
| 3 | Execution error |
| 4 | `--expect-mismatch` was requested and the mismatch did not reproduce |

## How this input was found

The original failing case had 14 valid tokens across 1 request(s). EvalLens reduced it to 2 token(s) across 1 request(s) — a 7.0x token reduction — using 4 predicate queries. Minimality: `one_minimal_wrt_declared_operations`.

This input is the output of the actual reducer. It was not handwritten.

## Where the divergence first becomes visible

earliest observed divergence at r0/block0/pos1/attn_out; 8/18 aligned checkpoints diverge (discrepancy reconverges later; not monotone)

That is evidence about where a difference becomes observable at the checkpoints these
adapters expose. It is not proof of root cause.

## What is in here

| Path | Contents |
|---|---|
| `repro.py` | The runner. Imports only the vendored sources next to it. |
| `repro_fixture/` | The fixture and adapter sources, copied from the EvalLens build that produced this package. Same code, not a re-implementation. |
| `case.json` | The exact input: tokens, prefill boundary, padding, execution mode. |
| `weights.npz` | Deterministic float32 weights, no pickled objects. Verified by SHA-256 before use. |
| `policy.json` | The numerical tolerance policy in force. |
| `manifest.json` | Hashes, adapter identities, source commit, and the recorded result. |

## Provenance

| Field | Value |
|---|---|
| EvalLens version | `0.1.0` |
| Source commit | `2e9f800a2bf4e0927e2af684b6eabcd56e561793` |
| Source clean | `False` |
| Original case | `case_605dd27327c54e4e` |
| Reduced case | `case_7287c8c752e22ccf` |
| Model config | `tiny-2L-64d-4h-89a0eb72a852` |
| Weights SHA-256 | `e954a43b36aaeb31595503cc0c0b1aec430e920feddac0d02388bd0dd23f8530` |
| Exported | `2026-09-11T21:34:14+00:00` |
| Produced on | macOS-15.6-arm64-arm-64bit-Mach-O, Python 3.13.7, torch 2.14.0 |

The model is a small transformer with **random, untrained weights**. It demonstrates
execution correctness, not language capability.
