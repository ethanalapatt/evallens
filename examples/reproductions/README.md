# Real generated reproductions

Each directory here is an **actual output of `evallens demo`** — the reducer produced the
input, the exporter wrote the package, and the package was verified by running it from a
fresh temporary directory in an isolated interpreter. Nothing here was handwritten.

> Every fault reproduced here was injected by EvalLens on purpose, as test material for its
> own detector. None of them is a bug discovered in PyTorch or any other third-party library.

## Running one

Requires only Python 3.11+, PyTorch, and NumPy. In particular it does **not** require
EvalLens, and each `repro.py` checks that itself — if an `evallens` module is loaded it exits
with a setup error rather than reporting a result that came from somewhere else.

```bash
cd cache-write-overwrite
python repro.py --expect-mismatch   # exits 0 only if the recorded mismatch reproduces
python repro.py                     # exits 1 because the mismatch is present
```

## What is here

| Directory | Fault | Reduced input |
|---|---|---|
| `cache-write-overwrite/` | During incremental decoding, each new key/value pair overwrites the most recent cache slot instead of extending the cache, so one earlier position is permanently lost. | 2 tokens, reduced from 14 (7.0x) |

Regenerate with:

```bash
evallens demo --out artifacts/demo --config configs/cpu.toml
cp -R artifacts/demo/repro examples/reproductions/<name>
```
