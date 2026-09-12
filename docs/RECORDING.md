# Recording a demo — steps, and what is still missing

## Status: incomplete

**There is no screenshot, GIF, or video of a completed EvalLens run.** No screen-capture or
browser-automation tooling was available on the development machine — `asciinema`, `vhs`, and
`ttyrec` are all absent, and no browser extension was connected — so none was produced. None has
been simulated, mocked up, or described as if it existed.

What does exist is real:

- [`demo-transcript.txt`](demo-transcript.txt) — verbatim captured stdout of `evallens doctor`
  and `evallens demo`, unedited.
- `tests/integration/test_viewer_render.py` — executes the actual `viewer/viewer.js` over an
  actual `record.json` under a DOM shim and asserts on what the page contains. This verifies the
  JavaScript render path. It does **not** verify layout, CSS, or dark-mode appearance, because
  it is not a browser.

## What is unverified

The rendered viewer page has never been seen. Layout, styling, responsive behavior, dark mode,
and the file-picker interaction are unconfirmed by execution.

## Steps to record it

These commands are each individually tested; the recording tool invocations are not, because
the tools are not installed here.

### Terminal capture

```bash
brew install asciinema          # or: pipx install asciinema
cd /path/to/evallens
asciinema rec docs/demo.cast --command "bash -c '
  evallens doctor
  evallens demo --out artifacts/demo
'"
# optional GIF conversion
npx --yes svg-term-cli --cast docs/demo.cast --out docs/demo.svg --window
```

`vhs` is a good alternative if a deterministic, scripted recording is wanted:

```bash
brew install vhs
vhs docs/demo.tape        # tape file not written; see the vhs docs for its syntax
```

### Viewer capture

```bash
evallens demo --out artifacts/demo          # writes artifacts/demo/record.json
evallens view artifacts/demo --host 127.0.0.1
# then open http://127.0.0.1:8777 and capture the window
```

On macOS, `Cmd-Shift-5` records a region or window to `.mov`; convert with
`ffmpeg -i in.mov -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 docs/viewer.gif`.

### What the recording should show

1. `evallens doctor` — the environment and the fixture-vs-oracle self-test passing.
2. `evallens demo` — the six steps, with the injected-fault banner visible.
3. The viewer, scrolled through: the injected-fault banner, verdict, original versus reduced
   input, the earliest observed divergence table including a reconvergence, the reduction
   timeline, and the verified reproduction.
4. `python artifacts/demo/repro/repro.py --expect-mismatch` exiting 0 from a fresh shell.

## Rules for whoever records it

- Record an **actual** run. Do not stage, retouch, or re-time the output.
- The injected-fault banner must stay in frame. The fault in the demo is one this project wrote;
  a recording that crops that out misrepresents the result.
- If a step fails during recording, publish the failure or fix the bug. Do not re-roll seeds
  until the output looks good.
