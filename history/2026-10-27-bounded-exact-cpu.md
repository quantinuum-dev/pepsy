# 2026-10-27 — Bounded Torch CPU exact replay

- Scope: assess and specialize the 6×6, two-corner-removed, Torch CPU
  `complex64` exact-batch roughening run; final sampling could be skipped if
  it obstructed evolution.
- Branch / baseline commit: `develop` at `e658bed` (one commit ahead of
  `origin/develop` before these edits).
- Commit status: working-tree edits only; nothing staged, committed, or
  published for this task. The sibling `pepsy_examples` tree already had
  unrelated work in progress, which was preserved.

## What changed

- Added opt-in bounded exact-batch replay with in-place Torch CPU complex64
  one-qubit and structured phase/parity Numba kernels. It fails on
  unsupported gate blocks and reuses the dense state across runs.
- Added bounded final Z sampling and its benchmark CLI integration. Final
  sampling was cheap in the 5×6 pilots, so it remains available; direct
  readout is measured separately.
- Documented the state ownership behavior, limitations, and measured evidence
  in [the numerical note](../docs/development/notes/2026-10-27-bounded-exact-cpu.md)
  and the owning API guides.

## How it was validated

- `pytest -q -o addopts='' tests/test_mps_exact_batch.py tests/test_sampler.py`
  → 148 passed, one preexisting Torch warning.
- `pytest -q -o addopts='' tests/test_public_api.py tests/test_package_layout.py`
  → 55 passed, two deprecation warnings.
- `ruff check src tests` and `git diff --check` → passed.
- In `pepsy_examples`, focused roughening tests → five passed;
  targeted Ruff and `git diff --check` → passed.
- After the direct readout change, benchmark `test_run.py`,
  `test_roughening.py`, and `test_vector_observables.py` → 280 passed,
  five optional GPU cases skipped. Changed-file Ruff passed; broader Ruff
  retained six unrelated existing errors in effective-model code/tests.
- 5×6 four-step in-place pilot → no state-sized peak after first step;
  final direct observables matched the prior deep-copy pilot within 3.2×10⁻⁹.
- 5×6 one-step, 16-worker pilot → 6.66-second evolution and 6.73 GiB peak;
  direct energy matched four-worker result within 1.6×10⁻¹⁰.
- Matched ordinary 5×6 Torch exact-batch at 16 threads → 8.98-second
  evolution and 8.73 GiB peak. Direct observables matched bounded replay
  within 5.7×10⁻⁸.
- 6×6 one-step, 16-worker pilot → 287-second engine stage, 384.73 GiB peak
  RSS, and final 64-shot Z sampling. Its optional direct readout was stopped
  before completion, so no final observable archive was produced.
- After optimizing direct CPU readout in the benchmark, a new 6×6 one-step
  pilot completed: 243-second evolution, 326-second final readout,
  384.73 GiB peak RSS, finalized energy and X/Z moments. On 5×6, the new
  readout took 4.61 seconds versus 23.29 seconds previously, with all
  observables within 1.3×10⁻⁸.

## Decisions / findings

- A 34-qubit dense `complex64` state uses 128 GiB. The first 6×6 step
  reached 384.73 GiB peak RSS. The four-worker version was interrupted
  after confirming this memory peak and seeing roughly nine seconds per
  gate. The 16-worker version completed evolution and final sampling.
- Old references to `optimizer.p` share the live dense data in this opt-in
  mode. Constructor input remains protected when `inplace=False`.
- A 60-step 6×6 angle would spend roughly three hours in evolution if later
  steps match the pilot's 177-second gate phase. This is an estimate; the
  full run was not timed.

## Suggested next step (subject to current task scope)

- Estimate full-depth runtime before launching any multi-angle production
  run. Keep direct readout at the requested final times, since one 6×6
  readout takes about 5.4 minutes.

## Open questions / blockers

- A full 60-step or seven-angle run has not been launched. The first pilot's
  interrupted output is incomplete; use the later finalized pilot for the
  one-step direct observables.
