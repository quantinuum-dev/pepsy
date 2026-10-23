# 2026-10-22 — Shared Drew wall postprocessing

Integrated one wall-metric implementation into the active eiHt-compress main
checkout and connected the diagonal-wall notebook and Pepsy examples to it.

- Scope: user approved bringing Drew's module to main, adding active-site
  support there, and replacing the examples' local pairing implementation.
- Baselines: eiHt-compress `main` at `88497d2`; examples `main` at `c3a0279`.
  Both already had substantial uncommitted edits, which were preserved.
- Source: `ising-dw-postprocessing` commit `b9cd03c`, available in the separate
  `eiHt-compress-ising-dw` worktree. That worktree remains unchanged.
- Final commit status: eiHt-compress `main` at `ce1bafe` and examples `main`
  at `88f8366`, following the user's explicit review-and-commit request.
  No push, installation or production simulation was performed.

## Implementation

- Added the source module, archive adapter, README and original tests under
  [ising_dw_postprocessing](../../eiHt-compress/ising_dw_postprocessing/README.md),
  with a normal package export. Notebook checkpoints and the separate example
  notebook were not copied.
- `snapshot_wall_metrics(..., site_coordinates=...)` accepts compact shots
  in an explicit logical order. Missing sites remove incident bonds and open
  adjacent dual plaquettes for boundary pairing. Existing vertex/edge indices
  preserve greedy ordering, edge-index BFS ties and chord roughness.
- The [diagonal-wall helper](../../eiHt-compress/diagonal_wall/helper.py)
  decodes Qiskit counts, delegates per-pattern metrics, and computes weighted
  shot means/standard errors. The notebook has a D_raw/Lmax readout cell with
  the existing N<16 local-simulation limit; supplied hardware counts need no
  local state vector. Original source outputs are preserved.
- The [examples cache adapter](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/mps_sample_wall_cache.py)
  now only maps samples and delegates to the shared module. Removed the local
  NetworkX pairing implementation. Cache schema 3 rejects earlier caches;
  no saved cache or data was regenerated in this session.
- The examples loader selects the main checkout by default and loads its file
  explicitly, avoiding a stale bare-module import from another checkout.
  `EIHT_COMPRESS_ROOT` overrides are respected. Historical fallback support
  remains for full rectangles; old modules receive an actionable error on
  removed-site requests. No rough plotting notebook source was changed.

## Validation

- Shared module and diagonal-wall tests: **87 passed**.
- Examples plotting, fidelity-reader and alignment tests: **164 passed**;
  after extending the retained-snapshot loader test to removed corners,
  its four full/removed and paired/unpaired cases also passed.
- Independently compared every metric against original `b9cd03c`: all 512
  configurations on 3×3 and 256 random 10×10 shots match exactly, including
  roughness and no-pair values.
- Temporary notebook copies executed for 2×3 (sampled D_raw/Lmax readout)
  and 5×5 (guarded hardware-count path). JSON/schema checked and all original
  cell outputs compared against the start-of-session snapshot: unchanged.
- Changed examples Python files pass Ruff; eiHt syntax/Pyflakes (`E9,F`)
  checks and both repositories' `git diff --check` pass. New source files
  were also checked for trailing whitespace.

## Limits and failures

- The unchanged `test_corrected_exact_default_finds_runtime_campaign` fails
  because existing local campaign data takes precedence over its temporary
  fixture. It was reported and excluded from the focused rerun. Two costly
  CLI evolution tests were deselected: this task changes postprocessing only.
- Full configured Ruff is not clean: 844 findings across the imported module,
  tests and diagonal-wall code under eiHt-compress's strict style policy;
  examples-wide Ruff reports its six existing effective-model findings.
  Broad formatting/annotation cleanup was not included in this integration.
- No full repository suite, GPU validation or hardware execution was run.
  Reported standard errors measure shot noise, not simulation truncation bias.

Restart an existing notebook kernel or rerun its setup to load the new helper.
Regenerate a wall cache explicitly when needed; older data remains valid but
schema-2 metric caches are intentionally stale.

## Final staged-source review

Validated exported Git-index contents, so unrelated working-tree plotting
changes could not affect the commit checks. Historical examples notebooks
and extrapolation edits remain uncommitted and were not altered by this review.

- eiHt-compress: **87 passed**; staged notebook schema valid; syntax/Pyflakes
  and whitespace checks pass.
- Examples alignment, plot helper, roughening, vector/wall diagnostics and
  entrypoints: **377 passed, 5 skipped, 4 failed**. Both complete seven-angle
  exact-batch/MPS smoke sweeps passed against the diagonal-wall circuit.
- The four bubble failures reproduce with unchanged examples `c3a0279` Python
  source. They concern existing incompatible block-size/mode options, including
  the removed `dmrg1` alias. See the
  [final examples review](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/2026-10-22-diagonal-wall-alignment.md#final-commit-review)
  for test names and scope.
- Sweep, collection and fidelity-reader tests: **94 passed, 9 deselected**;
  excluded tests concern historical notebook paths. Cache checks after fixing
  its rebuild-command message: **2 passed**. No full-suite success is claimed.
- Changed examples Python files pass Ruff; the six unrelated effective-model
  lint failures remain. Both repositories pass `git diff --check`.

The earlier two journal entries remain records of their pre-commit checks.
This handoff and those entries are included in a separate Pepsy documentation
commit on `develop`; Pepsy implementation source is unchanged.
