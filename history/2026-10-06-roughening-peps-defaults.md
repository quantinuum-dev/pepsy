# 2026-10-06 — Roughening full update uses PepsOptimizer defaults

- Scope: align `run_roughening.py` full-update/sweep mode with current
  PepsOptimizer defaults, keeping simple update separate.
- Branches / baselines: examples `main` at `fbdacf1`; Pepsy `develop` at
  `7cb1044`. All changes are uncommitted; nothing staged or published.
- Existing solver, plotting, notebook, and dataset changes were preserved.

## Implemented

Changed the examples [PEPS adapter](../../pepsy_examples/experiments/mps_magnetization/benchmark/magnetization/engines/peps.py),
its [tests](../../pepsy_examples/experiments/mps_magnetization/benchmark/tests/test_peps_optimizer.py),
and runner documentation. The compatibility launcher continues to select full
update with `--mode peps --peps-optimizer-mode sweep`.

- Automatic gate batching replaces the adapter's single-gate default;
  positive integer overrides remain available.
- Boundary selection and local solver settings now inherit PepsOptimizer
  defaults. Omitted maxeval/round-trip controls are forwarded as no override
  (currently 50 local evaluations and four round trips). Explicit flags still
  override just their setting. Existing cap resolution matches the package's
  independent `(4*D, 5*D)` defaults.
- Removed forced target normalization: unitary targets retain their scale and
  use the package's known-target-norm policy. Automatic cutoff values are
  passed through to Pepsy; concrete values remain in diagnostic summaries.
- Full-update identities record `peps_optimizer_policy=2` to prevent reuse of
  results under the previous policy, even with explicit numerical overrides.
  Simple-update behavior and historical identities are unchanged.
- Shared runner Torch registration is retained. No Pepsy algorithm or backend
  implementation changed. This adopts the current public API; no shim added.

## Fresh validation

Activated the shared Python 3.12 environment, selected local Pepsy source,
and used one BLAS/OpenMP thread for the tests. Installed versions/signatures
were inspected: Pepsy 0.5.0, Quimb 1.15.1.dev79+gb5e316200,
Autoray 0.11.1.dev9+g1291702f9, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev11+g1a3481803, NumPy 2.5.2, Torch 2.6.0+cu124,
NLopt 2.11.0.

- Examples `tests/test_peps_optimizer.py`: **25 passed**, including a direct
  default-PepsOptimizer comparison with actual compression/refinement,
  NumPy/Torch automatic batching, explicit overrides, saved records, and
  sweep-child launches. Six complex64 NLopt runtime warnings returned the
  best iterate; cap, normalization, and acceptance assertions passed.
- Pepsy `tests/test_optimize_peps.py tests/test_peps_optimizer_batching.py`:
  **179 passed**. These used the existing uncommitted gradient-solver edits.
- Examples `tests/test_peps.py tests/test_entrypoints.py
  tests/test_roughening_sweep.py tests/test_run.py tests/test_roughening.py`:
  **402 passed, 4 failed**. Failures were
  `test_bubble_defaults_use_dmrg_quality_tracking_and_mps_z` and
  `test_bubble_dmrg_modes_complete_complex64_long_range_smoke[fit|dmrg|dmrg1]`.
  All four were reproduced in a fresh process loading the original adapter
  from examples HEAD without changing working-tree files. They concern MPS
  `dmrg` requiring block size one and removed `dmrg1`, outside this PEPS task.
- Changed Python files pass Ruff; benchmark-wide Ruff still reports six
  existing E731/E402 issues in effective-model source/tests.
- Local links in the two edited READMEs and `git diff --check` passed.

No full package suite, GPU validation, or production simulation was run.
The broader runner suite is not clean; the four baseline failures remain.

## Follow-up: clarify SU versus sweep

The user requested clearer mode naming. CLI help now groups SU gauge flags
and full-update sweep overrides separately in both launchers. The READMEs
show the two explicit selections side by side, explain that bare `--mode peps`
defaults to SU, and distinguish parameter scans in `run_roughening_sweep.py`
from the PEPS sweep evolution algorithm. Algorithms and defaults are unchanged
by this clarification.

New checks: five existing incompatible-control/SU-identity tests passed;
both launchers' `--help` output was inspected. Changed-file Ruff, README local
links, and diff whitespace checks passed. Earlier numerical validation and
baseline failures above remain the relevant evidence; no commit or push.
