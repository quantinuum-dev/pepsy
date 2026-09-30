# 2026-09-29 — Roughening runner / MPS optimizer alignment audit

- Scope: user requested a careful review of the Pepsy Examples roughening
  runners against current Pepsy MPS development, especially `mode="dmrg"`.
- Pepsy branch / baseline: `develop` / `a233a40`.
- Examples branch / baseline: `main` / `efb7696`.
- Commit status: this audit record is uncommitted. No implementation changes,
  commits, pushes, or production-job changes were made. Existing operator/API
  edits in Pepsy and the roughening notebook edit in Examples were preserved.

## Scope and conclusion

Reviewed `run_roughening.py`, `run_roughening_sweep.py`, their canonical
`magnetization.runners` implementations, and the shared MPS engine under
`pepsy_examples/experiments/mps_magnetization/benchmark/`. The shared replay
helper is also used by equilibrium, bubble, and Kibble--Zurek frontends.
The projected effective-model ground-state solvers were outside this audit.

The core DMRG integration works with the current MPS API. The constructor's
positional `MpsOptimizer(psi, chi, ...)` form remains supported. Gates are
prepared on the selected backend, installed once, and replayed once per
Trotter step. Named DMRG schedules survive configuration resolution.
Sampling refreshes its state at each depth, and canonicalization/entropy
measurements operate on private copies. The default target is layered on the
dense paths tested, with an isolated SRC guess and exact `target_cutoff=0`.

Fresh imports, both with and without explicit `PYTHONPATH`, resolved to this
checkout's `pepsy/src/pepsy/__init__.py`. Installed distribution metadata says
`pepsy==0.4.0`, while the repository declares 0.5.0; that stale metadata does
not mean these tests executed the older implementation.

## Findings

1. **Sweep resume omits numerical configuration checks.**
   `magnetization/runners/roughening_sweep.py:289` compares chi, angle values,
   depth, lattice, and mode, but omits DMRG settings and other physical/runtime
   controls. A temporary manifest accepted changes to `n_iter`, `cutoff`,
   `fit_init_strategy`, `stabilize_unitary`, `dt`, backend, and sample count.
   The completed-child check also lacks those expected values. Reusing an
   output root after changing them can skip old results and combine settings
   within a sweep. This does not establish that existing runs are mixed.

2. **The sibling-source import fallback is incorrect after the runner move.**
   `magnetization/engines/shared.py:57` computes
   `pepsy_examples/pepsy/src`, which does not exist, instead of the sibling
   `pepsy/src`. The existing editable environment/PYTHONPATH masks it here.
   A different environment can load an unintended installed Pepsy or fail to
   import. Fix the root calculation or use an explicit installation contract.

3. **CLI coverage has drifted.**
   The shared parser accepts `--mode su`, but current `MpsOptimizer` rejects
   it with `ValueError: Unknown mode: su`. Its mode/guess lists omit Pepsy's
   `sdc`, `sdc-oversample`, `sdcr`, and `sdcr-oversample` families, including
   the `guess-*` forms relevant to DMRG. The parser also lacks switches for
   the current optional `finite_check` and `fit_overlap_diagnostics` controls.
   Core `dmrg`/`dmrg1`/`dmrg2`/`dmrg3` execution remains available.

4. **Per-depth diagnostics miss the compact polling API.**
   `magnetization/engines/shared.py:1356` calls `norm_diagnostics()` with its
   full-history default, then uses only the scalar fidelity. Current Pepsy
   supports `include_history=False` with incremental summaries. Reconstructing
   the full history every depth adds growing work; no production speedup was
   measured in this audit.

5. **Provenance is collected at finalization.**
   `magnetization/runners/roughening.py:3387` reads the current checkout's
   commit and dirty state after evolution. A checkout updated during a long
   run can therefore be attributed to the revision present at completion,
   rather than the source imported by that process. Capture launch provenance
   separately, and retain any end-of-run change report.

## Default differences, not API failures

| Control | Roughening | Current Pepsy `run()` |
| --- | --- | --- |
| `n_iter` | 8 | 8 |
| `fit_rtol` | `auto` | `auto` |
| Output cutoff / cutoff mode | `auto` / `auto` | `auto` / `auto` |
| Target cutoff / strategy | 0 / `auto` | 0 / `auto` |
| Effective dense initial guess | `guess-src` | `guess-src` |
| `stabilize_unitary` | True | False |
| `fit_single_pair_fast_path` | True | False (named dmrg2 has its own exception) |
| `fit_init_rand_strength` | 0.1 | 0.0 |

Stabilization restores working norms while the compression-survival ledger
continues to accumulate loss. The fast path allows an adjacent pair to finish
after one effective update. Random strength applies to explicit random guess
strategies; it does not add 0.1 noise to the default `guess-src` policy.
These are supported explicit choices, but the runner is not identical to
calling Pepsy with only its package defaults.

`mps_fidelity` is correctly labeled as cumulative compression survival. It
does not establish global state fidelity or convergence. `--reference-exact`
provides a separately evolved state-overlap reference for small systems.

## New validation

Environment: shared Python 3.12, local Pepsy source, NumPy 2.5.2,
Torch 2.6.0+cu124, Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f.
Validation processes used one CPU thread and hid CUDA to avoid competing
with the user's existing jobs.

- Examples: `test_entrypoints.py`, `test_run.py`, `test_roughening.py`,
  `test_roughening_sweep.py`, `test_runner_layout.py`, and `test_storage.py`:
  **256 passed, 2 skipped**, in 32.15 seconds. GPU/CuPy paths were skipped;
  two existing deprecation warnings concerned `build_contraction` and
  `ham_tn.build_mpo`.
- Pepsy: `test_mps_compression_modes.py`, `test_mps_normalization.py`,
  `test_mps_fit_kernels.py`, and `test_mps_audit_fixes.py`:
  **286 passed**, in 8.27 seconds. Warnings concerned explicit diagnostic
  checks and compressor policies.
- Eight end-to-end roughening probes: NumPy and Torch CPU, complex128,
  `dmrg`/`dmrg1`/`dmrg2`/`dmrg3`, 3x3, chi=16, three dt=0.05 steps,
  32 samples, entropy and exact local Z enabled. Each made exactly three
  DMRG replay calls and matched the exact reference with maximum observed
  infidelity **1.16e-11**. The last FIT record in each case reported layered
  targets, SRC guesses, the adjacent-pair fast path, and no fallback.
- Four deliberately truncated 3x3 chi=2, six-step probes exercised NumPy and
  Torch CPU with stabilization on/off. They completed with finite diagnostics;
  exact overlaps were about 0.99511–0.99571 while compression-survival values
  were about 0.99866–0.99905. Stabilized norms were one, and unstabilized norms
  retained compression loss. These runs did not establish cross-backend
  equality under strong truncation.

Temporary evidence is in `/tmp/roughening-alignment-audit-20260929.json`,
`/tmp/roughening-truncated-audit-20260929.json`,
`/tmp/roughening-alignment-tests-20260929.log`, and
`/tmp/roughening-pepsy-mps-tests-20260929.log`.
The substantive results are retained above because temporary files can vanish.

## Limits and suggested follow-up

No full-package suite or new CUDA numerical test was run. Small-lattice
agreement does not establish accuracy/convergence of the production 5x6
sweeps. Correct sweep compatibility and import resolution first, then update
the stale CLI choices and adopt compact fidelity polling. Keep any numerical
default changes explicit rather than silently changing ongoing experiments.
