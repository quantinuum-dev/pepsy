# 2026-09-30 — PepsOptimizer SRC boundary initialization

- Scope: user requested SRC initialization and ten boundary-FIT iterations at
  the Pepsy level, followed by D² environment caps and dtype-aware automatic
  cutoffs/tolerances, following the roughening sweep integration.
- Branch / baseline: `develop` / `414798e`.
- Commit status: prepared for the user's requested commit and push; earlier
  working-tree status entries below describe their respective checkpoints.

## Changes

`PepsOptimizer`'s shared boundary defaults now include
`fit_init_strategy="guess-src"`. This is the existing API spelling for an SRC
guess followed by FIT refinement; `"src"` is a compression-mode selector.
The existing `n_iter=10` default is retained. Explicit direct arguments and
`boundary_kwargs` overrides keep their precedence. The policy reaches
normalization, infidelity evaluation, and delegated sweep environments.
Standalone boundary helpers and standalone SweepOptimizer are unchanged.

Updated the optimizer API guide, docstring, and changelog. Regression tests
check all three consumers and both override mechanisms. The downstream
roughening adapter inherits the package policy without another override.

## Validation

- `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
  -o addopts='' tests/test_optimize_peps.py`: **62 passed**.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Downstream `pepsy_examples` roughening `tests/test_peps_optimizer.py`, using
  local Pepsy and single-thread BLAS/OpenMP: **18 passed** (NumPy/Torch,
  dense references, actual refinement, saved output, and angle-sweep resume).
  Complex64 NLopt retains the existing best-parameter fallback warnings.
  No full package suite was run for this scoped default change.

Reused this session's upstream audit and rechecked the unchanged installed
versions: Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124. Inspected Quimb's public
`tensor_network_1d_compress` signature and Pepsy's existing seeded SRC guess
dispatch. Classification: adopt existing public functionality; no new shim.
Native Symmray and one-site boundary fallbacks remain in the existing worker.

## Follow-up: default environment cap

The user subsequently requested `boundary_chi=D*D` by default. Omitted
`PepsOptimizer.boundary_chi` now resolves to `chi ** 2`; its automatic
standalone normalization and evaluation caps therefore become `2 * chi ** 2`.
Explicit scalar/tuple boundary caps and explicit metric caps retain precedence.
Updated API docs, changelog, and regressions covering the values actually
forwarded to normalization, loss evaluation, and sweep construction. The
roughening adapter supplies an explicit cap and keeps that policy. Sweep
scheduling is unchanged. The cap-only follow-up passed all 66 optimizer tests.

## Follow-up: dtype-aware automatic tolerances

`PepsOptimizer.run` now defaults `cutoff`, `cutoff_mode`, and `infidelity_tol`
to `"auto"`. Reused the gate cutoff resolvers and shared MPS FIT tolerance
scale: complex128 uses cutoff 1e-12 / infidelity tolerance 1e-9, complex64 uses
1e-6 / 1e-5, and 16-bit inputs use 1e-3 for both. Automatic cutoff mode is
`rsum2`. These are inferred from native PEPS array dtype on each run, including
after `set_state`, without host conversion. The infidelity threshold controls
entry to refinement; it is not the MPS FIT stopping criterion itself.

Explicit settings retain precedence. Invalid scalar cutoffs/tolerances fail
before state normalization. The exact two-site targets remain untruncated;
the warm-start policy does not change their cutoff. Gate/batch records save
resolved values. The roughening CLI now accepts/defaults its infidelity
threshold to `auto`, rather than imposing its previous fixed 1e-10 override.

Current package validation: **81 passed** in `tests/test_optimize_peps.py`,
including real NumPy/Torch complex64/complex128 compression, threshold decisions,
state dtype replacement, override routing, and invalid input checks. Full
package Ruff and both repository diff checks passed. Changed downstream Python
files pass Ruff. The downstream selection passed 22 cases; its remaining
test expected NaN rejection in the validator, but the shared auto parser now
rejects it earlier. Updated that test to verify both parser and programmatic
validation rejection. The four relevant invalid-control cases then passed;
all 23 selected downstream cases are covered across these runs. No production
simulations were launched.

## Follow-up: equal D² caps and DMRG2 verification

The user requested automatic normalization and evaluation caps of D² as well.
They now resolve independently from `chi ** 2`, even when `boundary_chi` is
overridden. Defaults are also forwarded to the delegated sweep/global
normalization settings; explicit caps retain precedence. The roughening
adapter's three automatic optimizer caps now follow D², while its measurement
and sampling caps retain their separate controls. Updated output/resume tests
and documentation accordingly.

Confirmed target normalization occurs after constructing each untruncated
target and before compression/the infidelity check. The initial state is
normalized on the first run, independently of the infidelity threshold.
`fit_mode="dmrg2", boundary_engine="dmrg"` is an existing opt-in for boundary
DMRG2; the default remains one-site/effective FIT. Added a real DMRG2-boundary
sweep regression checking successful refinement, norm, bond cap, and input
state isolation. Current package result: **82 passed**; package Ruff and
changed downstream-file Ruff passed. The downstream optimizer, per-D batch
configuration, and angle-sweep/resume selection passed **22 tests**. Existing
complex64 NLopt best-parameter fallback warnings remain visible.

## Follow-up: norm/overlap cap pairs

The user superseded the D² policy with `(4*D, 5*D)` for all three optimizer
settings. `boundary_chi`, `normalize_chi`, and `evaluation_chi` now independently
resolve to that pair; explicit scalar and pair overrides are supported at
construction, setter, and run time. Pairs consistently mean norm/overlap caps.
Normalization uses only the first entry. `peps_infidelity` now contracts both
norms with the first entry and the overlap with the second, while supplied
norms still skip their contractions. Sweep diagnostics no longer promote both
caps to their maximum. Low-level metric defaults remain unchanged.

The roughening adapter follows the same policy, accepts comma-separated CLI
pairs, and canonicalizes them to lists in saved controls/resume comparisons.
Readout and sampling controls are unchanged. Reused the upstream audit from
this ongoing task; no dependency or environment changes were made.

Validation with local source and the same py312 environment:

- Optimizer, boundary metrics, public API, and package-layout suites:
  **337 passed, 1 failed** in 20.85 s. The failure is the pre-existing installed
  metadata version mismatch (`0.4.0` installed versus `0.5.0` in pyproject).
  Dense-reference tests exercise both metric backends, scaled/unscaled norms,
  separate cap routing, supplied norms, cached boundary retuning, and overrides.
- Extended the real cleanup regression to both `dmrg`/DMRG2 and `quimb-mps`:
  **2 passed**, including the new engine case, in 4.31 s.
- Downstream optimizer, per-D batch, and angle-resume selection: **22 passed**
  in 51.24 s. Additional/updated scalar and pair CLI/resume and invalid-cap
  cases: **8 passed** in 18.26 s.
- Full package Ruff, changed downstream Python Ruff, and both diff whitespace
  checks passed. These are focused checks, not a full repository test run.

All current optimizer work remains uncommitted in Pepsy `develop` and examples
`main`. No production simulations, dependency upgrades, commits, or pushes.

## Follow-up: repaired stale package metadata

The user requested fixing the packaging-version check. Inspection showed the
editable installation already had `pepsy-0.5.0.dist-info`, but obsolete ignored
`src/pepsy.egg-info` still declared 0.4.0. `PYTHONPATH=src` made that stale
metadata take precedence. This corrects the earlier shorthand description of
an installed-version mismatch: there were two competing metadata directories.

Refreshed the editable installation offline with
`python -m pip install --no-deps --no-build-isolation --no-index -e .` in py312.
The current setuptools editable build left the obsolete source egg-info
untouched, so moved that generated directory into a temporary backup outside
the import path. No source version or test assertion was changed and no
dependencies were upgraded. Runtime and distribution metadata now both report
0.5.0 with `PYTHONPATH=src`, and only one Pepsy distribution is discovered.

Package-layout and public-API checks now pass: **54 passed**, two existing
deprecation warnings, in 5.86 s. Existing optimizer changes remain uncommitted.

## Completed 3×3 roughening validation and publication preparation

The requested open-boundary 3×3, D=2, Δθ=0 run completed all 60 steps at
dt=0.1 through t=6, plus 4096 Z samples. It used Torch CPU complex128 and
one CPU thread explicitly approved by the user. All three optimizer cap
pairs were (8, 10), with the default four round trips per axis and maxeval=100.
Runtime was 4719.16 seconds. All 696 attempted refinements were accepted;
720 local fidelity records were saved, with no missing values or invalid
inner losses. D stayed at most two and the maximum dense norm² error was
6.66e-15. No production algorithm changes were made for this validation.

Final fidelity against an independently evolved dense version of the same
untruncated Trotter circuit was 0.02890206867. The product of the recorded
local fidelities was 0.3208499074. That product requires no additional boundary
contractions but is only a diagnostic proxy for accumulated state fidelity.
This run confirms completion and normalization, not adequate D=2 accuracy.
The dense reference does not remove Trotter error versus continuous time.

Temporary evidence is under
`/tmp/pepsy-3x3-d2-sweep-1thread-20260930-1aiqez4l/`: `report.md`,
`summary.json`, `fidelity_trace.csv`, `fidelity_comparison.png`, and the
standard runner outputs in `run/`. The manifest and sample summary both
report complete; the launcher exited zero. Earlier interrupted attempts
remain separate. The user then requested committing and pushing Pepsy;
the downstream example changes are outside this commit.
