# 2026-09-28 — Pylint review of BP, Torch VMC, MPO, and PEPO

- Scope: user's approval to review remaining Pylint findings, starting with
  these four domains, and fix concrete problems with focused validation.
- Branch / baseline: `develop` / `f410fa0`. All earlier uncommitted work,
  including the [DMRG readability pass](2026-09-28-dmrg-stages-readability.md),
  is preserved.
- Commit status: working-tree changes; nothing staged, committed, or pushed.

## Fixes

- [Torch convergence estimates](../src/pepsy/vmc/torch/results.py): `tau`
  referenced the nonexistent `integrated_autorrelation_time`. It now returns
  `integrated_autocorrelation_time`. The existing convergence test checks
  `tau`, `tau_max`, and `rhat` against their owning fields.
- [BP Metropolis sampler](../src/pepsy/vmc/torch/sampler.py): investigating
  the override-signature warning exposed two problems. Multiple sweeps
  reported only the last sweep's proposal/acceptance counts; they now sum
  across the whole call, matching the parent sampler. The inherited
  `track_proposal_stats` option now raises a descriptive `ValueError` when
  enabled: local exchange/hopping counters do not describe whole-configuration
  BP proposals. The error occurs before sampling or RNG mutation.
- Removed a redundant `coefficient = coefficient` statement from the
  [MPO input parser](../src/pepsy/operators/mpo_semantic.py).
- Updated the [VMC API guide](../docs/api/vmc.md) and
  [changelog](../CHANGELOG.md). No dependencies, numerical acceptance formulas,
  source modules, lint suppressions, or CI selections changed.

One focused regression was added to the existing
[proposal-adaptation suite](../tests/test_vmc_importance.py). It compares
three BP sweeps against three individual calls with a deterministic proposal,
checks rejected proposals, final configurations/amplitudes/probabilities and
RNG state, verifies `burn_in` totals, and checks unsupported local-move
diagnostics through the sweep, burn-in, sample, and tuning entry points.
It remains outside the default smoke selection.

## Review findings

Fresh standard Pylint, using `--rcfile=/dev/null --persistent=n` and JSON2
output, covered `bp/`, `vmc/torch/`, `optimizers/mpo/`, and operator modules
matching `mpo*.py` and `pepo*.py`:

| Scope | Before | After |
| --- | ---: | ---: |
| Total diagnostics | 1,806 | 1,802 |
| Error-labelled | 72 | 71 |
| Warnings | 300 | 298 |
| Refactor suggestions | 1,086 | 1,085 |
| Conventions | 348 | 348 |

Both runs exit 30. This is not a clean standard-Pylint result. Counts are
comparable between these two identical scoped commands, not directly with the
earlier package-wide audit. Temporary raw results are in
`/tmp/pepsy-pylint-domains-{before,after}.json`.

Reviewed remaining patterns:

| Finding | Evidence / decision |
| --- | --- |
| Missing exports and lazy imports | All 25 flagged export/import names resolve at runtime. Preserve namespace behavior. |
| Missing MPO methods | The runtime class factory combines the dense-order mixin with Quimb's MPO class. The reported inherited members and indexing exist. |
| Non-callable Torch FFT/linalg functions; missing NumPy `finfo` attributes | The five reported Torch functions are callable; `eps` and `max` exist. No eager-import workaround. |
| Optional CuPy imports | Three findings require unavailable CuPy/CuPyX. Preserve optional imports; these paths remain unvalidated here. |
| BP `best` tuple indexing | Relay counts are validated positive; the first loop iteration assigns `best` before tuple use. |
| MPO replay cache indexing | The cache is a scoped `WeakKeyDictionary`; access is guarded against `None`. |
| MPO coefficient-batch shape indexing | Shape presence and rank are checked before indexing. |
| PEPO `left`/`right` initialization | The symmetric eigenvalue branch returns before the SVD factors are read. |
| PEPO model shape attribute | `__post_init__` validates and installs square arrays before `local_dim` uses their shape. |
| Proposal vmap flag initialization | The forced path is called by the boundary subclass, which initializes the flag. Base forward calls do not force it. |
| Mutable compiler default / loop closures | Boundary manifests intentionally bind per-window data and are read by retained compiler closures. MPO append and Torch sort closures execute within the current iteration. |
| PEPO list mutation during iteration | These are intentional breadth-first queues; iterating over a snapshot would omit descendants. |
| PEPO generator `StopIteration` warning | `next(iter(common))` follows `len(common) == 1`; the set is nonempty. |
| PEPO mixed return warning | The local tree builder returns a trace-sector certificate for exact fixed histories and `None` for the rank-capped path; the caller explicitly distinguishes them. |
| Fermion driver constructor/signature warnings | Initialization reaches `super().__init__` through the setup helper; the BP factory intentionally derives symmetry/sector/encoding from its PEPS metadata. |

The review prioritizes runtime errors and suspicious control flow. Naming,
module-size, argument-count, unused-name, and broad-exception suggestions have
not all been individually resolved. No exhaustive semantic audit or
package-wide clean-lint claim is made.

## Validation

- Before source fixes, the two regressions failed at the misspelled `tau`
  accessor and the incorrect BP multi-sweep counter (`2` instead of `6`).
- Initial convergence/proposal-adaptation suites after fixes: **8 passed**.
- Expanded domain suites: **535 passed, 28 warnings in 60.02s**, exit 0.
  Command: `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''`
  with `test_vmc_*.py`, `test_optimize_mpo.py`, `test_mpo_*.py`,
  `test_bp_*.py`, `test_cluster_expansion.py`, and `test_pepo_*.py`.
- Ruff, the two CI mypy targets, all 61 local Markdown link targets in
  changed guides/handoffs, and `git diff --check` pass. Targeted Pylint checks
  for undefined/uninitialized variables and self-assignment pass in the three
  changed source modules.
- Default smoke: **89 passed, 2 compatibility warnings in 19.21s**, exit 0.
- Full combined working-tree suite:
  `MPLBACKEND=Agg python -m pytest -q -ra -o addopts=''` →
  **5,169 passed, 129 skipped, 790 warnings in 512.79s (8m32s)**, exit 0.
  The skips require unavailable CuPy/CUDA/Metal, multiple MPI processes,
  or an explicit two-device JAX configuration. They do not validate those
  paths. Warnings remain visible; no warning filters were added.

No contraction, compression, backend dispatch, charge-sector, or upstream
compatibility behavior changed. GPU and distributed coverage remain limited
by the local environment.
