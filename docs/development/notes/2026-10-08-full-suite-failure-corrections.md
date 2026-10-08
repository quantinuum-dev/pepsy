# 2026-10-08 — Full-suite failure corrections

## Scope and status

The user requested fixes for the failures in the
[full recheck](2026-10-08-full-update-full-recheck.md). Baseline is `develop`
at `ad8ed05`; prior PEPS corrections and concurrent MPS/JAX edits are preserved.
All changes in this record are working-tree edits, not committed or published.
The final uninterrupted full run passed: **8,263 passed, 9 skipped**, zero
failures, 2,116 warnings, 2,726.62 s. Separate PEPS refinement edits appeared
late in the run; their five closest suites passed **132 tests** in a fresh
process. They are not claimed as changes made by this task.

## Implemented corrections

- **BP compatibility shim:** Quimb can report convergence when its pending
  queue empties, although its last sweep residual exceeds Pepsy's strict
  tolerance. Re-enter the public run method with the remaining iteration
  budget to confirm the residual. Keep local scheduling, user tolerance,
  explicit rolling stops, and total iteration accounting. Shared plain,
  relay, cluster, and series paths use the same helper. A tree reconstruction
  regression checks both exhausted and sufficient iteration budgets.
- **Adopt eager dispatch binding:** resolve Autoray transpose/einsum callables
  when binding compact channels, before Torch graph capture. This prevents
  lazy AutoNamespace cache mutation inside the compiled function while
  retaining backend dispatch and differentiability.
- **CTMRG compatibility shim:** contract all shared bond axes of native
  fermionic reduced factors with graded tensordot. Current Quimb can pass
  rank-three or higher unfused factors; the old adapter used matrix multiply.
  Keep the context scoped and preserve its existing zero-sector safeguards.
- **MPS readback:** use the boolean tensor's explicit scalar read at the
  deferred validation boundary, retaining the single-read replay contract
  and the separate traced Torch/JAX handling.

## Test assumptions corrected without relaxing tolerances

The test harness defaults JAX to highest matrix-product precision and disables
its large initial GPU memory reservation. Explicit caller environment settings
take precedence. No library default, array dtype, installed package, or shared
environment was changed. Reduced-precision GPU dot accumulation is unsuitable
for these numerical oracle tolerances, and preallocation caused contention
with the other backend checks in the earlier broad run.

The first broad validation exposed an additional test-order dependency:
NetKet's import installs a global JAX mesh on the default GPU. That makes later
qMERA and PEPS global tests with CPU-backed arrays fail with an incompatible
device/context-mesh error. A minimal import-NetKet/CPU-conjugate probe reproduced
the failure. An optional test fixture now snapshots and restores the mesh using
public, capability-checked `jax.sharding.get_mesh` / `set_mesh` APIs. It does
not import JAX for tests that do not use it or change application behavior.
See the [JAX sharding API](https://docs.jax.dev/en/latest/jax.sharding.html).

The MPS ledger parity test now supplies complex64 gates to its NumPy reference,
matching the device test. Previously those gates promoted only the reference
state to complex128. The original tolerance remains unchanged, and the test
now also checks that the CPU reference stays complex64.

The native CTMRG exactness test now permits the double-layer rank, D squared
(four), rather than the single-layer rank (two). Independent cap sweeps gave:

| cap | default gauging | layered gauging |
| --- | --- | --- |
| 2 | 1.075260633558835 | 1.0627012784810652 |
| 4 | 1.0760613053174823 | 1.0760613053174837 |
| 8 | 1.0760613053174823 | 1.0760613053174837 |
| 16 | 1.0760613053174823 | 1.0760613053174837 |

Exact contraction is 1.0760613053174832. Imaginary roundoff is below 1e-15.
The test retains its original accuracy tolerance; production truncation is
unchanged. The chi-two result is an approximation, not an exactness oracle.

## Upstream audit

Installed versions: Quimb `1.15.1.dev90+g6a3906cbe`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Symmray
`0.4.1.dev15+g0374aaa3c`, Torch `2.6.0+cu124`, JAX `0.10.2`.
The optional NetKet version is `3.22.3`.

Inspected installed `BeliefPropagationCommon.run`, `D2BP.iterate` and its
constructor, and `compute_oblique_projectors`. The BP loop can stop on
`npending == 0`; its D2 scheduler repopulates an empty queue on the next
iteration. The oblique-projector implementation explicitly supports multiple
unfused bond axes, with outer axes at the ends. The adapter follows that
public factor ordering without modifying installed source.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The requested Symmray abelian-array documentation page returned an internal
error; installed source and the official repository supplied the audit fallback.
No dependency upgrade is proposed or required by these changes.

## Fresh validation

- BP relay/open-series/native/compression suites: **134 passed**, one warning.
- Native symmetry, Quimb compatibility, BP relay/reduced-update/reference
  suites: **282 passed**, 16 warnings.
- Added BP iteration-budget/reference regression: **1 passed**.
- MPS ledger parity after correcting reference dtype: **10 passed**.
- NetKet module followed by all three qMERA JAX regressions: **16 passed**,
  confirming mesh isolation across the actual import/test sequence.
- Broad backend selection: **379 passed, one skipped**, and one failure from
  the old MPS reference dtype (corrected and covered by the ten-case rerun).
- Separately scheduled CPU exact U1U1 Hubbard case: **1 passed**, 102.59 s.
- Ruff and whitespace checks passed during implementation.
- Final full-suite run: **8,263 passed, 9 skipped**, zero failures, with every
  collected test included in one invocation. JUnit results match all **8,270**
  collected IDs: 8,263 passes and seven individual skips, plus two MPI modules
  skipped during collection because mpi4py is unavailable. No IDs are missing.
- Ruff, whitespace, and skill-catalog validation (12 skills) passed again.

The pre-mesh-isolation full run was intentionally interrupted after 4,111 passes,
five context-mesh failures, four skips (including two collection skips), and one
deselection. These results are diagnostic, not a completed full-suite claim.
A fresh full run validated the fixture correction, including the slow exact
Hubbard case again. Its artifacts are `/tmp/pepsy_fixes_final_full.{log,xml}`;
the coverage reconciliation is `/tmp/pepsy_fixes_final_summary.json`.

The seven individual skips are unavailable Metal, four checks requiring
additional logical CPU/JAX devices, one requiring two CUDA devices, and the
CuPy float32 subnormal-flushing case. These remain unvalidated configurations;
the passing result does not claim MPI or unavailable-device coverage.

During the final run, separate changes appeared in PEPS strip refinement and
its tests, carrying separate norm/overlap caps and calibrated handles. Those
edits are preserved. Since the full process could have imported earlier source,
the five closest PEPS suites were rerun in a fresh process against the late
working tree: **132 passed**, 33 warnings, no skips, 24.91 s. Source/test
hashes were unchanged during that run. Artifacts:
`/tmp/pepsy_fixes_late_peps.{log,xml}`. Final Ruff, whitespace, report-link,
and skill-catalog checks passed. No unresolved failure from the reported
recheck remains under the documented test configuration.

Intermediate runs launched before all fixes completed are diagnostic only.
Their failures are not the final validation result.

Artifacts also use `/tmp/pepsy_fixes_` prefixes: `bp`, `native`, `backend`,
`ledger`, `mesh`, `slow`, and `collection`.
