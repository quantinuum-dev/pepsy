# September 2026 — CI dependency-profile compatibility

## Evidence and scope

The local development suite passed before this investigation, while hosted CI
on `9d6d7ef` failed in core tests, MPI, and mypy. Reproducing the declared core
dependencies under the selected Python 3.12 interpreter, isolated with `-S`
and a temporary dependency directory, exposed **158 failures**. The first
repaired full core run passed **2858 tests**, with **1037 optional/capability
skips**. See the [session handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-24-ci-dependency-profiles.md)
for final validation and publication status.

| Package | Installed development environment | Isolated released dependencies |
| --- | --- | --- |
| Quimb | `1.15.1.dev66+ge927f06e1` | `1.15.0` |
| Cotengra | `0.8.3.dev7+g1d7fd333f` | `0.8.2` |
| Autoray | `0.11.1.dev3+g1b476b305` | `0.11.0` |
| Symmray | `0.4.1.dev8+gc45f91457` | absent in core; `0.4.0` in optional probes |
| NumPy | `2.5.2` | `2.5.3` |

The public [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
places SDC, the native MPO gate sandwich, and related improvements in its
unreleased section. Checked the [Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
The Symmray array documentation was unavailable. Installed source and public
signatures supplied the concrete capability checks below.

## Compatibility shims

- **Random seeds:** released Quimb SRC compressors do not explicitly accept
  `seed`; their catch-all options reach Cotengra, which rejects it. Reused
  MPS's existing legacy seeding policy in a shared helper for MPO, boundary,
  MPO-product warm starts, and tree-PEPS callers. Modern APIs receive a native
  seed. Legacy APIs run under one shared Quimb RNG lock. The exact target,
  requested compressor, cutoff, and rank limit remain unchanged.
- **Torch random compressors:** old SRC implementations allocate NumPy noise.
  Explicit non-NumPy SRC now fails with an upgrade/direct-compression message.
  Dense MPS/MPO FIT warm starts use direct compression with a warning on those
  builds, retaining the exact variational target. Torch SDCR similarly requires
  the dtype-aware Quimb random split API; older drivers mix real noise with
  complex Torch matrices. Supported development builds keep native random paths.
- **Single-precision tree edge compression:** Quimb 1.15.0's NumPy
  `svd:eig` path raises a Numba `TypingError` when its float literal promotes
  intermediate eigenvalues and produces inconsistent complex64/complex128
  return types. The development source uses dtype-preserving clipping.
  Pepsy probes the public split API once per affected dtype. A failed compile
  triggers a warning and direct SVD for that tree edge, preserving dtype and
  truncation settings. No upstream code or registrations are changed.
- **Trotter capability:** interacting `exp_trotter` calls now report a missing
  `get_trotter_gates` method explicitly. Gibbs preparation already checked it.
  Tests for the native scheduler require it; beta-zero and isolated onsite
  behavior remains covered on released Quimb.

## Test and CI corrections

- Missing optional backends cause explicit skips at the relevant test or
  parameter; dense cases in mixed suites continue to run. MPI CI explicitly
  installs Stim to execute stabilizer trajectories. The extended extra now
  includes Autograd for tests that explicitly select that backend.
- The MPI throughput benchmark warms one serial shot per rank before timed
  worker execution. A cold threaded run stalled in Stim/NumPy native import
  locks; the warmed two- and three-rank smoke runs both completed. This is a
  benchmark initialization policy, not a general cold-start concurrency fix.
- The MPS ledger test excludes upstream adaptive SVD rank selection from its
  scalar-read spy, just as it excludes FIT convergence. Integer rank allocation
  is an intentional host boundary, distinct from norm/fidelity bookkeeping.
- Python 3.10 uses a development-only `tomli` backport to read project metadata.
- Mypy runs with the Python 3.12 language target matching its runner and
  installed NumPy stubs. The Python 3.10 test job retains interpreter coverage.
- Smoke tests passed but collected only about 9% whole-package coverage.
  The smoke job now checks contracts directly; the extended job retains its
  existing 60% whole-package coverage gate.

Unavailable features are **deferred** to a capable upstream build, not
reimplemented. Capability tests also exercise clear rejection where practical.
These changes make no package-size or numerical-speedup claim.
