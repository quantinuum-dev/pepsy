# 2026-10-05 — MPS trajectory corrections and dependency audit

Implemented in the working tree on `develop`, baseline `bec773a`. Scope is
the four findings in the [trajectory re-review](../../../history/2026-10-05-mps-trajectory-rereview.md).
No dependency, environment, or installed-library changes were made.

## Implementation

- Coalesced measurement probabilities use the ordinary exact-to-MPS preparation
  and tracked-center boundary. Exact reconstruction explicitly uses zero cutoff;
  a forced branch with probability `1e-40` survives reconstruction and normalizes.
- Dense Kraus probabilities contract projected amplitudes rather than a reduced
  density matrix with `K†K`. Adjacent supports use a mixed-canonical block;
  nonadjacent supports move together using untruncated swaps on a private MPS.
  The block fuses physical legs in the declared operator order. Memory grows
  with the support and routed bonds, not exponentially with the intervening
  physical span. Swaps may increase intermediate bond dimensions and cost;
  this work makes no performance claim. Exact modes use their full amplitudes.
- Real-valued generated NumPy constants retain a real state dtype. Genuinely
  complex operators on real non-NumPy states produce an actionable error;
  there is no discarded imaginary part or implicit live-state promotion.
  Native Symmray payloads retain their existing conversion boundary.
- The existing tree JAX precision context now lives in the backend conversion
  module and remains available at its old private import path. MPS replay,
  canonicalization, normalization, exact reconstruction, readout and probability
  evaluation reuse it. Guarding only probabilities left errors inherited from
  earlier reduced-precision canonicalization. The scope uses public
  `jax.default_matmul_precision("highest")`, preserves array dtype/device,
  restores the caller's setting on exit, and remains local to the calling
  thread. Optional JAX is imported only for actual JAX arrays.

For `psi = |++> + 1e-10 |-->` and the two-site projector onto `|-->`, the
old Gram path reported `5.551115123125783e-17`; projected amplitudes return
`1.0000012757041735e-20`, matching application to the actual dense state.
No global relative-error guarantee is inferred for arbitrary ill-conditioned
inputs; the change removes the extra cancellation from forming Gram expectations.

## Upstream audit

Used the existing activated `~/envs/py312` environment. Installed versions:
Quimb `1.15.1.dev79+gb5e316200`, Autoray `0.11.1.dev9+g1291702f9`,
Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev11+g1a3481803`,
NumPy `2.5.2`, Torch `2.6.0+cu124`, JAX `0.10.2`, CuPy `14.1.1`.
JAX uses a CUDA device with x64 disabled in the default GPU checks.

Read the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray repository](https://github.com/jcmgray/symmray), and
[JAX precision documentation](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html).
The requested [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable; used the official repository and installed source instead.
Latest documentation version banners are not treated as installed versions.

Inspected installed signatures for `MatrixProductState.swap_site_to`,
`gate_with_auto_swap`, `gate_nonlocal`, `from_dense`, `normalize`,
`local_expectation_canonical`, `compute_local_expectation`, and `Tensor.gate`.
The public swap operation forwards `max_bond`, cutoff and canonical `info`;
Tensor fusion retains the supplied physical-index order. `from_dense` accepts
split options, including explicit zero cutoff. The environment selector uses
`route` in this installed Quimb.

Decision: **adopt** these public Quimb and JAX APIs and the existing shared
backend converter. **Defer** dependency upgrades, native symmetry algorithm
changes and performance tuning. No upstream internal implementation was copied.

## Validation and limits

Commands use `python -m pytest -q -ra -o addopts=''` with one BLAS/OpenMP thread
and JAX GPU preallocation disabled where applicable.

- New review regressions plus public API/package layout: **90 passed**.
  The 36 new cases cover exact controls, rare postselection, direct/dmrg2/svd/
  exact/exact-batch Kraus weights, nonlocal/reversed supports, real Torch gates,
  and restoration of caller JAX precision. On an isolated archive of original
  `bec773a`, the new file gives **25 failed, 11 passed**, without local edits.
- Final native/control selection (`test_mps_fermions`, `test_mps_controls`,
  `test_mps_normalization`, `test_mpi`, and the new file), excluding slow cases:
  **319 passed, 30 deselected**. Native spinful fermionic MPO comparisons ran.
- Trajectory, importance, density-reference, dynamic-control, tree-successive,
  backend and import-boundary selection: **337 passed, 1 failed, 1 skipped**.
  The failure is the tree `sdcr-jax-2` projection comparison, outside MPS
  trajectory replay; the same test fails on the isolated original archive
  with a relative discrepancy of approximately `2.17e-4`. The skip is an
  unconfigured second JAX device.
- Separate two-logical-CPU-device JAX check: **3 passed**, including nondefault
  device placement and both new precision-scope cases.
- Final GPU MPS backend file: **47 passed, 3 failed, 1 skipped**. The remaining
  failures are `backend_ledger` for `jax-True`, `cuda-False`, `cuda-True`.
  Metal is unavailable. The earlier JAX probability, reconstruction, random
  FIT and warm-control accuracy failures now pass at the default caller setting.
- A broader domain run reached **533 passed, 1 skipped** before being
  interrupted in optional large-lattice slow Symmray cases. The final non-slow
  selection above completed; the interrupted run is not a full-domain pass.
- Full-package attempt with `--maxfail=5`: **84 passed, 5 failed, 1 skipped**,
  stopping at BP convergence checks. All five failures reproduce on the original
  archive: sequential projected messages, sequential topology refresh,
  simultaneous boundary snapshot, simultaneous initializer candidates, and
  parallel simultaneous PEPS sweep in `test_bp_compression.py`. MPI integration
  cannot collect because `mpi4py` is absent. No full-suite success is claimed.
- Ruff across `src tests` and `git diff --check` pass. No multi-rank MPI or
  performance benchmark was run.

The three strict cross-backend ledger comparisons remain unchanged. Errors
against the CPU reference are about `3.7e-6` to `4.1e-6` for a `3e-6` tolerance.
A separate dense complex128 readout of each backend's actual input/output
states shows each step's recorded loss agrees with its own norm ratio within
`5.7e-7`. This supports accumulated complex64 compression differences rather
than a large ledger arithmetic error. It does not justify silently relaxing
the tests or changing simulation precision to force agreement. The diagnostic
policy and tolerances were left intact.

Temporary evidence is under `/tmp/pepsy-trajectory-fix-*.log`,
`/tmp/pepsy-trajectory-public-repros.py`, and `/tmp/pepsy-ledger-audit.py`.
Important results are retained here because those files may expire.
