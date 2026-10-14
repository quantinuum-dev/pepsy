# 2026-10-14 — Integrate Pepsy implicit FIT into NetKet DMRG

- Scope: use Pepsy's fixed-point FIT derivative in NetKet boundary DMRG.
- Branch: `develop`; HEAD observed during concurrent work: `a820f3c`.
- Commit status: this session did not stage, commit or publish changes.
  Earlier FIT/backend and concurrent NetKet/TDVP edits were preserved.

## Changes

- Public `compress_jax_fit` supports Quimb callable dispatch and reconstructs
  dense boundary fits through Pepsy `FunctionalFIT`.
- NetKet `dmrg`/`dmrg2` select one-/two-site Pepsy compression. Default `auto`
  uses checked implicit dense gradients; explicit `unrolled` selects finite
  sweeps. Native arrays retain their existing finite Quimb FIT path, and
  explicit native implicit requests raise.
- Batched QR pullbacks use primal selection and a nonsingular inactive chart
  so JAX's implicit linear solve can transpose them.
- API docs and changelog describe the dense default/convergence budget change.

## Validation

CPU JAX, existing Python 3.12 environment; no dependency changes. Focused
checks include actual truncated public-amplitude derivatives, native finite
derivatives, wrapper reconstruction/scales/convergence rejection, explicit
policy dispatch, and multiple configurations through repeated compression.
- FIT/compression/QR selection: **61 passed**, 326.47 s. Includes existing
  finite-sweep NetKet SR/TDVP integration and native flat derivative checks.
- Public API/layout and SVD/projector selection: **122 passed**, 45.92 s.
- Dispatch/native implicit rejection and batched repeated-compression
  derivative checks: **4 passed**, 20.31 s.
- Final callable reconstruction/scale/nonconvergence check: **1 passed**,
  14.94 s, after the private site-index format correction.
- Whole-source/test Ruff, whitespace and four local link targets passed.
- Default import smoke: **94 passed**, 29.59 s, with two existing
  compatibility-alias deprecation warnings.
- Selections overlap; counts are not additive. No full numerical suite or
  GPU benchmark was run.

## Remaining limitation

Native implicit FIT is not integrated. Independent complex derivative checks
failed for the temporary sector-alignment prototype; it was removed rather
than installed. The retained native path differentiates finite sweeps. See
the [dated evidence](../docs/development/notes/2026-10-14-netket-implicit-fit.md).
