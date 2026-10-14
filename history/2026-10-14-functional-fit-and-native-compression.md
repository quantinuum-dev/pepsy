# 2026-10-14 — Functional JAX FIT and native BdyMPS compression

- Scope: implement a functional FIT interface with implicit differentiation,
  fix the retained-degenerate reconstruction case, and enable native
  Symmray BdyMPS direct/zipup/dm compression.
- Branch / baseline: `develop` / `8f8695b`.
- Commit status: working-tree edits only; this session did not stage, commit
  or publish. Earlier and concurrently changing NetKet/TDVP files were preserved.

## Changes and limits

- Added `pepsy.fitting.make_jax_fit` / `FunctionalFIT`: fresh packed-parameter
  FIT evaluations, checkpointed finite-sweep AD, dense gauge-aligned implicit
  AD through one cycle and public JAX GMRES, checked forward/adjoint residuals.
  Native implicit charts remain explicitly rejected. Fixed ranks/positive
  charts and first-order gradients are required; no physical convergence claim.
- Added composed JAX reconstruction and paired splits. The formerly expected
  failure now passes through the physical reconstruction operation. Raw
  factor-wise SVD behavior is preserved; arbitrary vector gauges and native
  spectral degeneracies are not universally repaired.
- QR VJP chart selection depends on primal factors, preserving transposable
  linear pullbacks. Flat bosonic FIT uses supported native contraction modes.
- BdyMPS direct/zipup/dm now accepts native arrays without conversion; tests
  include genuinely truncated native fermionic JIT directional gradients.
- Public docs and changelog updated. Implementation and upstream decisions:
  [dated review](../docs/development/notes/2026-10-14-functional-fit-and-native-compression.md).

## Fresh validation

Activated the existing Python 3.12 environment; JAX numerical checks used CPU.
No dependency changes, GPU benchmarks or simulations were launched.

- New functional/projector/native-boundary tests plus QR/SVD stability:
  **96 passed**, 108.56 s.
- Backends, SVD derivatives, existing FIT AD, Torch projector, public API and
  package layout: **212 passed**, 39.65 s.
- FIT sum/gate/hotpath/kernel and boundary growth/engine regressions:
  **238 passed**, 131.54 s.
- Default smoke: **94 passed**, 34.09 s. Seven local Markdown link targets,
  whole-source/test Ruff and whitespace checks passed.
- Counts overlap and must not be added. The two-site implicit case was added
  after the above 96-case selection: final functional FIT module
  **11 passed**, 66.55 s, including one-/two-site implicit finite differences.
- No full 9,000-test numerical suite was run: validation stayed scoped to
  the requested work, and separate NetKet files changed concurrently.

Earlier failures in exploratory checks were corrected or capability-gated:
callable hashing, sparse-only flat contraction mode, cotangent-dependent QR
branches, and native implicit upstream limitations. No failing supported
numerical selection remains.
