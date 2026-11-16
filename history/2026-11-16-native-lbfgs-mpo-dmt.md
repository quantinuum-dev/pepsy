# 2026-11-16 — Native JAX L-BFGS and ordinary MPO DMT

- Scope: add `GradientOptimizer(solver="jax-lbfgs")`; retain protected SVD as
  the primary direct method and extend it to ordinary ket/bra MPO tensors.
- Branch / baseline: `develop`, `758e83f5`.
- Publication: the user authorized committing and pushing this feature on
  2026-11-16. This handoff accompanies the feature commit; Git records its
  final revision. Unrelated sampling, test-harness and journal edits remain
  outside the commit, including the separate sampling changelog entry.
- Used the repository maintainer and tensor-fitting skills. No tutorial or
  sibling repository was edited.

## Implemented

- Lazy Optax L-BFGS runner in `solvers._jax_lbfgs`: compiled update chunks
  containing zoom line search, scalar/real checks including search trials,
  real/imaginary coordinates, per-leaf dtype/device preservation, stopping,
  best restoration and objective evaluation accounting. Existing SciPy aliases
  retain their meaning. Fresh numerical state per call; compiled-function
  reuse across separate calls is not promised.
- `operators.compress_mpo_dmt`, `MPODMTReport`, `MPODMTBondReport` for dense,
  open, uniform square local dimensions. Shared kernel accepts a trace
  covector and independent trace-aligned protection vectors; no conversion
  of stored MPOs to Pauli or a dense global matrix. Default triples and
  explicit regions are supported. Physical cutoff/report units include
  external exponents; metadata and transactional writes are checked.
- The new trace-gradient oracle initially found native SVD backward NaNs.
  Removed redundant local protection vectors and validated through actual
  truncation using Pepsy's existing explicit stabilized Torch linalg policy.
  The projected residual's zero singular values still require that policy;
  no native or global backend policy was changed.
- Public docs, changelog, operator map and
  [dated audit/timings](../docs/development/notes/2026-11-16-native-lbfgs-mpo-dmt.md).
- Second review: preserve `linesearch_failed` when a failed zero fallback
  satisfies `ftol` with a nonzero gradient; prepare every writeback array
  before modifying MPO tensors. Added regressions for both and compiled
  `min_steps` stopping.
- Verified and documented a numerical limit: degenerate complex ordinary-basis
  cuts can break global Hermiticity, even when protected reduced operators
  remain unchanged. Real Pauli coefficients remain useful for preserving
  global Hermiticity; no global Hermiticity guarantee was added to this API.

## Validation and limits

- Complete focused rerun after the independent-basis correction:
  **408 passed, one pre-existing expected failure**, seven existing Torch QR
  warnings, 161.30 seconds. Includes all five Pauli DMT modules and the
  ordinary-MPO, new native L-BFGS and existing gradient-solver modules.
- After the final review fixes, reran the four affected solver/ordinary-MPO
  modules: **187 passed** in 36.99 seconds. This includes 47 tests in the two
  new files. Together with the unchanged Pauli regressions, the focused
  checks cover 412 passing cases and one expected failure across runs.
- The analytic Torch trace-gradient test passes through actual compression
  under the existing stabilized SVD/QR policy. The expected failure remains
  unresolved float32 auto trace classification on an ill-conditioned input.
- API/layout: **54 passed, one known installed-version mismatch** (`0.5.0`
  distribution versus `0.6.0` checkout).
- Full-suite attempt: **102 passed, 7 skipped, one pre-existing BP dispatch
  failure**, `test_loopunion_selects_both_plaquettes_and_clips_at_edge[where0-expected0]`.
  NumPy input reaches a Torch namespace in Autoray `fuse`; no task code is
  on that failure path. Installed Autoray is older than the declared minimum.
- Ruff (`src tests`), local Markdown-link checks and `git diff --check` passed.
- CPU only. Ordinary MPO DMT remains adaptive and host-controlled; native
  Symmray, cyclic MPOs, global constrained variational fitting, automatic
  2D gate replay and a generally smooth DMT derivative are not implemented.
