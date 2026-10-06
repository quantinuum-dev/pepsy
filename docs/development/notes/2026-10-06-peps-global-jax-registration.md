# 2026-10-06 — Register JAX SVD before global PEPS optimization

User authorized automatic registration after confirming that only Torch had
an automatic linalg setup hook. Branch `develop`, baseline `8f7c896`;
working-tree changes only. This follows the
[JAX defaults change](2026-10-06-peps-global-jax-defaults.md).

## Implemented

The PEPS driver's backend setup now routes JAX to the existing public
`register_jax_linalg(stabilized=True)` before either optimizer entry point
constructs its Quimb TNOptimizer. The fallback path uses the same setup
dispatcher. Torch delegates to its unchanged configuration helper, including
its explicit policy, legacy opt-out, and native Symmray split-driver handling.

The JAX registration is idempotent, process-wide Autoray configuration.
It installs Pepsy's existing thin-SVD custom VJP, which restores truncated
cotangent shapes then delegates differentiation to JAX. QR remains native;
this is not Torch-style singular-value regularization. Registration happens
with or without JIT. No new backend kernels, dependency changes, or standalone
GlobalOptimizer behavior were introduced.

Classification: **adopt** the existing backend-owned registration API.
Reused the unchanged installed-environment and upstream audit from the
preceding JAX investigation/defaults task. Inspected registration dispatch,
the custom SVD forward/backward implementation, and both global entry paths.

## Validation scope

- Orchestration regressions assert registration precedes optimization for
  JAX, covers NLopt and the standard optimizer, respects explicit loss/JIT
  overrides, and handles a JAX-selected fallback without importing JAX in
  the mocked configuration tests. Existing Torch policy tests remain active.
- Real 3x3 complex128 RZZ integration cases start with native JAX SVD
  registration, then assert the custom SVD is installed before TNOptimizer
  creation. They verify directional gradients against finite differences,
  optimization improvement, unit output norm, preserved input/backend/dtype,
  and retained bond cap for both NumPy and JAX input arrays. The fixture
  restores the previous process-wide SVD dispatch and x64 setting afterward.
- Existing JAX backend checks cover thin-SVD reconstruction, JIT gradients,
  real/complex truncated cotangents, registration switching/idempotence,
  and restoring native registration.

Fresh results: **230 passed** in the combined JAX integration, PEPS/global,
global safeguards, public API, and package-layout selection (86.10 seconds).
After adding standard-optimizer and JAX-fallback routing cases, the complete
PEPS orchestration file passed **136 tests**. The selected JAX backend suite
passed **8 tests**, with 63 unrelated backend tests deselected. These counts
overlap; they are separate runs, not additive suite totals. Ruff over `src`
and `tests`, local documentation links, and `git diff --check` passed.
Warnings include existing NumPy shape deprecations, compatibility aliases,
Quimb split defaults, and Torch rank-deficient QR diagnostics. No failures.

The new JAX integration checks are dense CPU tests. Native Symmray/JAX,
GPU, and degenerate-singular-value robustness are not established by them.
The existing native Torch checks cover preservation of the unchanged Torch
path. Explicit JAX exponent stripping retains the limitation documented in
the previous review.
