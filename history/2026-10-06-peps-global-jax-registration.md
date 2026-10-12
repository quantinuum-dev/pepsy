# 2026-10-06 — Global JAX SVD registration

- Scope: enable Pepsy's existing JAX SVD derivative registration before global
  optimization/JIT tracing.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: working-tree edits only. Nothing staged, committed, or
  pushed; earlier work and unrelated solver changes preserved.

Added a global backend setup dispatcher that invokes
`register_jax_linalg(stabilized=True)` for JAX and reuses the unchanged Torch
setup otherwise. Both primary optimizer routes and fallback call it before
constructing/tracing Quimb's loss. JAX QR remains native.

Updated routing and numerical regressions, API documentation, and changelog.
See [detailed evidence and limits](../docs/development/notes/2026-10-06-peps-global-jax-registration.md).

Validation: combined focused selection **230 passed**; final PEPS routing
file **136 passed**; selected JAX backend checks **8 passed** (overlapping
selections, not an additive total). Ruff, local links, and diff whitespace
checks passed. No full repository suite or GPU run; native JAX symmetry
and degeneracy robustness remain unverified.
