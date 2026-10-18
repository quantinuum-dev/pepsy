# 2026-10-18 — Review and validation of NetKet and JAX optimizer changes

- Scope: review the pending flat-Z2 TFIM NetKet path and chunked JAX Optax
  updates before committing the requested work.
- Branch / baseline: `develop`; baseline `7ec0a01` was already one commit ahead
  of `origin/develop`.
- Commit / push status at review: pending.

## Validation

- `tests/test_gradient_solver_jax.py`: 58 passed.
- Ruff and `git diff --check`: passed.
- Selected NetKet suites: 177 passed, 3 failed. Two flat-Z2 boundary cases
  fail because the installed JAX does not support `jnp.take(mode="raise")`
  while tracing Symmray charge selection. One dense-oracle underflow regression
  was fixed by restoring per-tensor norm equalization for dense PEPS, and its
  targeted regression test passed.
- The selected NetKet suites include CPU integration coverage; no GPU
  trajectory was run. The 2026-10-14 session record remains authoritative for
  the unresolved TDVP/QGT convergence issue.
