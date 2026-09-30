# 2026-09-24 — Exact-batch replay aligned with Pepsy develop

- Scope: commit the MPS exact-batch work, pull current Pepsy `develop`, and
  reconcile the implementation with incoming MPS changes.
- Branch / baseline commit: `develop`, remote `28d9b6c`.
- Commit status: committed on local `develop` in this change; not pushed.

## What changed

- Added opt-in structured dense replay and focused tests, described in the
  [exact-batch audit](../docs/development/notes/mps_exact_batch.md).
- Rebased onto eight newer remote commits. Kept the newer shared MPS rebuild
  path, which preserves backend, device, physical indices, and site tags for
  both exact modes. Updated the [API guide](../docs/api/optimizers/mps.md) and
  [MPS skill](../.github/skills/mps-optimizer/SKILL.md) accordingly.
- Preserved the separate, uncommitted tree-layout work in the working tree;
  it is outside this commit.

## Validation

- Exact-batch suite: 22 passed.
- Public API and package layout: 58 passed.
- Exact-batch, dynamic controls, Quimb compatibility, public API, and package
  layout: 128 passed, 1 failed. The JAX complex64 Kraus-probability precision
  failure occurs on an isolated archive of remote `28d9b6c` too.
- Remote MPS backend exact-reconstruction selection: 9 passed, 3 JAX
  complex64 precision failures. The `jax-direct` case reproduces on that
  isolated remote baseline with the same values; the other two have the same
  amplitude rounding pattern.
- Ruff, skill quick validator, skill catalog validator, and whitespace checks
  passed. The full suite was not run for this focused integration.

## Findings and remaining work

The installed Quimb, Autoray, Cotengra, and Symmray versions and inspected
callable signatures did not change with the Pepsy pull. The upstream review
and algorithm benchmarks are in the [exact-batch audit](../docs/development/notes/mps_exact_batch.md).
The inherited JAX precision failures remain separate from the exact-batch
route. No remote push was requested or performed.
