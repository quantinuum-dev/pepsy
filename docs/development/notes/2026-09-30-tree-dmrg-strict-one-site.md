# 2026-09-30 — Enforce one-site Tree DMRG

The user requested that TreeOptimizer match MPS DMRG's strict one-site
contract. Generic `dmrg`, its `fit` alias, and deprecated `dmrg1` now reject
`fit_block_size=2` or `3`, with an error directing callers to `dmrg2` or
`dmrg3`. Valid one-site `dmrg1` still warns to migrate to `dmrg`.

Validation occurs at construction and before replay installs mode, queue,
RNG, or shot overrides. The effective block resolver also enforces the
restriction. Named block schedules retain their existing warm-up and
one-site handoff. This changes option validation, not TreeFIT kernels:
guess/target preparation can still decompose tensors and open bond support.
Standalone TreeFIT and the separate TreePepsOptimizer are unchanged.

Roughening rejects larger explicit blocks for both tree names during
configuration resolution, including sweep configuration, while preserving
package defaults and valid explicit controls. Notebook sources and saved
outputs were not changed in this follow-up.

This supersedes the explicit-larger-block allowance in the
[alias implementation](2026-09-30-tree-dmrg1-alias.md). Classification:
**adopt** the existing MPS option contract; **defer** unrelated upstream
changes. Reused the active-task
[upstream audit](2026-09-30-tree-dmrg-one-site-default.md#compatibility-audit).
Installed versions rechecked unchanged: Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev7+g83fb22865. Rechecked `TreeFIT.run_gate`: the existing explicit
`block_size` dispatch remains intact. No dependency or backend changes.

Validation results are recorded in the
[session handoff](../../../history/2026-09-30-tree-dmrg-strict-one-site.md).
