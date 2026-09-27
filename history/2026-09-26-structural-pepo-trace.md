# 2026-09-26 — Structural PEPO trace sectors

- Scope: improve the joint cluster PEPO local-cost parameter/trace path and
  audit downstream MPS/CTMRG differentiation.
- Branch / baseline: `develop`, `83ae41e`.
- Commit status: this entry accompanies the local implementation commit;
  nothing published. Device-local override excluded.

## Changes and evidence

Located exact Pauli trees now certify identity-subtree sectors for complete
traces. Trace allocation removes only algebraically impossible histories;
the operator, lower-cluster subtraction, live zeros, backend, and gradients
are preserved. Rank-capped and uncertified builders retain all sectors.

See the [audit and measurements](../docs/development/notes/2026-09-26-structural-pepo-trace.md)
and updated [API guide](../docs/api/operators/cluster_expansion.md).
Storage/cluster suites: 70 passed; public API/layout: 58 passed; full Ruff
passes. The full Pepsy suite was not run for this localized operator change.

## Remaining limits

Downstream order-2 boundary gradients improve substantially, but higher-order
MPS and conditioned CTMRG still have measured biases. No universal autodiff
safety claim or default boundary-policy change was made.
