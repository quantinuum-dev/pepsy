# 2026-09-29 — Fix remaining native cluster MPO failures

- Scope: the user's request to fix remaining issues after the joint-product
  maturity review; maintain Pepsy tensor-network ownership and Gaugy's public
  downstream boundary.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: Pepsy implementation, tests and docs are uncommitted working
  tree changes alongside the earlier authorized cluster work. No push.

## Changes and evidence

Charge-sector residual SVDs and assembled virtual flux fix conserved hopping
and zero-cutoff null channels for both single and joint native MPOs. Further
tests found and fixed repeated-physical-charge ordering in native dense
export. Native live tensor backends fail explicitly without host conversion.

See the [implementation and compatibility evidence](../docs/development/notes/2026-09-29-native-cluster-charges.md)
and [API contract](../docs/api/operators/mpo_cluster.md#native-block-sparse-cluster-mpos).
Final validation: 316 Pepsy tests and 137 downstream Gaugy tests passed,
including 16 CUDA cases in total. Source/test Ruff and changed-document
checks passed. These are focused suites, not a full-repository claim.

## Remaining boundaries

Native NumPy direct construction is now checked for U1, Z2, U1U1 and Z2Z2.
Native cluster autodiff, native streaming/recursive assembly, graded
fermionic clusters and changing-rank derivatives remain unsupported.
GPU correctness on the tested small systems does not establish large-system
performance. Earlier dated failure records are preserved; the status ledger
points to this correction.
