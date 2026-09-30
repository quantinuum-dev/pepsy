# 2026-09-29 — Trace the constructed cluster PEPO

The requested observable is the trace of the returned PEPO, including its
rank cap and compression. The former scalar subset closure computed the
uncompressed partition expansion independently of materialization.

## Implementation and compatibility

`trace_pepo` belongs to Pepsy. Square/graph active blocks close their actual
physical matrices and contract matching virtual sectors through sparse joins.
The existing fixed-Pauli trace-sector certificate removes only structurally
forbidden histories. NumPy numerical zeros may be omitted; tensor-backend
zeros retain their derivatives. Cached join plans contain integer indices
and sector keys, never tensor values or autodiff graphs. Sparse budgets
raise on excessive work rather than dropping terms. No global dense matrix
is needed. Dense site tensors are also unnecessary for this default route.

Materialized square PEPOs and graph networks use public Quimb `trace` with
explicit physical-index pairs and caller contraction options. Builder
`trace_exp` calls construction before this measurement. Requested frozen
compression or supported Quimb compression is therefore included. The
explicit `partition_trace_exp` name retains the previous scalar shortcut.
MPO `trace_exp` and Gaugy's legacy connected-log engine are unchanged.

The upstream/version audit from the same active task is reused:
[installed versions and official review](2026-09-29-factor-reuse-graph-autodiff.md).
The environment is unchanged. Also inspected installed Quimb
`PEPO.trace(self, left_inds=None, right_inds=None, **contract_opts)` and
`TensorNetwork.trace`: they close physical index pairs on a reindexed copy
and contract the resulting network. **Adopt** those public operations and
Pepsy's existing backend scatter-add helper. No compatibility shim, upstream
patch, or dependency change. **Defer** new GPU performance, full Torch
compiler capture of trace, and native-symmetry sparse trace guarantees.

## Periodic construction discrepancy

Measuring the actual operator exposed a pre-existing uniform builder defect
on short periodic axes. Infinite-lattice shape channels could alias sites
and miss parallel physical bond occurrences. On a two-by-two torus with
onsite X=0.2, edge ZZ=0.3 and step=-0.1j, order-two actual PEPO trace was
15.9296652061 while the finite partition reference was 15.8722038218.
Independent dense contraction confirmed the old PEPO value: this was not a
sparse contraction error. The old scalar trace bypassed that discrepancy.

When a periodic axis length is at most the spatial cutoff, construction and
compiler preparation now select the existing finite located-cluster route,
even for uniform terms. That route retains all directed bond occurrences
and places each residual once on its physical support, with verified reuse.
Two-by-two periodic C4/non-C4 tests check the actual PEPO, its trace, and the
full-cutoff dense exponential. The existing asymmetric YZ Gaugy periodic
regression passes at its original tolerance. No numerical tolerance was
relaxed; the old generic-cache assertion now checks finite geometry/reuse.

## Scope and limits

Frozen projections differentiate the projected operator, not selected ranks.
`trace_exp` constructs afresh; use `trace_pepo(pepo)` to measure an existing
or subsequently modified network. Sparse contraction can still be costly,
especially with backend zeros retained; raising the budget is explicit.
Finite-lattice construction can have different bond/storage costs from the
old uniform channel route. New sparse-trace gradients are checked on CPU
Torch and JAX JIT, including zero parameters, against dense contractions of
the same compressed PEPO. These are scoped checks, not a large-system
performance or universal approximation-accuracy claim.

See the [handoff](../../../history/2026-09-29-pepo-operator-trace.md) for results.
