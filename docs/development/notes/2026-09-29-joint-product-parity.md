# 2026-09-29 — Joint exponential capability review

Scope: compare connected-cluster `exp(A) @ exp(B) @ exp(C)` with single
exponentials in Pepsy MPO/PEPO and Gaugy. This reviews the local working tree,
not a released version or every higher-order Taylor/history MPO API.

## Assessment

The dense joint construction has the same core mathematical and numerical
controls as its single-factor cluster route. It is validated for the tested
CPU workflows. It is not an unrestricted production-readiness claim for
native charge sectors, adaptive-rank gradients, large lattices or GPU use.

| Capability | MPO clusters | PEPO clusters | Gaugy facade |
| --- | --- | --- | --- |
| Complete local generators, algebraic product order | Joint engine | Joint engine | Public Pepsy handoff; legacy Pauli order reversed once |
| Geometry from all factors | Shared plan or `graph="interactions"` | Shared plan; direct square basis covers the lattice | Shared plan / square geometry |
| Single-factor / identity-factor limit | Same engine | Checked through cutoff four, open and periodic | Shared bound operator API |
| Runtime term vectors, defaults, scales, step | Live values | Live values | Resolved once into fresh bound values |
| Whole-product and factor reuse | Verified graph reuse; interval target reuse | Verified reuse and equal-size batches | Preserves aliases; graph reuse switch now exposed |
| Torch/JAX gradients | Fixed dense paths; rank-changing SVD limits | Fixed dense paths and frozen projections | Public forwarding and local Pauli paths |
| Operator construction | Semantic or Quimb MPO | Active blocks or square/graph network | Explicit `to_pepo` |
| Compression and reports | Rank/assembly policies, compiled `return_report` | Rank caps, frozen plans, supported Quimb compression, work counts | Bound preparation, construction and work reports |
| Trace | Explicit uncompressed scalar endpoint; trace returned MPO to include truncation | Constructs and traces selected PEPO | Same PEPO trace; separate partition/connected-log endpoints |

The `compile_exp` contract caches topology, not runtime tensor graphs. Legacy
single-PEPO `tau`/`beta` wrappers and MPO history/Taylor `exp_batch` are separate
conveniences/algorithms, not evidence of a missing physical joint operation;
the joint cluster API uses `step` and one runtime binding set per evaluation.

## Corrections

1. Direct joint PEPO `exp(..., max_bond=...)` silently ignored the keyword
   unless `compress=True`. It now raises before numerical work. The
   incompatible `compress=True, materialize=False` combination also raises
   before resolving callbacks. Explicit frozen compression remains separate.
2. Gaugy's graph facade omitted Pepsy's `spatial_reuse` control. It now passes
   that option through its public builder and every cached binding mode.
   No tensor-network implementation moved into Gaugy.

## Independent evidence

New Pepsy tests use three generators with different edges `(0,1)`, `(1,2)`
and `(0,2)`. The inferred union is the triangle. At cutoff three they compare
the complete ordered matrix with SciPy exponentials; at cutoff two they
independently enumerate singleton backgrounds and the three pair residuals.
Direct and recursive graph MPOs, graph PEPOs and routed square PEPOs agree.
All term, factor-scale and step derivatives are compared with independent
finite differences at two parameter sets, including zero factors/coefficients.
No global dense operator is used in the package implementation; dense
matrices here are deliberately small test references.

Single PEPO versus identity-extended joint checks cover open/periodic cutoffs
one through four. Separate Gaugy tests cover three-factor fixed compression,
actual returned-PEPO trace gradients, reports, live parameter refresh and
reuse enabled/disabled across direct, auto, graph and square factories.
Existing backend/JAX/compiler, binding, geometry and compression selections
are rerun; exact counts are in the handoff.

## Unresolved native U(1) findings

Native direct MPO assembly converts residual SVD channels to sparse virtual
blocks without assigning general nonzero virtual charges. The conserved
two-site hopping generator `sigma+_0 sigma-_1 + sigma-_0 sigma+_1` fails the
charge-flow validation at materialization, both alone and between onsite-Z
exponentials. Adding term charge metadata does not supply charges for these
new residual channels. General sector-aware residual factorization and
virtual-charge propagation are required; they were not implemented here.

Even diagonal onsite-Z/ZZ products can fail with `cutoff=0`, which retains
null SVD channels with forbidden local entries. The existing default
`cutoff=1e-12` diagonal route passes an independent dense check. This is a
distinct tested policy, not a relaxed acceptance tolerance or a general fix.
Four regression cases deliberately assert the current materialization error;
their passing status does not mean native hopping or zero-cutoff support.
No xfail, silent dense fallback, charge-check bypass or error clipping was added.

Other limits remain: fixed/reference subspaces do not differentiate adaptive
rank decisions; graph MPO `auto` may use a reported bounded collection
approximation; sparse PEPO trace budgets may be exceeded; square routing can
increase bonds; GPU and large-lattice performance were not validated here.

## Dependencies and ownership

Installed versions were checked again and match the
[earlier upstream audit](2026-09-29-factor-reuse-graph-autodiff.md): Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124,
JAX 0.10.2. Reuse the audit for this continued maintenance task. This review
changes argument validation/forwarding, not upstream contraction/SVD code.
**Adopt** existing public Pepsy controls. **Defer** general sector-aware
residual channels; no shim or installed-library modification. Pepsy remains
independent of Gaugy, with downstream integration tests owned by Gaugy.
