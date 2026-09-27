# 2026-09-26 — Joint PEPO products: trace storage and backend accuracy

This supports Gaugy's local exponential-product PEPO evaluator. The existing
Pepsy joint residual definition and physical operator convention are unchanged.

## Installed API audit

Environment: Pepsy 0.5.0, Quimb 1.15.1.dev66+ge927f06e1, Cotengra
0.8.3.dev7+g1d7fd333f, Autoray 0.11.1.dev3+g1b476b305, Symmray
0.4.1.dev8+gc45f91457, Torch 2.9.1, JAX 0.8.2. Inspected installed
`PauliPEPOBasis.compile`, `PEPOClusterProductExpansion.from_bases`, and
`contract_flat` signatures; no dependency upgrades.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The requested Symmray
array documentation page returned an internal error; installed APIs/source
were used instead. Quimb documents future cutoff-default changes; downstream
comparisons use explicit cutoffs and the installed signature.

- **Adopt:** existing joint cluster products, fixed histories, backend dtype
  metadata, bounded batches of equal-size located cluster targets, and early
  physical trace. No separate full-lattice factor PEPO multiplication.
- **Compatibility shim:** reuse existing scoped Torch and cyclic CTMRG
  compatibility contexts downstream; no new installed-library patch.
- **Defer:** native graded trace, whole-objective compilation, variational
  compression, and general accuracy claims for high-order large lattices.

## Implemented and measured

`ActivePEPOBlocks.to_trace_network()` returns an unnormalized bosonic 2D
network, tracing each physical block before dense allocation. Zero histories
remain structural channels. Backend graphs and values are preserved. Dense
trace storage is smaller by physical_dim squared; estimates use Python
integers and never NumPy-convert differentiable Torch blocks. Estimates omit
construction, contraction workspace, and backward storage.

For Torch 2.9.1, a scalar 2×2 exponential at angle 0.04377333 had approximately
1.31e-10 error against SciPy; a genuine batch of two gave approximately
7e-18. Located products now batch up to eight equal-size embeddings, matching
the existing homogeneous strategy. This removed a downstream truncated-PBC
gradient discrepancy (1.50e-9) at its original strict tolerance. It is measured
backend behavior, not a proof that all matrix-exponential regimes are exact.

New regressions cover analytic onsite values/derivatives, Torch float32/64
block-storage metadata, trace-before/after-materialization values/gradients,
JAX x64 PBC trace gradients, and integer-overflow-safe storage estimation.
The float32 metadata fixture casts already-built blocks; it does not claim
full float32 located-builder coverage.

Focused cluster/ordered-product/API/layout validation: **125 passed**, eight
existing compatibility warnings. `ruff check src tests` and diff checks pass.
Downstream Gaugy: **503 passed**, plus its unchanged default reference
notebook and OBC/PBC 4×4 order-2 directional checks. Exact contraction errors
were below 7e-10. At chi=16/cutoff=1e-12, MPS gradients disagreed with finite
differences and CTMRG gradients were nonfinite despite close forward values.
These boundary-gradient limitations remain; exact contraction is the new
downstream default. No full Pepsy suite or native-symmetry validation is
claimed for this dense-only change.
