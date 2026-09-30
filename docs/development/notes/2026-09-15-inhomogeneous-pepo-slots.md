# Inhomogeneous fixed-channel PEPO slots

## Scope

`PauliPEPOTerm.where` adds explicit finite site and nearest-neighbour edge
locations. `direction` additionally identifies a physical bond occurrence on
small periodic tori where endpoints alone are ambiguous. Homogeneous orders
one through nine retain their translated-cluster implementation.

The localized implementation now supports OBC and PBC through order nine. It
enumerates connected finite-lattice site subsets rather than translated shape
orbits. For each subset it retains every internal positive-direction bond
occurrence, forms the actual ordered local product, contracts the completed
lower-order active PEPO with zero boundary sectors, and factorizes the
remaining residual. This is occurrence-aware by construction; independent
coefficients are never replaced by a translation representative. C4 quotient
reuse therefore remains rejected for localized inputs.

The exact default uses a deterministic spanning tree rooted to minimize its
largest branch. Non-root tensors are fixed Pauli-history selectors and only
the root contains residual coefficients. The ranks and topology depend on the
cluster geometry, not parameter values, so Torch/JAX differentiation does not
cross an SVD gauge. If `max_tree_rank` is smaller than the exact history rank,
the existing backend SVD supplies the requested truncation.

Occurrence-specific histories use globally unique ids in
`ActivePEPOBlocks`. During `to_pepo()`, those ids are remapped independently
on every physical bond. This preserves exact channel isolation without
padding all site tensors to the total network-wide sector count. The opt-out
`compact_bonds=False` retains the old global-axis representation for
diagnostics.

The remaining cost is local PEPO width, not global id padding. A complex128
smoke build with one localized slot gave the following materialization
estimates: 3x3 order three, about 25.7 MiB with maximum local bond 25; 3x3
exact order four, about 19 GiB with maximum local bond 137. With
`max_tree_rank=2`, the latter estimate fell to about 161 MiB. Thus order two
or three is the practical optimization default; exact order four and above
should normally remain active-only, use an explicit rank cap, or stay on the
scalar-cluster endpoint.

## Upstream audit (2026-09-15)

The active editable environment reported:

- Quimb `1.15.1.dev55+gd0591eb70`
- Autoray `0.11.1.dev3+g1b476b305`
- Cotengra `0.8.3.dev7+g1d7fd333f`
- Symmray `0.3.2.dev8+g6c6dd34b5`

The installed signatures of `TensorNetwork2D.contract_boundary`,
`TensorNetwork2D.contract_ctmrg`, `PEPO.trace`, and `Tensor.trace` were
inspected in the same environment. The public upstream references were also
checked: the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html),
and [Symmray repository](https://github.com/jcmgray/symmray).

Classification: **adopt existing APIs**. The local PEPO builder uses existing
Autoray operations and Quimb materialization only. Quimb's current explicit
`PEPO(..., cyclic=...)` contract and its fix keeping open/periodic legs
distinct for dimensions one and two are sufficient for per-occurrence PBC
materialization. No compatibility shim, new upstream algorithm, native
Symmray conversion rule, or Cotengra optimizer default is introduced. CTMRG
remains routed through Pepsy's existing `contract_flat` compatibility
boundary.

## Validation

- Default Pepsy suite: 130 passed.
- Complete cluster-expansion integration file: 62 passed.
- Dense closure: 1x3 order three, 2x2 PBC order four with parallel directed
  bonds, and 1x5 order five agreed with full matrix exponentials.
- Torch value/gradient tests cover the exact fixed-history route and the
  explicit rank-capped SVD route. A JAX order-three smoke produced finite
  gradients.
- Changed-file Ruff checks passed. The Sphinx build could not start because
  the active environment does not contain the optional `autoapi` extension.
