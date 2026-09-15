# Inhomogeneous fixed-channel PEPO slots

## Scope

`PauliPEPOTerm.where` adds explicit finite site and nearest-neighbour edge
locations. A localized order-one/two builder evaluates each physical site's
background and each physical edge's exact two-site residual. The same 16
Pauli sectors are reused on every bond; endpoint tensor data, not sector IDs,
carry the inhomogeneity. Homogeneous orders one through nine retain their
existing translated-cluster implementation.

The first localized implementation is deliberately open-boundary, has no C4
quotient, and rejects order greater than two. Extending it requires actual
per-embedding higher connected residuals; translation-orbit reuse would be
incorrect for independent coefficients.

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
Autoray operations and Quimb materialization only. No compatibility shim,
new upstream algorithm, native Symmray conversion rule, or Cotengra optimizer
default is introduced. CTMRG remains routed through Pepsy's existing
`contract_flat` compatibility boundary.
