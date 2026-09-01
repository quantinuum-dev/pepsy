# GibbsMps implementation note

## 2026-09-01 compatibility audit

The first Gibbs purification route uses the installed public tensor-network
surfaces:

- Quimb `1.15.1.dev39+g369d09b9d`, with
  `MatrixProductState.gate_nonlocal_(G, where, dims=None, method="direct", ...)`
  for long-range physical gates and
  `MatrixProductState.partial_trace_to_mpo(keep, upper_ind_id="b{}", ...)`
  for ancilla tracing.
- Autoray `0.11.1.dev1+gc56f64427`, with `linalg.expm`, `multiply`, `kron`,
  `reshape`, and backend-aware array dispatch for generated gates.
- Cotengra `0.8.3.dev6+g08fe1a3a1`, used indirectly by Quimb contractions.
- Symmray `0.3.2.dev6+ga17699db6`; native Symmray is deferred for this first
  class because an interleaved ordinary MPS needs a charge-aware Bell-pair
  constructor and symmetry-preserving imaginary-time gates.

The compatibility decision is **adopt** for the ordinary dense/Autoray path:
`bell_ro_mps` uses Quimb's `MatrixProductState` constructor and
`apply_to_arrays`; term operators, exponential gates, MPS replay, and the
traced MPO all stay on the selected backend. Generated gate streams are passed
through the same converter before installation. The class uses `MpsOptimizer`'s
non-unitary path and deliberately does not request unitary stabilization or
overlap-fidelity diagnostics.

Deferred work includes multi-site gates for explicit string terms across gaps,
native Symmray purification, and observables that contract directly against
the purification without first producing an MPO.
