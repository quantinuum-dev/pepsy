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
- Quimb's `LocalHamGen.get_trotter_gates(x, order, steps, ordering,
  fuse_adjacent, alternate)` is used as the graph-aware product-formula
  scheduler. The purified evolution passes `x=-beta/(2 * n_steps)` and
  `steps=n_steps`, so the total physical-side exponent is `-beta / 2`.
- Cotengra `0.8.3.dev6+g08fe1a3a1`, used indirectly by Quimb contractions.
- Symmray `0.3.2.dev6+ga17699db6`; native Symmray is deferred for this first
  class because an interleaved ordinary MPS needs a charge-aware Bell-pair
  constructor and symmetry-preserving imaginary-time gates.

The installed Quimb `partial_trace_to_mpo` signature is
`(keep, upper_ind_id='b{}', rescale_sites=True)` and does not expose a
`contract_tags` keyword. Its implementation nevertheless performs the
reduction through local tag contractions. It also propagates the live MPS
global exponent to the returned MPO. `GibbsMps` therefore uses this native
reducer for both unscaled and Pepsy-rescaled states, keeping the ordinary
readout on Quimb's optimized path. If explicit contraction options are
supplied, it uses the equivalent native `contract_tags` and
`contract_cumulative` operations directly with `strip_exponent=True`, because
the installed public reducer does not expose those options. This preserves
scale-aware readout without ever densifying the purification.

The resolved graph-layer ordering is passed back to Quimb explicitly rather
than asking `LocalHamGen` to color the same interaction graph twice. This is
important for randomized ordering: the schedule retained in
`GibbsMps.trotter_layers` is now exactly the schedule used for replay. Stable
non-random orderings are cached on the Gibbs object because the Hamiltonian
topology is immutable while coefficients and imaginary time are rebound for
each preparation.

The compatibility decision is **adopt** for the ordinary dense/Autoray path:
`bell_to_mps` uses Quimb's `MatrixProductState` constructor and
`apply_to_arrays`; term operators, graph-combined local Hamiltonians,
exponential gates, MPS replay, and the traced MPO all stay on the selected
backend. One-site lifting is performed at the Pepsy boundary with Autoray
Kronecker products because the installed Quimb `H1` absorption helper builds
host identities. Generated gate streams are passed through the same converter
before installation. The class uses `MpsOptimizer`'s non-unitary path and
deliberately does not request unitary stabilization or overlap-fidelity
diagnostics.

Quimb's native `TrotterGate` objects are retained as `GibbsMps.trotter_gates`;
the optimizer receives a separate normalized `(gate, physical_where)` stream
because its public stream contract uses the interleaved purification positions.
Adjacent layer fusion remains enabled by default, so a many-step second-order
stream is intentionally shorter than a literal half-step replay while having
the same product formula. Isolated one-site terms are exact and are appended
as one-site gates rather than represented by dummy edges.

Deferred work includes multi-site gates for explicit string terms across gaps,
native Symmray purification, and observables that contract directly against
the purification without first producing an MPO.
