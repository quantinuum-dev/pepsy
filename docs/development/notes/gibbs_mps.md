# GibbsMps implementation note

## 2026-09-02 scheduler and replay audit

The active shared Python 3.12 environment was probed directly:

- Quimb `1.15.1.dev39+g369d09b9d`
- Autoray `0.11.1.dev1+gc56f64427`
- Cotengra `0.8.3.dev7+g1d7fd333f`
- Symmray `0.3.2.dev7+gd63bb4e3f`

The relevant callable signatures are:

- `LocalHamGen.get_trotter_gates(x, order=2, steps=1,
  ordering="sort", fuse_adjacent=True, alternate=True)`
- `LocalHamGen.get_auto_ordering(order="sort", group=False, **kwargs)`
- `MatrixProductState.gate_nonlocal_(G, where, dims=None, method="direct",
  transpose=False, info=None, *, inplace=True, **compress_opts)`
- `MatrixProductState.partial_trace_to_mpo(keep, upper_ind_id="b{}",
  rescale_sites=True)`

The scheduler implementation constructs each local gate as
`expm(frac * x * H_where)`, with the product-formula fractions summing to one
per Trotter step. GibbsMps supplies `x=-beta/(2*n_steps)` and
`steps=n_steps`, so the physical-side replay has total exponent `-beta/2`.
The `order=2` schedule is symmetric, `order=1` is a single ordered sweep,
and `order=4` uses Quimb's Suzuki recursion. `fuse_adjacent` only combines
consecutive applications of the same layer; it does not change the product
formula. `alternate` reverses within-layer traversal for MPS center movement
without changing the commuting-layer result.

The supplied edge list defines the interaction graph, so square and
triangular coordinate graphs use the same generic scheduler and need no
geometry-name switch. Integer coordinate locations can infer 1D/2D/3D
regular shapes and then use `OneDMap`; Quimb geometries with non-integer
sublattice labels such as `(x, y, "A")` remain outside the regular coordinate
inference path.

Connected one-site terms are combined per logical site and distributed evenly
over the incident pair terms before `LocalHamGen` construction. This preserves
the Hamiltonian while allowing the pair scheduler to fuse onsite contributions.
One-site terms on sites with no incident pair remain exact one-site gates.

The public GibbsMps default is now `mode="direct"`, which normalizes to
MpsOptimizer's `quimb-direct` path. The legacy `mode="mpo"` spelling maps to
the same compressor. `non_unitary=True` marks the replay as norm-changing and
disables unitary stabilization; it does not normalize after every gate by
itself. With `normalize_every=True`, each replay step first retains the
canonical center, canonicalizes the active span only when needed, divides one
center tensor by its norm, and accumulates the removed base-10 scale in
`p.exponent`. `normalize_final=True` performs the same operation once for a
trailing unnormalized step.

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
