# `pepsy.optimizers.tree_stabilizer`

`TreeStabOptimizer` is the first tree-backed Stabilizer Tensor Network
milestone. It represents the state as

```text
|psi> = C |p>
```

where `C` is a Stim tableau Clifford and `|p>` is a dense two-level
`TreeTensorNetwork` evolved by `TreeOptimizer`.

The first milestone supports:

- named and matrix-valued Clifford gates, which update only the tableau;
- bounded dense one- and few-qubit matrices, including non-Clifford and
  non-unitary matrices, through coefficient-frame Pauli decomposition;
- physical Pauli rotations, frame-mapped with `C† P C` and applied through the
  tree-native Pauli rotation/sub-MPO path;
- fixed-basis Pauli expectation, measurement, and projection;
- basis-updating Pauli measurement, reset, and measure-reset;
- immediate/recycled magic-state injection for `T`, `T†`, and non-Clifford
  `Rz(k*pi/4)` entries;
- deferred/MAST injection with one fresh ancilla per injectable gate and
  configurable final projection order;
- constructive exact cooling for multi-site Pauli rotations when a
  tree-isolated product-stabilizer pivot exists;
- explicit, caller-scheduled greedy two-qubit Clifford disentangling over
  selected logical qubit pairs;
- dynamic frame-layout planning from the queued ``C† P C`` supports;
- conditional/batched computational-basis sampling without a ``2**n``
  statevector;
- ``chi=None`` for exact evolution up to the requested singular-value cutoff;
- ``from_stim`` and non-consuming ``analyze_stream`` compatibility with the MPS
  STN frontend.
- coefficient-frame ``submpo`` stream events for arbitrary unitary or
  non-unitary MPO operators.
- bounded physical ``cap`` events, with dense state reconstruction guarded by
  ``max_dense_cap_qubits`` and an identity-frame rebuild on ``n - 1`` qubits;
 ``amplitude`` and ``probability`` provide matching small-state readouts.
- optional Torch/JAX/CuPy-compatible coefficient backends through
  ``to_backend`` and ``backend_info``;
- ``ghz``, ``pseudo_stabilizer_rank``, ``norm_diagnostics``, and the shared
  stream-advice/runner surface: ``recommend_magic_strategy``,
  ``recommend_settings``, ``run_stream``, and ``simulate``.

```python
import pepsy

sim = pepsy.TreeStabOptimizer(4, gates=[
    ("h", 0),
    ("cnot", 0, 1),
    ("rz", 0.2, 0),
])
sim.run()
print(sim.to_statevector())
outcome, probability = sim.measure_pauli("Z", 0)
```

The tree plan is fixed before replay. Pass `tree=` or `layout=` when a
specific geometry is required; otherwise the initial stream supports are sent
to `TreeLayoutFinder`. For STN streams whose physical supports spread under
Cliffords, pass `frame_layout="auto"` (or call `current_frame_layout`) to
plan from the pre-replay coefficient-frame supports ``C† P C``. An entangled
coefficient TTN must retain its existing plan, following the same state/layout
contract as `TreeOptimizer`.

Automatic tree construction accepts `layout_kwargs`, including the hybrid
path/congestion weights and bounded greedy/nevergrad refinement options from
`TreeLayoutFinder`. Static `frame_layout="auto"` is intentionally rejected for
branch-dependent feed-forward streams unless an explicit plan is supplied.
When `with_injection` or `with_deferred_injection` builds an automatic plan, the
synthetic magic preparation, data--ancilla CNOT, and projection supports are
included in a magic-aware hybrid/auto layout by default; pass `layout_kwargs` to
increase the search or refinement budget.

Greedy Clifford disentangling is separate from ordinary replay and exact
cooling. Call `sim.disentangle_cliffords(...)` at sparse checkpoints, or place
`("disentangle", {"sweeps": 1, "bonds": ((q0, q1),)})` in a stream. `bonds`
contains logical qubit pairs; `None` visits all unordered pairs in increasing
tree-distance order. The 20 two-qubit Clifford classes are scored by the
aggregate numerical rank and entropy of the affected tree-geodesic edges.
Accepted moves apply `D` to `|p>` and absorb `D†` into the tableau, preserving
the represented physical state while reducing coefficient-tree entanglement.
Constructive exact cooling is enabled by default for multi-site Pauli rotations: it chooses a
tree-distance-aware product-stabilizer pivot, applies one local coefficient
rotation, and absorbs the controlled-Pauli remainder into the tableau. Set
`exact_cooling=False` to force the ordinary tree Pauli-rotation path. The
shared trajectory runner also accepts
TreeStab factories for random-unitary mixtures, depolarizing channels, and
state-dependent Kraus channels; selected branches are normalized at their
trajectory boundary. Dense matrix
  decomposition is bounded by ``max_operator_qubits=2`` by default (the MPS-compatible alias
`max_pauli_decomposition_qubits` is also accepted) and uses the tree-native
Pauli-sum MPO path. `max_pauli_terms` (default 256) is a second guard for
explicit larger-matrix opt-ins; it fails before constructing an oversized
coefficient-frame MPO. Immediate injection is explicit: use
`prepare_magic`/`inject_rz` for one gadget or `run_with_injection` for a stream
and reserved ancilla pool. The runner chooses a nearest clean ancilla in the
fixed tree geometry, recycles it through `reset`, and reports projection cost
plus peak coefficient-tree bond. Deferred injection is explicit through
`run_with_deferred_injection` or `with_deferred_injection`: it applies branch
corrections at their original stream locations, then performs final
basis-updating projections in `input`, `middle_out`, explicit, or greedy
`min_span` order. Basis-updating measurement uses a tree-distance-aware
Clifford localizer before absorbing the inverse Clifford into the tableau;
`absorb_basis=True` is never silently treated as a fixed-basis projector.

The safe MPS naming compatibility surface includes `from_mps`,
`from_tableau_and_nu`, `run(..., progbar=True)`, `apply(..., progbar=True)`,
`expectation_pauli_sum`, and tree truncation accessors. MPS-only
`apply_layout` remains chain-specific. Tree also exposes `from_stim`,
`analyze_stream`, `queued_stream_analysis`, `current_frame_layout`,
`apply_frame_layout`, `probability_bits`, `probability_bits_many`, and
`iter_sample_bits`, plus `amplitude`, `probability`, and `cap`. Computational-basis sampling uses tree-native conditional
projections and is not bounded by `max_dense_sample_qubits`; that constructor
argument remains accepted for compatibility with older callers.

`cap(where, vec)` contracts one physical qubit with a length-two vector and
compacts the remaining labels, matching the MPS physical-cap semantics. It is
a correctness-first dense-state fallback with hierarchical TTN factorization,
so replay rejects states larger than `max_dense_cap_qubits`; it does not build a
full `2**(n-1) x 2**(n-1)` replacement operator. Scalable DEM-style capping
should still use structured weighted-XOR or coin streams. A cap resets the
stabilizer frame to identity and cannot be combined with a static STN frame
layout.

An entry such as ``("submpo", mpo, where)`` acts directly on the coefficient
state ``|p>`` in the same way as the MPS STN API; it is not conjugated through
the physical Clifford frame. The payload must expose a usable MPO interface,
or TreeOptimizer may lower it to a bounded dense operator. For an arbitrary
physical operator, pass ``(matrix, where)`` instead so TreeStab can Pauli-map it
through ``C† G C``; large dense physical matrices remain subject to
``max_operator_qubits``.

Feed-forward uses `("if", record, bit, action)`, with Stim-style negative
measurement offsets and computational bits (`+1 -> 0`, `-1 -> 1`). The action
must be one gate entry. Stim `CX/CY/CZ rec[k] q` instructions are lowered to
this same form by `compile_stim_circuit`.


> API details are maintained as handwritten Markdown in this page.
