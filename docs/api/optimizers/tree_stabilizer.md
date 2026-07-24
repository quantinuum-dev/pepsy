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
- dense statevector readout for small correctness checks.

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
to `TreeLayoutFinder`. An entangled coefficient TTN must retain its existing
plan, following the same state/layout contract as `TreeOptimizer`.

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
decomposition is bounded by `max_operator_qubits` (the MPS-compatible alias
`max_pauli_decomposition_qubits` is also accepted) and uses the tree-native
Pauli-sum MPO path. Immediate injection is explicit: use
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
`expectation_pauli_sum`, and tree truncation accessors. MPS-only APIs such as
`from_stim` and `apply_layout` are not aliases: their chain-specific semantics
remain intentionally separate. Tree computational-
basis sampling is available through `sample_bits`/`sample_bitstrings` using
dense readout, bounded by `max_dense_sample_qubits`, so it is intended for
small trajectory/coalescing checks.


> API details are maintained as handwritten Markdown in this page.
