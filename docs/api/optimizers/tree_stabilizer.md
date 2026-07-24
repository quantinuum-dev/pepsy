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
- physical Pauli rotations, frame-mapped with `C† P C` and applied through the
  tree-native Pauli rotation/sub-MPO path;
- fixed-basis Pauli expectation, measurement, and projection;
- basis-updating Pauli measurement, reset, and measure-reset;
- immediate/recycled magic-state injection for `T`, `T†`, and non-Clifford
  `Rz(k*pi/4)` entries;
- deferred/MAST injection with one fresh ancilla per injectable gate and
  configurable final projection order;
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

Noisy trajectories, cooling, and general non-Clifford matrix decomposition are
later milestones. Immediate injection is explicit: use
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

```{eval-rst}
.. automodule:: pepsy.optimizers.tree_stabilizer.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```
