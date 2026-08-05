# Simulator planning

`SimulatorPlanner` compares four Pepsy circuit strategies before any circuit
execution:

- `MpsOptimizer`
- `TreeOptimizer`
- `MpsStabOptimizer`
- `TreeStabOptimizer`

The result is typed, ranked advice. It includes the selected MPS order or tree
plan, the physical and stabilizer-frame supports used for pricing, and every
cost component needed to audit the recommendation.

```python
import pepsy

stream = [
    ("h", 0),
    ("cnot", 0, 1),
    ("cnot", 1, 2),
    ("rz", 0.31, 2),
]

advice = pepsy.recommend_simulator(
    stream,
    n_qubits=3,
    chi=64,
)

print(advice.recommended)
for candidate in advice.candidates:
    print(
        candidate.optimizer,
        candidate.applicable,
        candidate.relative_score,
        candidate.max_geometry,
    )
```

`SimulatorPlanner(...).plan()` and `.recommend()` are equivalent. The
convenience function above returns a `SimulatorPlan`; its `best` property is
the first applicable `SimulatorCandidate`, and
`candidate("TreeStabOptimizer")` selects one record by class name. Both record
types also support mapping-style access and `as_dict()`.

## What is priced

The planner first uses `MpsStabOptimizer.analyze_stream` to count Clifford,
injectable, other non-Clifford, structural, and opaque entries. It then builds
two circuit descriptions:

1. Physical supports price all known circuit events for the ordinary MPS and
   tree candidates.
2. A tableau-only dry run removes Clifford events from coefficient-network
   work and records the actual support of each current dressed operator
   `C† O C` for the stabilizer candidates.

Each description receives its own layout. MPS candidates use optimized
one-dimensional window widths. Tree candidates use a `TreeLayoutFinder` plan
and the number of nodes in the minimal connected subtree spanning each
support. Consequently, the comparison can distinguish a physically local gate
that becomes a long dressed Pauli from one that remains local in the
stabilizer frame.

At target bond dimension `chi`, the default work proxy is

```text
one-site event:       weight * chi^2
routed event:         weight * geometry * chi^3
dressed Pauli event:  16 * the corresponding tensor work
frame bookkeeping:    n_qubits * (Clifford entries + dressed events)
```

The dressed-MPO factor of 16 is the operation-count constant used for the
bond-dimension-two HSMPO contraction in
[MPStab](https://arxiv.org/abs/2607.24258). Pepsy differs from that global
estimate by pricing the measured dressed support and by updating one live
Clifford frame incrementally. Override `frame_mpo_factor` or
`tableau_factor` when calibration against local benchmarks supports a
different machine-specific ratio.

`weight_mode="count"` is the default. `"angle"` and `"auto"` reduce the weight
of named rotations with small absolute angles. The default MPS search is
deterministic and does not invoke optional Nevergrad or KaHyPar backends.
`tree_layout_kwargs` can override ordinary `TreeLayoutFinder` structure and
objective options; the circuit, qubit count, `chi`, and event weights remain
planner-owned.

## Interpretation and limits

Scores are relative static work proxies, not runtime, memory, fidelity, or
error bounds. A recommendation does not replace a `chi` convergence sweep.
Exact cooling and immediate or deferred magic-state injection can materially
change stabilizer peak bonds, so benchmark candidates with similar scores.
The returned stabilizer settings enable exact cooling and infidelity tracking,
but the conservative score itself models direct dressed-operator replay.

Unknown physical supports are conservatively priced across all qubits. If the
tableau dry run cannot be defined—for example, a `cap` changes the register
length or feed-forward makes the static frame branch-dependent—the ordinary
candidates remain ranked and both stabilizer candidates are returned with
`applicable=False` and an explanatory warning.

For a purely Clifford circuit, the planner ranks the stabilizer tensor-network
variants cheaply because their coefficient networks are unchanged. Use a
tableau simulator such as Stim instead when no tensor-network state access is
needed.
