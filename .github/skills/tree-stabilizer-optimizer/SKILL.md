---
name: tree-stabilizer-optimizer
description: 'Implement, review, test, or extend pepsy.TreeStabOptimizer: a stabilizer-tableau plus tree-tensor-network coefficient simulator. Use for TTN-backed C|p⟩ evolution, frame-mapped Pauli rotations and measurements, tree basis updates, fixed TreePlan layout, or TreeOptimizer coefficient-backend integration.'
---

# TreeStabOptimizer in pepsy

Use this skill for work under
`src/pepsy/optimizers/tree_stabilizer/` and its tests/docs. Read this file
before editing TreeStabOptimizer, then read the closest tree and stabilizer
skills as well:

- [`tree-optimizer`](../tree-optimizer/SKILL.md)
- [`stabilizer-tensor-networks`](../stabilizer-tensor-networks/SKILL.md)
- [`stabilizer-tensor-networks/references/method.md`](../stabilizer-tensor-networks/references/method.md)
- [`stabilizer-tensor-networks/references/pepsy_stim_api.md`](../stabilizer-tensor-networks/references/pepsy_stim_api.md)

## Representation and ownership

The live state is always

```text
|psi> = C |p>
```

where `C` is a `stim.TableauSimulator` basis Clifford and `|p>` is a dense,
two-level `TreeTensorNetwork` owned by a `TreeOptimizer` coefficient engine.
Clifford circuit events update `C` only. Physical Pauli rotations,
measurements, and immediate magic-state gadgets are frame-mapped with
`C† P C` and applied to `|p>`.

Do not subclass `MpsStabOptimizer`: its site maps, `swap+split` updates,
linear canonical-centre metadata, and MPS layout logic are chain-specific.
Do not make `STNState` silently accept TTNs until a deliberate shared tableau
adapter exists; preserve the current MPS API.

## Tree/layout invariants

- `TreeLayoutFinder` is circuit/support-only. Build a `TreePlan` before replay,
  then pass the coefficient TTN separately.
- A live entangled TTN must retain its plan. Never relayout it implicitly.
  Product TTNs may be remounted exactly, with the existing warning.
- The TTN owns its canonical center/region. Delegate movement, normalization,
  compression, and readout to `TreeOptimizer` / `TreeTensorNetwork`.
- Keep backend, dtype, and device consistent across all live TTN tensors.
  Internal Pauli and projector tensors must be created on the state backend.
- Avoid dense `2**k` or `4**k` work for long sparse Pauli supports. Prefer the
  existing tree-native `apply_pauli_rotation`, `apply_pauli_sum`,
  `expectation_pauli`, and `project_pauli` paths.

## Execution rules

The first supported path is intentionally narrow:

1. Construct from an integer, bits, a tableau plus `TreeTensorNetwork`, or a
   product state on a selected `TreePlan`.
2. Apply named or matrix-valued Clifford gates to the tableau only. For a
   non-Clifford or non-unitary dense matrix within the configured qubit budget,
   decompose it into physical Paulis, frame-map each branch, and delegate the
   weighted sum to `TreeOptimizer.apply_pauli_sum`.
3. Apply physical Pauli rotations by extracting the signed frame Pauli and
   delegating the coefficient update to `TreeOptimizer.apply_pauli_rotation`.
4. Measure/project a physical Pauli in the fixed basis by evaluating the frame
   expectation, sampling or validating the outcome, and delegating the
   coefficient projector to `TreeOptimizer.project_pauli`.
5. For basis-updating measurement, construct a tree-distance-aware Clifford
   localizer `V`, apply `V` to `|p>`, absorb `V†` into `C`, and project the
   localized coefficient leaf onto the required computational basis value.
6. Build reset and measure-reset from basis-updating measurement, applying an
   anticommuting physical Clifford after a `-1` result when needed.
7. Prepare `Rz(phi)|+>` on a clean ancilla and implement immediate injection
   with `CNOT(data, ancilla)`, basis-updating ancilla measurement, and the
   Clifford `Rz(2*phi)` correction on a `-1` branch. Recycle ancillas only
   through explicit `run_with_injection(...)`, choosing them by tree distance.
8. Implement deferred/MAST injection explicitly with one fresh ancilla per
   injectable gate: apply preparation, CNOT, and branch correction during
   replay, then project the ancillas at the end in `input`, `middle_out`,
   explicit, or tree-span-greedy `min_span` order.
9. Expose dense statevector readout as `C @ p_dense` in logical qubit order.

10. Use the shared `pepsy.run_trajectory_shots` runner for random-unitary
    mixtures and state-dependent Kraus channels. A selected Kraus branch is
    evaluated on a copied TreeStab state, applied to the live state, then
    normalized before replay continues. Coalesced replay may share
    deterministic prefixes and branch mid-circuit measurements through the
    existing `expectation`/`measure` paths.

11. Keep constructive exact cooling enabled by default for multi-site Pauli
    rotations. Choose a tree-distance-aware leaf whose coefficient state is
    rank-one across its tree edge and a one-qubit stabilizer eigenstate; apply
    the local rotation to that leaf and absorb the controlled-Pauli Clifford
    remainder into the tableau. If no such pivot exists, use the ordinary
    tree Pauli-rotation path. `disentangle_cliffords(...)` is the separate
    caller-scheduled greedy two-qubit Clifford sweep: it scores the affected
    tree-geodesic Schmidt spectra, applies only improving gauge moves, and
    absorbs the inverse Clifford into the tableau. It is not part of ordinary
    replay.

Do not
implement basis-updating operations or injection measurements by calling
TreeOptimizer's physical measurement directly: they must localize the frame
Pauli, update `C -> C V†`, and project the coefficient state while preserving
`C|p>`.

## Validation contract

Every new update needs a dense reference test for small qubit counts:

- Clifford-only evolution leaves the coefficient TTN unchanged and matches
  Stim/dense statevector evolution.
- Frame-mapped rotations match direct dense physical rotations.
- Fixed measurement probabilities, forced outcomes, and repeated projections
  match dense Born probabilities.
- Basis-updating measurement preserves the correct post-measurement physical
  state, and reset returns the target to the requested positive basis state.
- Immediate `T`/`T†` and `Rz(k*pi/4)` injection matches direct dense evolution
  for both measurement branches, leaves recycled ancillas clean, and reports
  projection cost and peak coefficient-tree bond.
- Deferred injection matches direct dense evolution for forced branches and
  each supported projection order, requires one fresh ancilla per injectable
  gate, and leaves the reserved pool clean when requested.
- Sparse long-support rotations/projections work under a small dense-operator
  limit and retain the TreeOptimizer diagnostics.
- Invalid plan/state/backend combinations fail before tensor work.

Run focused tests first, then the tree and stabilizer suites. Public additions
must update subpackage exports, top-level lazy exports, docs, and
`tests/test_public_api.py`.
