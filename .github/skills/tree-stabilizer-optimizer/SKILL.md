---
name: tree-stabilizer-optimizer
description: 'Implement, review, test, or extend pepsy.TreeStabOptimizer: a stabilizer-tableau plus tree-tensor-network coefficient simulator. Use for TTN-backed C|p⟩ evolution, frame-mapped Pauli rotations and measurements, dynamic C†PC frame-support layouts, conditional computational-basis sampling, chi=None exact evolution, Stim/stream analysis, bounded dense operators, exact cooling, Clifford gauge disentangling, magic-state injection, trajectory replay, or TreeOptimizer coefficient-backend integration.'
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

Before changing a public path, also read
`tests/test_optimize_tree_stabilizer.py`, the TreeStab section of
`docs/api/optimizers/tree_stabilizer.md`, and the TreeStab cases in
`tests/test_trajectory_noise.py`. Keep package implementation in Pepsy; keep
Tensy-specific integration in its sibling repository.

## Representation and ownership

Maintain the state representation

```text
|psi> = C |p>
```

where `C` is the Stim tableau frame and `|p>` is a dense, two-level
`TreeTensorNetwork` owned by a `TreeOptimizer` coefficient engine. Clifford
circuit events update `C` only. Physical Pauli rotations, measurements, and
magic-state gadgets map with `C† P C` and update `|p>`.

Do not subclass `MpsStabOptimizer`: its site maps, swap-and-split updates,
linear canonical-center metadata, and MPS layout logic are chain-specific. Do
not make `STNState` silently accept TTNs. Keep `to_statevector()` equal to
`C @ p_dense` in logical big-endian order and keep `norm()` delegated to the
coefficient tree.

## Tree and stream invariants

- `TreeLayoutFinder` is circuit/support-only. Build a `TreePlan` before replay,
  then pass the coefficient TTN separately. For STN streams, a frame-layout
  prepass may derive the supports of the current `C† P C` images and feed those
  supports to `TreeLayoutFinder`.
- A live entangled TTN must retain its plan. Never relayout it implicitly;
  product TTNs may be remounted exactly with the existing warning. Reject
  `tree=` and `layout=` together and keep the selected geometry fixed during
  replay.
- The TTN owns its canonical center/region. Delegate movement, normalization,
  compression, readout, and backend conversion to `TreeOptimizer` /
  `TreeTensorNetwork`. Keep backend, dtype, and device consistent across all
  live tensors; create internal Pauli/projector tensors on that backend.
- Treat `chi=None` as an uncapped bond limit. The requested singular-value
  `cutoff` still applies, so “exact” means exact up to that numerical cutoff.
- Keep physical `cap(where, vec)` deliberately bounded and explicit: it
  reconstructs `C @ p`, contracts one physical leg, rebuilds a reduced
  coefficient tree with an identity tableau frame, and enforces
  `max_dense_cap_qubits`. Use hierarchical dense-to-TTN factorization for the
  rebuild; do not materialize a `2**(n-1) x 2**(n-1)` rank-one replacement
  operator. Compact the remaining labels and reject static frame layouts,
  matching the MPS physical-cap contract.
- Keep `amplitude(bits)` and `probability(bits)` as dense small-state
  diagnostics over the same leftmost-qubit ordering as `to_statevector()`.
- Avoid dense `2**k` or `4**k` work for long sparse supports. Prefer
  `apply_pauli_rotation`, `apply_pauli_sum`, `expectation_pauli`, and
  `project_pauli`.
- Dispatch coefficient-frame `submpo` events directly to
  `TreeOptimizer.apply_submpo`, matching MPS semantics. A sub-MPO is not an
  arbitrary physical black-box gate: it must be MPO-like (or accept the
  bounded dense fallback), and it is not conjugated through `C`. Physical
  arbitrary operators use bounded `(matrix, where)` entries and Pauli-map
  through the frame.
- `run` removes only successfully applied entries, leaving a failing entry
  queued for retry. Preserve `set_gates` replacement and `add_gates` extension
  semantics. `copy()` must own independent TTN, tableau, queue, RNG, and event
  records.

## Execution rules

Keep the supported replay paths explicit:

1. Construct from an integer, bits, a tableau plus `TreeTensorNetwork`, or a
   product state on a selected `TreePlan`.
2. Apply named Clifford gates and unitary matrices recognized by
   `stim.Tableau.from_unitary_matrix` to the tableau only. For non-Clifford or
   non-unitary matrices, enforce `max_operator_qubits` before decomposition
   (`max_pauli_decomposition_qubits` is its compatibility alias), decompose
   into physical Paulis, frame-map each branch, and delegate the weighted sum
   to `TreeOptimizer.apply_pauli_sum`. Do not coerce arbitrary unitaries into
   Cliffords; route a zero decomposition through the tree-native zero operator.
3. Apply physical Pauli rotations by extracting the signed frame Pauli and
   delegating to `TreeOptimizer.apply_pauli_rotation`.
4. Measure/project a physical Pauli in the fixed basis by evaluating the frame
   expectation, sampling or validating `+1`/`-1`, projecting the coefficient
   tree, and normalizing the selected branch. Reject forced outcomes with
   effectively zero probability before mutation.
5. For basis-updating measurement, construct a tree-distance-aware Clifford
   localizer `V`, apply `V` to `|p>`, absorb `V†` into `C`, and project the
   localized coefficient leaf onto the required computational basis value.
6. Build reset and measure-reset from basis-updating measurement, applying an
   anticommuting physical Clifford after a `-1` result when needed. Do not
   create extra public measurement records for internal reset measurements.
7. Prepare `Rz(phi)|+>` on a clean ancilla and implement immediate injection
   with `CNOT(data, ancilla)`, basis-updating ancilla measurement, and the
   `Rz(2*phi)` correction on a `-1` branch. Recycle ancillas only through
   `run_with_injection(...)`, selecting them by tree distance.
8. Implement deferred/MAST injection explicitly with one fresh ancilla per
   injectable gate: apply preparation, CNOT, and branch correction during
   replay, then project ancillas at the end in `input`, `middle_out`, explicit,
   or tree-span-greedy `min_span` order. Preserve projection records and report
   fields.
9. Expose `sample_bits`/`sample_bitstrings` through conditional TTN
   projections. Share collapsed copies for common prefixes and use binomial
   counts for batched shots; never build a `2**n` statevector. Keep logical
   columns stable while allowing a measurement-order permutation, packed
   output, probability queries, and chunked iterators.
10. Keep `from_stim`, `analyze_stream`, and `queued_stream_analysis` aligned
    with the MPS STN frontend. Reuse the shared Stim compiler/sample objects
    and retain `stim_plan`/`stim_sample` on `from_stim` instances.

Do not implement basis-updating operations or injection measurements by calling
TreeOptimizer's physical measurement directly: localize the frame Pauli,
update `C -> C V†`, and project the coefficient state while preserving `C|p>`.

## Exact cooling and gauge disentangling

Keep `exact_cooling=True` as the default for multi-site Pauli rotations. When a
valid product-stabilizer pivot exists, apply the local coefficient rotation and
absorb the controlled-Pauli remainder into the tableau, recording
`exact_cooling_events`; otherwise fall back to the ordinary tree-native path.
With `exact_cooling=False`, always use the ordinary path.

Keep `disentangle_cliffords(...)` caller-scheduled. For selected logical pairs,
score the 20 two-qubit Clifford representatives on the pair's tree-geodesic
edges using numerical rank and entropy, accept only improving candidates, apply
the coefficient gauge without truncation, and absorb its inverse into the
tableau. Preserve `C|p>` up to the requested cutoff, update
`disentangle_events` and `bond_history`, and never invoke this sweep during
ordinary replay. Stream checkpoints accept only `sweeps`, `bonds`, and `tol`.

## Trajectory integration

Use the shared `pepsy.run_trajectory_shots` and
`pepsy.run_coalesced_trajectory_shots` runners; do not duplicate them in the
TreeStab module. Keep the protocol marker and these semantics intact:

- random-unitary mixtures and depolarizing outcomes enter through the normal
  TreeStab stream path;
- state-dependent Kraus probabilities are computed on copied TreeStab states
  that retain the current Clifford frame, and the selected branch is applied
  and normalized at the trajectory boundary;
- coalesced replay may share deterministic prefixes and branch mid-circuit
  measurement/reset events through the existing expectation/measurement paths;
- native stochastic stream entries require the trajectory runner, while plain
  `run()` remains deterministic.

## Validation contract

Every new update needs a small dense reference test for the changed contract:

- Clifford-only evolution leaves `p` unchanged and matches Stim/dense state
  vectors; frame-mapped rotations and dense operators match direct evolution.
- Fixed measurement probabilities, forced outcomes, repeated projections,
  basis-updating measurements, reset, and measure-reset match dense branches.
- Immediate `T`/`T†` and `Rz(k*pi/4)` injection match both branches, recycle
  clean ancillas, and report projection cost and peak tree bond.
- Deferred injection matches direct evolution for every projection order,
  requires one fresh ancilla per injectable gate, and protects the reserved
  pool.
- Exact-cooling success/fallback, scheduled disentangling, long-support
  operations under a small operator budget, queue retry/copy isolation,
  `chi=None`, dynamic frame layouts, and conditional/batched sampling are
  covered.
- Random-unitary, state-dependent Kraus, coalesced measurement, and terminal
  sampling paths match the shared trajectory result contract.
- Invalid plan/state/backend combinations fail before tensor work.

Use the Python 3.12 environment and temporary caches. This suite is tagged as
extended integration coverage, so the default `pytest -q` deselects it; clear
`addopts` when running it:

```bash
source ~/envs/py312/bin/activate
cd /home/reza.haghshenas@quantinuum.com/pepsy
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPYCACHEPREFIX=/tmp pytest -q -o addopts='' \
  tests/test_optimize_tree_stabilizer.py
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPYCACHEPREFIX=/tmp pytest -q tests/test_trajectory_noise.py
```

Then run `tests/test_optimize_tree.py`, the stabilizer suite, and relevant
public API/package-layout tests when shared TreeOptimizer code, exports, or
docs are affected. Public additions must update subpackage exports, top-level
lazy exports, docs, and `tests/test_public_api.py`.
