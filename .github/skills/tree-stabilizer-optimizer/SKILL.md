---
name: tree-stabilizer-optimizer
description: 'Implement, review, test, or extend pepsy.TreeStabOptimizer: a stabilizer-tableau plus tree-tensor-network coefficient simulator. Use for TTN-backed C|p⟩ evolution, frame-mapped Pauli operations, measurements/reset, exact cooling, Clifford gauge disentangling, magic-state injection, trajectory replay, or TreeOptimizer integration.'
---

# TreeStabOptimizer in Pepsy

Use this skill for code, tests, or documentation under
`src/pepsy/optimizers/tree_stabilizer/`. Read these shared contracts first:

- [`tree-optimizer`](../tree-optimizer/SKILL.md) for TTN layout, backend, and
  canonical-region rules.
- [`stabilizer-tensor-networks`](../stabilizer-tensor-networks/SKILL.md) for
  the common Stim/tableau, coefficient-state, cooling, injection, and
  trajectory semantics.
- [`stabilizer-tensor-networks/references/method.md`](../stabilizer-tensor-networks/references/method.md)
  for the mathematical update rules.
- [`stabilizer-tensor-networks/references/pepsy_stim_api.md`](../stabilizer-tensor-networks/references/pepsy_stim_api.md)
  for the public stream and record contract.

Before changing a public path, read
`tests/test_optimize_tree_stabilizer.py`, the TreeStab section of
`docs/api/optimizers/tree_stabilizer.md`, and the TreeStab cases in
`tests/test_trajectory_noise.py`.

## Representation

Preserve

```text
|psi> = C |p>
```

where `C` is the Stim tableau frame and `|p>` is a `TreeTensorNetwork` owned
by `TreeOptimizer`. Do not subclass `MpsStabOptimizer`: site maps, layout,
swap/split updates, and linear canonical metadata are chain-specific. Keep
`to_statevector()` equal to `C @ p_dense` in logical big-endian order, and
delegate `norm()` to the coefficient tree.

Keep coefficient canonicality single-owned as well. Local isometry proofs live
only on ``TreeTensorNetwork`` tensors' ``left_inds``; TreeStab exposes
read-only delegates to the tree API and must not cache another map. Direct,
MPO, and coefficient-frame sub-MPO paths inherit TreeOptimizer's
metadata-gated one-sided compression. Use ``apply_to_arrays`` for backend-only
conversion, and when a TreeStab constructor independently proves a canonical
tree (such as dense cap factorization), install the proven metadata without a
redundant numerical QR sweep.

Clifford events update `C`. Physical Pauli rotations, measurements, resets,
and magic gadgets map through `C† P C` and update `|p>`. Coefficient-frame
sub-MPO events go directly to `TreeOptimizer.apply_submpo`; they are not
conjugated through `C`.

## Non-negotiable invariants

- Build a `TreePlan` from the circuit before replay. Never implicitly relayout
  an entangled coefficient TTN; product TTNs may be remounted exactly.
- Keep backend, dtype, device, logical qubit labels, and tree geometry stable
  during replay. Every user gate, sub-MPO tensor, and native TreeMPO tensor must
  match the coefficient TTN backend and device at stream installation. Non-NumPy
  payloads must also match its dtype; NumPy-to-NumPy dtype promotion is compatible. Reject a
  mismatch with `TypeError` and require explicit preparation. Internal
  Pauli/projector tensors and sampled trajectory matrices follow the live
  backend through explicit internal conversion.
- Treat `chi=None` as uncapped but still honor the configured SVD cutoff.
- Bound dense operator and Pauli decomposition work with
  `max_operator_qubits` / `max_pauli_decomposition_qubits`; use native Pauli,
  expectation, projection, and sub-MPO paths for long sparse supports.
- Preserve queue replacement/extension and retry semantics. `copy()` must
  own an independent TTN, tableau, queue, RNG, and record history.
- Basis-updating measurement must localize the frame Pauli, update
  `C -> C V†`, and project the coefficient tree. Do not implement it through
  TreeOptimizer's physical measurement alone.
- Reset and measure-reset are built from basis-updating measurement without
  exposing internal reset measurements as extra public records.
- Immediate injection recycles clean ancillas; deferred/MAST injection uses
  one reserved ancilla per injectable gate and projects them only afterward.
  Never let the input stream act on the reserved pool.
- Exact cooling is the default constructive pre-check. Clifford
  disentangling is a caller-scheduled sparse sweep, never an implicit replay
  step.
- Use shared `run_trajectory_shots` and
  `run_coalesced_trajectory_shots`; do not duplicate trajectory runners.

## Public stream behavior

Keep `from_stim`, `analyze_stream`, `queued_stream_analysis`, sampling, and
typed result records aligned with the MPS STN frontend. Preserve direct
execution as the default; advisor output must not silently select injection
or deferred execution. Keep logical output ordering stable while allowing
measurement-order permutations, packed output, conditional probabilities, and
chunked sampling without materializing a full `2**n` statevector.

## Validation

Add a small dense/reference regression for every changed contract. Cover the
changed path plus invalid plan/state/backend combinations before tensor work.
For shared trajectory or public API changes, also run the corresponding MPS
STN and package-layout tests.

```bash
source ~/envs/py312/bin/activate
cd /home/reza.haghshenas@quantinuum.com/pepsy
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPYCACHEPREFIX=/tmp pytest -q -o addopts='' \
  tests/test_optimize_tree_stabilizer.py
pytest -q tests/test_trajectory_noise.py
pytest -q tests/test_public_api.py tests/test_package_layout.py
```

Keep implementation in Pepsy and Tensy-specific integration in its sibling
repository. Update the owning subpackage exports, top-level lazy exports,
docs, and public API tests together.
