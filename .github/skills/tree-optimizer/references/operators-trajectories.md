# Tree operators and noisy trajectories

Read this reference when changing TreeMPO construction or exact expectation,
native operator charge sectors, tree-operator compression, or trajectory
replay backed by `TreeOptimizer`.

## Tree-native MPO API

When the consumer is a `TreeTensorNetwork`, `TreeMPO` is the primary operator
API. Use `TreePlan.build_tree_operator(...)` or
`Fermion.build_tree_operator(..., tree=plan)`:

```python
tree_operator = fermion.build_tree_operator(
    hamiltonian=hamiltonian,
    tree=plan,
    compress=True,
)
energy = tree_operator.expectation(tree)
# equivalent exact readout through the state API:
energy = tree.expectation_mpo_exact(tree_operator, range(plan.n))
```

`tree_operator.chain_mpo` is optional compatibility data for ordinary MPS/MPO
workflows. `TreePlan.to_mpo(...)` and `tree_mpo(...)` remain compatibility
builders that return that regular chain MPO and attach the `TreeMPO`; they do
not change the tree contraction route. `to_tree_mpo(...)` remains a
compatibility alias for `build_tree_operator(...)`.
The chain MPO must not be moved into the tree, densified, or compressed as a
state update for exact tree measurement.

`TreeMPO` subclasses Quimb's `TensorNetworkGenOperator`, analogous to
`TreeTensorNetwork` subclassing `TensorNetworkGenVector`. It is the tree twin
of Quimb's `MatrixProductOperator`: its public operator surface includes
`sites`, `nsites`, `site_tag`, `upper_ind`, `lower_ind`, `to_dense`, `H`,
`copy`, `identity`, `from_dense`, `add_MPO`, `singular_values`, `amplitude`,
and canonicalize/compress helpers, while `plan`, `node_tensor`, `neighbors`,
and `bond` provide the branched geometry. It cannot inherit the chain-only
`MatrixProductOperator` implementation because a tree has no left/right
ordering; `chain_mpo` remains the separate chain-compatible representation.

Mixed native operator charges are represented as one public `TreeMPO` with one
homogeneous Symmray tree network per charge in `tree_networks`. Use
`charge_sectors=True` only when separate `TreeMPO` objects are explicitly
needed.

For native fermionic Hamiltonians, one-, two-, and higher-site neutral terms
are fused and factorized from their native Symmray operator tensor over the
TreePlan Steiner subtree, then amalgamated into one charge-aware direct-sum
TTNO. This is the normal general-term route and is canonicalizable/compressible;
it is not a list of ordinary hyperedges. Structured observables may select a
smaller dedicated TTNO, such as the four-state eta-pair endpoint automaton.
`TreeMPO.canonicalize()` performs lossless native QR gauge fixing and
`TreeMPO.compress(cutoff=..., max_bond=...)` performs native graded SVD
truncation. Native operator QR uses the same centralized
`_native_qr_split_tensor` policy as tree-state QR, including the
`stabilized=False` structural-zero safeguard for Symmray arrays.

## Noisy trajectory replay

`run_trajectory_shots` and `run_coalesced_trajectory_shots` support
`TreeOptimizer` factories as well as MPS and stabilizer-TN factories. Use them
for trajectory simulation without forming a density matrix:

- Independent replay samples random-unitary mixtures, Pauli/depolarizing
  channels, and state-dependent Kraus channels. For a Kraus event, the runner
  applies each branch to a copied TTN, obtains its squared norm, samples the
  conditional probability, then applies and normalizes the selected branch on
  the live TTN.
- Coalesced replay shares deterministic prefixes and branches exact
  mid-circuit `measure`, `reset`, and `measure_reset` events. Tree measurement
  probabilities come from `TreeOptimizer.expectation_pauli`; each resulting
  leaf remains normalized.
- The runner converts generated dense matrices through the live state backend.
  When constructing a direct Tree stream, use matrix-valued gate payloads such
  as `pepsy.h()`; textual MPS gate aliases are not normalized by the Tree gate
  parser.
- Regression coverage lives in `tests/test_trajectory_noise.py`, including
  Tree state-dependent Kraus sampling and coalesced measurement branching.
