# Tree operators and noisy trajectories

Read this reference when changing TreeMPO construction or exact expectation,
native operator charge sectors, tree-operator compression, or trajectory
replay backed by `TreeOptimizer`.

## Tree-native MPO API

`TreeOptimizer.apply_sub_mpotree` is the primary gate-to-SubTreeMPO entry point.
Ordinary gate replay and explicit TreeMPO application share it across modes.
Keep `apply_subtreempo` / `apply_sub_tree_mpo` / `apply_subttno` as aliases;
`sub_mpotree_event` retains the established stream marker. The FIT traversal
choices `auto`, `depth`, and `depth-first` are independent of compression
mode: auto uses path windows on geodesics and depth-first on branches, while
explicit policies retain their requested FIT order on multi-site operators.

The TreeMPO public exponent and its primary network exponent are separate
Quimb attributes. Use `_scaled_tree_networks` for sector readout/copies so
the public offset is applied once and relative sector scales survive. Local
operator routing must add the operator exponent only after successful
reduction; FIT already carries it in its exact layered target. Never form a
large power of ten merely to apply an operator.
Native statevector readout must retain separate physical legs, restore their
declared sector maps after sparse contraction, and explicitly unpack Symmray
blocks before Autoray host conversion. Fused or trimmed charge bases are not
the full physical output basis. Keep this expansion confined to dense readout.

Local gates build `SubTreeMPO` directly on their Steiner subtree. Its exterior
identity action is implicit; never allocate a full TreeMPO identity layer
and strip it afterward. Preserve original node IDs, physical tags, input/output
indices, compact copies, and region-local validation. FIT targets have state
tensors alone outside the operator region. Full `TreeMPO` remains the general
operator representation with the distinct proof rule below.

For full TreeMPO objects, only builder-proven unchanged exterior identities permit support-based
elision. Keep this proof through copies, identity-preserving conjugation and
internal backend conversion; drop to full-tree application after exterior
tensor changes or for unproven external operators. After unmanaged in-place
array edits, `invalidate_canonical_form` clears both proofs. Do not recertify
an arbitrary copied or canonicalized exterior merely from `operator_support`.

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

Optimizer `expectation_mpo` deliberately retains its numerical update mode.
Preserve the parent RNG around private copies, including failed copies. Its
measurement-local edge records and warnings must work with replay history
disabled, without enabling spectral probes. Multi-node FIT has no edge-cut
records: expose its FIT diagnostics and approximation risk separately rather
than treating `truncated=False` as an exactness certificate. Single-node exact
FIT needs no approximation warning. Exact readout stays a separate contraction.

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

Constructor, legacy `two_site_mode`, and replay selectors use one resolver.
Validate prospective replay settings and streams before installing them.
Ordinary mode/compression/seed/tracking overrides persist; shot overrides are
child-only, with explicit `run_kwargs` taking precedence. Restore the parent
RNG even when the shot template copy fails. MPI-only options must raise
without MPI rather than disappear into the single-replay default.
Caps and state replacement must update the stored root arity to the live
plan so a subsequent copy or shot does not reject its own topology.
Clear geometry-bound gate-factor caches on both operations. Same-size state
replacement can change the TreePlan or site order while reusing a gate object.
Direct computational measurement uses compact positions only for tensor
access and logical labels for public gate calls. Its local projected weights
accept positive branches without a fixed probability floor; projection uses
`track_norm=False` so Born loss does not enter compression diagnostics.

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
  probabilities come from the paired `_measurement_probabilities` protocol,
  also used by direct and queued Pauli measurements. Never reconstruct the
  negative branch as `1 - p_plus`. One-site readout uses projected canonical
  amplitudes; multi-site readout carries one XOR parity index through
  lossless active-subtree QR messages before taking the two norms. Preserve
  each parity sector's coherence, backend, logical-label mapping, and global
  exponent independence. Probability queries must not truncate, form dense
  multi-site operators, or copy optimizer histories. Normalize positive
  selected branches with `eps=0`; each resulting leaf remains normalized.
- The runner converts generated dense matrices through the live state backend.
  When constructing a direct Tree stream, use matrix-valued gate payloads such
  as `pepsy.h()`; textual MPS gate aliases are not normalized by the Tree gate
  parser.
- Regression coverage lives in `tests/test_trajectory_noise.py`, including
  Tree state-dependent Kraus sampling and coalesced measurement branching.
