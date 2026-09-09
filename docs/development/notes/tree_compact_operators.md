# Compact local tree operators

Implemented 2026-09-09.

## Representation contract

Ordinary TreeOptimizer gates now build `SubTreeMPO.from_gate` directly.
`SubTreeMPO` retains the original state TreePlan and its node IDs, site tags,
and physical input/output indices, but stores only the connected Steiner
subtree. Its `active_nodes` includes connecting structural nodes; `sites`
includes any physical routing node, while `operator_support` is the actual
gate support. A physical routing node can have a local identity pair. There
are no operator tensors or open operator bonds outside the connected region.

Do not regress this to constructing a full TreeMPO with exterior identities
and stripping them afterward. Dense and native term factorization helpers
have an explicit `active_only` construction path which never iterates over
the exterior. Full TreeMPO constructors keep their general-operator semantics.
The numerical factorization rules, including structural-null cutoffs, are
unchanged.

Compact validation checks only stored operator nodes and their bonds;
application verifies compatibility with the owning state plan. It bypasses
the full-TreeMPO exterior-identity proof entirely, including after compact
operator gauge changes. Copying, conversion, canonicalization, compression,
and exact expectation retain the compact region. `to_dense()` covers only
the physical `sites` in that region. Full-operator algebra remains the
separate TreeMPO API; compact construction is provided through `from_gate`.

All ordinary modes use the compact application boundary: direct/DM retain
their exact routing then compression, SRC/SDC retain complementary-environment
projection, and zipup retains streamed truncation. FIT targets contain both
layers inside the operator region and only state tensors outside it. The
numerical SRC/SDC environment algorithm, cache, RNG, and release schedule were
not changed. Regional canonical recovery still uses the leaf queue and live
`left_inds` proofs added earlier in this session.

## Compatibility audit

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray Abelian HTML page again failed retrieval.

Installed versions: Quimb `1.15.1.dev51+g2e99c793e`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.3.2.dev8+g6c6dd34b5`. Inspected installed Quimb `gate_with_submpo` source
and signatures for that method, `Tensor.split`, `tensor_contract`, and
`TensorNetwork.copy`. Reprobed NumPy/Torch/JAX `svd_truncated` and
`qr_stabilized` dispatch; the earlier same-session canonical/compression
API probes remain applicable. Classification: **adopt** Quimb's compact
local-operator representation principle on tree geometry. No upstream
dependency, dispatch, or compatibility-shim changes were needed. GPU
performance and general compact-operator algebra extensions are deferred.

## Validation

The follow-up dispatch regression explicitly checks `auto`, `direct`, `dm`,
`src`, `sdc`, `zipup`, `mpo`, both `tree_mpo_*` aliases, and `dmrg1/2/3`.
Every ordinary `apply_gate` call must supply a compact operator and invoke
the expected routing, successive, zipup, or FIT kernel. Repeated gates reuse
the compact factorization; legacy private dense-gate kernels are forbidden
in this test. Queued ordinary replay calls this same `apply_gate` boundary.
Explicit low-level `apply_1q`/`apply_2q` compatibility methods retain their
documented specialized kernels.

Canonical preparation is checked for all four direct/DM/SRC/SDC compressors
with an inside center, outside center, contained multi-node canonical region,
and unknown gauge. Inside/contained cases perform no preparation QR; an
outside center stops at the first subtree entry; unknown gauge uses the
safe subtree canonicalization fallback. The follow-up selection passed
112 dispatch, canonical-center/isometry, compact-operator, and successive
environment tests; repository-wide Ruff and whitespace checks also passed.

The broader tree, TreeMPO, SRC/SDC, path, FIT, zipup, trajectory, public API,
package layout, operator representation, and stabilizer selection passed
983 tests and skipped 5. After the final local default-support adjustment,
all 19 compact-operator tests passed. Repository-wide Ruff, skill validation,
catalog validation, and whitespace checks passed. The complete repository
test suite was not run; validation covered the changed tree subsystem and
its shared FIT, trajectory, and stabilizer callers.

New regressions forbid a full-tree node iteration during compact construction,
compare compact/full applications in six modes on paths and branching regions,
check original/custom labels and implicit exterior action, verify cache reuse,
copy/gauge/compression behavior, reject an extra exterior tensor before state
mutation, and compare native U1/U1U1 graded actions. The old FIT test's demand
for two tensors at every tree node was replaced with the required two-layer
active region and one-layer exterior assertion.

A 256-site balanced tree with a two-site sibling gate stores 3 compact
operator tensors rather than 511 full-tree tensors. This is a representation
size check, not a replay timing or GPU-memory benchmark. Plan metadata and
state-side validation still have their own costs; no claim of entirely
region-linear end-to-end replay is made.

## Subsequent direct/DM/SRC/SDC review

Historical findings below were resolved by the follow-up recorded at the end
of this note; they no longer describe the current public gate routes.

The current six-file optimizer, TreeMPO, compact operator, successive, path,
and API-consistency selection passed 655 tests and skipped 4. Separate
failure-path and operation-count probes found the following unresolved issues;
this review did not change their implementation.

- **Native DM rejects too late.** A four-site balanced U1 spinful product
  state with occupations `[0, 1, 0, 1]`, complex128 hopping term `(0, 3)`,
  mode DM, chi 1, cutoff 1e-8, and `track_norm=False` raises the expected
  dense-only `NotImplementedError` after installing the exact operator
  action. The statevector difference from the pre-call state has norm
  1.4142135623730951, and the center also changes. `_update` aborts diagnostics
  but explicitly does not roll back tensors. Unsupported mode/backend
  combinations should be rejected before routing or state installation.
  SRC/SDC raised with zero statevector difference in the same probe.
- **Ordinary one-site unitaries move the center unnecessarily.** On a
  32-site balanced random TTN (D=3, seed 18), ordinary Pauli-X application
  at site 0 moves center 62 to node 0 through five unproven QR edges in
  each of direct/DM/SRC/SDC. A physical unitary can preserve the old center
  and isometry proof, as the explicit low-level one-site kernel already
  does. A fast path should remain inside the compact application contract
  and require reliable unitary provenance or certification; arbitrary
  nonunitary compact operators must still prepare the canonical region.
- **Explicit lower-level entry points remain different.** In particular,
  `apply_subtree_operator` uses compact lowering only for DMRG/zipup.
  Direct/DM/SRC/SDC retain its legacy exact preparation, and SRC/SDC there
  compress the already-routed state. Ordinary `apply_gate`/queued replay
  correctly use compact original-layer application. This is documented
  compatibility behavior, but prevents claiming identical preparation and
  efficiency across every public entry point.

## DMRG compact-region follow-up

Rechecked `dmrg`, `dmrg2`, and `dmrg3` on compact two-site paths and
three-site branching regions, including custom operator tags and physical
indices. Existing automatic traversal already supplies bidirectional
endpoint windows on paths and depth-first blocks on branches. Its choice
depends on induced region geometry, not gate arity or overall region size.
The exact operator/state layers and disposable compressed guess stay separate.

Removed redundant preparation in `TreeFIT._canonicalize_for_block`: a known
canonical region contained in the next block proves its exterior just as a
single contained center does. Preserve cached exterior messages and leave
the interior for the local solve. Unknown/noncontained regions retain the
conservative cache clear and subtree canonicalization, but no longer collapse
the block before replacing it. Tests check unchanged exact state, final
numerical canonical form, absence of interior center moves, and reuse of the
same exterior message object for contained regions.

The 2026-09-09 upstream audit was repeated against the sources linked above
(Symmray Abelian HTML retrieval still failed). Installed versions are unchanged.
Reprobed Tensor.split, tensor_contract, canonize_around, canonize_between,
and NumPy/Torch/JAX QR/SVD dispatch. Classification: **adopt** existing
canonical-region proofs; **defer** unrelated upstream algorithms. No shim,
dependency, factorization, or numerical environment-contraction changes.

Validation: 626 passed, 4 skipped across compact operators, FIT messages,
FIT priorities, path execution, shared TreePepsOptimizer, zipup, and
TreeOptimizer tests. This includes dense NumPy/Torch/JAX environment checks
and even-parity native Symmray FIT regressions. Repository-wide Ruff and
whitespace checks passed; no full-repository test run or timing benchmark.

## Public gate consistency and early native-DM rejection

Resolved the three review findings above. `apply_subtree_operator` now lowers
every mode directly to compact SubTreeMPO and calls `apply_sub_mpotree`.
Explicit SRC/SDC therefore project original operator/state layers, and direct/DM
use the same positive-cutoff behavior as ordinary replay. Private legacy
helpers remain available to the separate chain-MPO compatibility machinery.

The common update boundary rejects native DM before accounting, canonical
preparation, or state installation. Tests check original tensor data objects,
indices, isometry proofs, canonical region, dense state, and empty diagnostics
for ordinary gates, explicit subtree gates, and supplied compact operators.

A single-node compact operator is checked for unitarity using its current
physical matrix, with tolerance tied to arithmetic precision. Only this small
matrix is read on the host; no state is densified. Even native operators are
supported, while odd native operators retain the general graded route.
Certified absorption preserves the canonical region and original `left_inds`,
and carries the operator exponent once. It bypasses FIT and clears its latest
diagnostic record; general one-site filters still use their canonical solve.
No unitary flag is cached across operator mutation. Tests mutate a supplied
identity into a filter to ensure the next call recertifies it.

Repeated the upstream source audit on 2026-09-09; installed versions and
dispatch from the preceding audit are unchanged. Inspected installed
`TensorNetwork.gate_inds` and `Tensor.transpose` signatures. Classification:
**adopt** existing public gate absorption and shared compact routing; **defer**
native odd-unitary certification and unrelated upstream features. No shims.

Validation: 740 passed, 4 skipped across compact operators, successive
compression, paths, FIT messages/priorities, zipup, TreeOptimizer, TreeMPO,
and trajectories. The final RNG-preservation and profiling adjustment passed
16 targeted checks. Repository-wide Ruff, skill and catalog validation, and
whitespace checks passed. No full-repository test run or timing benchmark.

## TreeFIT environment-cache review

The 2026-09-09 review found no stale-message defect in managed FIT updates.
Messages use fixed private target bonds and live fitted bonds. Every cached
message retains its incoming dependencies; invalidation therefore safely
stops at an absent message. Center movements invalidate the changed path
before QR, local replacement invalidates dependent outgoing messages, and
unaffected incoming branches retain the same tensor objects. Unknown gauge
preparation clears the cache. Effective-block projections are discarded on
each update, so subsequent blocks use the current fitted exterior.

Validation: 221 existing FIT-message, priority, path, compact-operator, and
shared TreePepsOptimizer checks passed. An added regression passed after
priming every directed environment and jumping among nonadjacent one-, two-,
and three-node blocks at finite rank. After every update, every cached
message matched an independent complete-branch contraction, unaffected
objects were reused, and storage remained bounded by the directed edge count.
The existing suite includes NumPy/Torch/JAX comparisons and even-native
per-block phase checks. Ruff and whitespace checks passed.

Scope: initial exterior environments can traverse the full state; compact
operator storage does not make every first FIT contraction region-only.
Caches are reused within a fit, not across unrelated gate targets. Standalone
raw state edits need explicit cache and canonical-metadata invalidation.
No numerical implementation change was required by this review.
