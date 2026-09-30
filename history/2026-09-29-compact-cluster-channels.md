# 2026-09-29 — Compact spin-cluster channels

- Scope: literature-informed compact MPO/PEPO construction and autodiff,
  restricted to the user's Pauli/spin workflow.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: Pepsy changes remain unstaged/uncommitted; no push.
  Earlier cluster changes and unrelated user journals are preserved.

## Implementation

Experimental `prepare_cluster_channels` / `ClusterChannelPlan` prepare
constant virtual bases with pivoted QR and replay sparse physical blocks
directly into compact site tensors, without SVD/QR or rank selection in the
gradient path. Fixed direct MPO, native MPO and graph/square PEPO routes
share the API. Optional extra references and finite-difference tangents
inform preparation. Exact graph edge relabeling independently removes
global padding. A Pauli-transform precision mismatch found in probes is fixed.

Gaugy adds only public binding/materialization adapters. Its compact traces
measure the constructed projected PEPO. No reverse dependency or duplicated
tensor-network algorithm was introduced.

See [API](../docs/api/operators/cluster_channels.md) and
[literature, measurements and limits](../docs/development/notes/2026-09-29-compact-cluster-channels.md).

## Validation

Shared Python 3.12 environment, one BLAS/OpenMP thread, JAX CPU,
Torch CPU and CUDA (NVIDIA RTX A5000). All commands use
`python -m pytest -q -o addopts=''`.

- Initial affected selection: **137 passed**, two existing deprecation warnings,
  39.63 seconds. Files: `test_cluster_channels`, `test_graph_pepo_autodiff`,
  `test_graph_pepo_product`, `test_pepo_routing_compression_reports`,
  `test_mpo_cluster_recursive`, `test_public_api`, `test_package_layout`.
- Broader numerical selection: **192 passed, one stale test failed**, 238.82 s.
  Files: `test_cluster_channels`, `test_cluster_fixed_factorization`,
  `test_native_cluster_autodiff`, `test_joint_cluster_parity`,
  `test_graph_pepo_autodiff`, `test_graph_pepo_product`,
  `test_pepo_routing_compression_reports`, `test_pepo_operator_trace`,
  `test_square_cluster_plan`, `test_cluster_factor_reuse`.
  The stale assertion expected native fixed construction to fail, although
  that capability was added in the preceding task. It now asserts the
  independent dense conserved target; other rejection checks remain.
- Final recheck: **26 passed**, 22.57 seconds: all 25 new channel cases plus
  the corrected fixed-policy regression. The earlier precision fixture had
  mixed complex128 operators with float32 parameters; it now explicitly
  requests complex64 operators as well, and all four output paths pass.
- Gaugy affected selection: **159 passed**, six existing Quimb warnings,
  41.85 seconds, including four new channel adapter cases.
- Pepsy source/tests Ruff, Gaugy changed-file Ruff and whitespace checks pass.
  Full repository suites were not run. Timing probes are scoped development
  measurements, not a general performance guarantee.

## Limits and possible later work

Reduced channels are reference-dependent projected objectives. Local QR
defects do not certify global values or gradients; the evidence record shows
nonzero held-out errors and mixed speedups. Rank decisions remain outside
autodiff. Full symbolic channel minimization, environment-optimal projection,
frozen projections inside recursive MPO assembly, and complete Torch builder
capture are not implemented here. Source sparse histories, local residuals
and graph collection budgets still matter. Native PEPO/fermion projection is
unsupported. These are explicit limits, not authorization to expand scope.
