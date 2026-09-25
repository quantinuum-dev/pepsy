# Design and research notes

These notes record design decisions, compatibility checks, and measurements.
Read dates and validation limits before applying a finding to current code.
Document supported public behavior in the [API guides](../../api/index.md)
and [tutorials](../../tutorials/index.md).

## Contents

- [`release_readiness_2026_09.md`](release_readiness_2026_09.md) — Release
  compatibility review, migration actions, merge scope, and validation evidence.
- [`dependency_minimums_2026_09.md`](dependency_minimums_2026_09.md) — Required
  upstream APIs, tested lower bounds, and optional-backend validation limits.
- [`ci_compatibility_2026_09.md`](ci_compatibility_2026_09.md) — Released versus
  development dependency failures, seed handling, and CI profile corrections.
- [`package_simplicity_2026_09.md`](package_simplicity_2026_09.md) — Package,
  alias, dependency, and agent-guidance assessment; proposed cleanup order.
- [`belief_propagation.md`](belief_propagation.md) — Belief propagation for tensor networks, BP gauging, and the
  loop series / loop cluster expansion corrections.
- [`quimb.md`](quimb.md) — How Pepsy leans on `quimb` (and `cotengra`,
  `autoray`); the "pepsy concept → quimb API" map.
- [`quimb_symmray_opportunities_2026_09.md`](quimb_symmray_opportunities_2026_09.md)
  — Remaining measurement, compression, environment, and native fermion
  integration opportunities after the September compatibility work.
- [`cotengra_2026_09.md`](cotengra_2026_09.md) — Cotengra minimum-version
  validation and exact-PEPS traversal memory measurements.
- [`autoray_2026_09.md`](autoray_2026_09.md) — Autoray device, random,
  conversion, dispatch, and compilation opportunity audit.
- [`symmray.md`](symmray.md) — Block-sparse abelian-symmetric and fermionic
  tensors via `symmray`, and how they bridge into pepsy's tagging conventions.
- [`symmray_2026_09.md`](symmray_2026_09.md) — Symmray 0.4 phase fixes,
  flat-array batching, and opt-in compression opportunities.
- [`fermionic_mpo.md`](fermionic_mpo.md) — Fermionic Fermi-Hubbard MPO
  conventions, the Symmray/Jordan-Wigner bridge, and current validation status.
- [`mpo_cluster_graph_assembly.md`](mpo_cluster_graph_assembly.md) — Upstream
  compatibility and bounded graph-collection assembly audit for MPO clusters.

## Relationship to other docs

- [Project roadmap](../plans/project.md) — historical progress and proposals.
- [`../modules/`](../modules/README.md) — concise implementation maps.
- [Session journal](https://github.com/quantinuum-dev/pepsy/blob/develop/history/README.md) — dated handoffs and validation
  records. Historical findings and proposed next steps are not active policy.
- [User documentation](../../index.md) — installation, workflows, and APIs.

```{toctree}
:hidden:

release_readiness_2026_09
package_simplicity_2026_09
dependency_minimums_2026_09
ci_compatibility_2026_09
belief_propagation
quimb
quimb_symmray_opportunities_2026_09
cotengra_2026_09
autoray_2026_09
symmray
symmray_2026_09
fermionic_mpo
higher_order_mpo_benchmarks
mpo_cluster_graph_assembly
2026-09-15-inhomogeneous-pepo-slots
2026-09-15-sparse-hessian-trust-region
boundary_quimb_compression_modes
contract_flat_backend_scalars
fit_one_site_qr
gibbs_mps
mpo_mps_alignment
mps_dynamic_controls
mps_entropy_backend
mps_final_audit
mps_fit_copy_validation
mps_gpu_backend_audit
mps_layout_scheduler
mps_quimb_compression_audit
mps_stab_modes_fit
mps_to_ttn
mps_transfer
pepo_backend_dtype
stabilizer_gpu_backend_audit
torch_export_compile_compatibility
tree_api_consistency
tree_auto_cutoff
tree_compact_operators
tree_entropy
tree_fit_execution
tree_fit_incremental
tree_fit_large_benchmarks
tree_gpu_backend_audit
tree_modes_review
tree_mps_fit_parity
tree_path_execution
tree_sampler_symmray
tree_successive_environments
tree_zipup_fit
```
