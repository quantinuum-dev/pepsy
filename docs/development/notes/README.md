# Design and research notes

These notes record design decisions, compatibility checks, and measurements.
Read dates and validation limits before applying a finding to current code.
Document supported public behavior in the [API guides](../../api/index.md)
and [tutorials](../../tutorials/index.md).

## Contents

- [PEPS reusable planner and row-cache policy](peps_sampler_planner_policy.md)

- [9×9 and 10×10 D=4 sampler resource probes](peps_sampler_large_corner_cases.md)

- [PEPS sampler efficiency and bounded batches](peps_sampler_efficiency.md)

- [Real 4×4 D=4 OBC sampler validation](peps_sampler_4x4_validation.md)
  — 8,192 draws against the full state, truncation, weights, and cache lifecycle.

- [PEPS sampler cache audit](peps_sampler_cache_audit.md)
  — Boundary/row reuse, rare-prefix scaling, and bounded performance checks.

- [PEPS sampler rho repair and speed](peps_sampler_rho_repair.md)
  — Optional positive proposals, qubit formula, and measured overhead.

- [PEPS sampler relative-cutoff scaling fix](peps_sampler_boundary_scaling.md)
  — Follow-up to the maturity audit; backend, scale, and cache regressions.

- [PEPS sampler maturity and Verstraete comparison](peps_sampler_maturity_comparison.md)
  — Current validation, measured throughput, and complex64 truncation limits.

- [Cluster optimization status](../cluster_optimization_status.md) — Current
  published backend/downstream capabilities, validation scopes and limitations.
- [Composed boundary-factor derivatives](2026-09-26-projector-boundary-gradients.md)
  — Rank-deficient MPS correction, mathematical contract, and regressions.
- [PEPO product traces](2026-09-26-pepo-product-trace.md) — Product-expansion
  trace construction and contraction options.
- [Structural PEPO trace reduction](2026-09-26-structural-pepo-trace.md) —
  Parameter-independent identity-history reduction and its validation.
- [Repository organization audit](module_organization_2026_09.md) — Completed
  sampler, optimizer, and BP splits; remaining design boundaries to review.
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
