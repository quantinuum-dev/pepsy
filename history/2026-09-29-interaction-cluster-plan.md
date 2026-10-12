# 2026-09-29 — Shared finite interaction-graph cluster planning

- Baseline: Pepsy `develop` at `a233a40`; implementation remains local and
  uncommitted, alongside the earlier API alignment/review changes.
- Scope: operators and Gaugy cluster integration. Concurrent MPS changes and
  unrelated journals were preserved and are not part of this task.

## Implemented

- Public `pepsy.operators.ClusterPlan`: located term/support inference,
  higher-body clique connectivity, isolated sites, preserved support
  multiplicities, explicit boundary metadata, lazy cached inventories,
  connected subclusters, exact partition/collection counts with hard budgets.
- Verified finite coordinate symmetry candidates and local labelled-graph
  representative/axis maps, with finite multiplicities and safe bounded-search
  fallback. Parameter identities remain distinct. No numerical tensor cache.
- Shared `from_plan` adapters for MPO, uniform dense graph PEPO, general
  ordered graph PEPO, and Gaugy exact/Taylor/Pauli frontends. MPO also accepts
  `graph="interactions"`. General graph products retain asymmetric,
  nonuniform, duplicate and higher-body terms in algebraic factor order.
- Graph PEPO materialization corrected physical bra/ket orientation and
  complex dtype promotion. Mixed NumPy/backend residual targets are aligned
  before connected subtraction. Scalar trace recurrences share the planner's
  structural collection cache.
- New graph guide, API links and executable NN+NNN example.

## Validation

Shared Python 3.12 environment; JAX numerical checks used `JAX_PLATFORMS=cpu`
after an initial GPU initialization failed from insufficient available memory.
The following final focused selections cover **336 distinct passing tests**:

- `test_interaction_cluster_plan`, `test_cluster_expansion`,
  `test_cluster_trace`, `test_cluster_fixed_factorization`,
  `test_cluster_correctness_review`, `test_mpo_cluster_recursive`: 169 passed
  before adding one final restricted-family regression.
- `test_cluster_api`, `test_cluster_api_review`, `test_cluster_spatial_reuse`,
  `test_cluster_fixed_compile`, `test_cluster_jit_gradients`,
  `test_mpo_cluster_compression`, `test_mpo_cluster`: 107 passed.
- `test_graph_pepo_product`, `test_public_api`, `test_package_layout`:
  59 passed, two compatibility deprecation warnings.
- Final new planner/graph-PEPO rerun: 31 passed (includes the added regression).
- `python -m ruff check src tests` and `git diff --check`: passed.
- `python examples/interaction_cluster_plan.py`: passed; counts `{1: 4,
  2: 5, 3: 4}`, complete collection count 12, matching MPO/PEPO traces.
- Gaugy guide example also executed; Pauli/PEPO normalized traces differed
  by `2.6e-18`. Broader Gaugy results are recorded in its matching journal.

Tests independently enumerate small set partitions and connected subsets;
compare ordered full-size targets to SciPy dense exponentials; test asymmetric
Pauli Y, duplicate periodic bonds, triangular coordinate actions, independent
parameters, and repeated Torch/JAX gradients. Full repository suite was not
run; the selected suite covers the changed operator paths and public layout.

## Measured planning probe

Single local CPU measurements after imports, not a general performance claim.
The times include plan construction and inventory access, with warm access
using an equivalent new plan. Geometric representatives alone do not certify
Hamiltonian reuse.

| Graph / cutoff | Finite placements | Geometry representatives | Exact complete collections | Cold / warm inventory | Count recurrence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24-site open NN+NNN / 4 | 309 | 7 | 887,896,460 | 0.695 / 0.122 ms | 0.636 ms |
| 4x4 open triangular / 3 | 129 | 4 | 573,786 | 0.297 / 0.121 ms | 1.267 ms |

## Environment and upstream review

Installed: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev7+g83fb22865`, Torch `2.6.0+cu124`, JAX `0.10.2`.
Reviewed primary [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra changes](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray](https://github.com/jcmgray/symmray), plus installed materialization
and backend behavior. The requested Symmray documentation page was unavailable;
its primary repository was accessible. No dependency upgrade or upstream shim
was introduced. Local NumPy tree factorization remains the graph materializer.

## Limits and publication

- Geometry alone cannot infer a unique physical embedding or boundary choice.
  Periodic terms must be supplied explicitly; symmetry candidates are checked
  against actual graph and, when supplied, operator/binding labels.
- Enumeration and exact counts remain combinatorial; budget exhaustion raises.
- Graph PEPO materialization is NumPy-only. Scalar traces/residuals preserve
  autodiff, and rank-capped materialization can have a different trace.
- Gaugy connected-log closure is distinct from the residual partition trace.
- New Gaugy graph/automatic-symmetry calls require these matching Pepsy changes.
  Pepsy was not committed or pushed; no permission for publication was inferred.

Gaugy publication was confirmed at implementation commit `b6cfe55` on
`origin/develop`; its final focused selection passed 255 tests. Wider Gaugy
probes exposed an SU zero-rank target issue and unavailable Torch Inductor
Python headers; both issue classes were reproduced with the original Pepsy
and Gaugy commits. See the Gaugy journal for interrupted wider-run scope.
Pepsy remains uncommitted under its separate publication policy.
