# MPO graph-cluster assembly audit

Audit date: 2026-09-01

## Scope

This note records the graph-aware `exp_mpo_cluster` assembly safety change.
The local residual construction remains exact for the selected connected
clusters. The change is at the MPO materialization boundary, where crossing
or nested graph clusters can require products of disjoint long-range paths.

## Upstream compatibility probe

The probe ran in the Pepsy `genpy` environment:

- Autoray: `0.11.1.dev1+gc56f64427`
- Quimb: `1.15.1.dev39+g369d09b9d`
- `qtn.MatrixProductOperator` accepts `arrays`, `shape`, and the standard
  index/tag options used by `FirstDegreeMPO.to_mpo()`.
- `qtn.MatrixProductOperator.from_TN` remains available.
- Pepsy `_scatter_add_2d(array, rows, columns, values)` remains the backend
  scatter boundary for NumPy, Torch, JAX, and CuPy paths.
- Autoray dispatch for `add`, `astype`, `zeros`, `stack`, and `tensordot`
  remains available for the NumPy namespace.

Classification:

- Existing Autoray/Quimb interfaces: **adopt**; no upstream patch or local
  vendor copy is required.
- Long-range graph-gap dtype promotion: **compatibility shim**; gap operators
  are explicitly aligned to the residual backend dtype before scatter-add so
  real identity backgrounds cannot discard complex real-time components.
- Full cutwidth/treewidth dynamic programming: **defer**; the current release
  uses a bounded, explicit graph-collection approximation instead.

## Assembly policy

`graph_assembly="auto"` probes only up to the finite
`collection_budget=128`. It keeps small collection plans exact and emits a
`RuntimeWarning` before switching to the one-cluster bounded approximation.
`graph_assembly="exact"` raises on budget overflow, while
`graph_assembly="bounded"` exposes `max_collection_order` for a deliberate
approximation.

The selected mode, collection count, frontier width, budget, and truncation
flag are retained in `MPOClusterExpansionReport` and returned MPO metadata.

## Validation

- `tests/test_mpo_cluster.py`: 22 passed.
- Focused MPO/operator regression set: 166 passed.
- Ruff checks passed for the modified source and test files.
- `git diff --check` passed.

The bounded path reduces the notebook-shaped 4x4 periodic square call from
the previous unbounded collection assembly to approximately 0.09 seconds on
this environment. A full exact 4x4 periodic cluster-size-3 plan still has
about 1.54 million compatible collections and should remain explicitly
opt-in only.
