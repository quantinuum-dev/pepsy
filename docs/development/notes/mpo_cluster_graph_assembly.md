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
- Frontier/cutwidth dynamic programming: **adopt** for collection counting and
  safety planning. Full tree-decomposition-based tensor assembly remains a
  future optimization; the current planner still materializes the actual
  collection list only when its count is within the configured budget.

## Assembly policy

`graph_assembly="auto"` first counts compatible collections with a
chain-frontier dynamic program. It explicitly materializes the collection
list only when the count is within the finite `collection_budget=128`; a
planner state-work overflow also selects the bounded one-cluster
approximation. `graph_assembly="exact"` raises on a planner or collection
budget overflow, while `graph_assembly="bounded"` exposes
`max_collection_order` for a deliberate approximation.

The selected mode, collection count, frontier width, budget, and truncation
flag are retained in `MPOClusterExpansionReport` and returned MPO metadata.

## Streaming materialization

`assembly="streaming"` is an independent numerical working-memory boundary.
It builds only local cores for each selected graph residual path (or each
selected collection path), inserts a bounded batch directly into the
accumulator's virtual direct sum, and applies fixed-rank semantic TT-SVD with
`assembly_chi` before continuing. No temporary path or batch MPO is built. The
local residual matrices and their operator-Schmidt factorizations are
unchanged.

For a direct or bounded one-cluster graph plan, the streamed result is the
singleton rail plus an additive sum of individual residual paths. It does not
implicitly recover products of compatible paths that the direct shared-rail
assembly can represent. For exact or higher-order bounded collection plans,
the collection paths are themselves streamed, so the untruncated result
matches that selected collection plan up to floating-point accumulation order.
Finite `assembly_chi` truncation is therefore an additional, order-sensitive
approximation. `assembly_batch_size` trades SVD frequency against the peak
pre-compression direct-sum bond dimension.

## Native blocks and adaptive streaming

Direct graph-cluster results now accept the same bosonic Abelian physical
metadata as the higher-order MPO boundary: `symmetry`, `physical_charges`, or
an `MPOPhysicalSpace`. NumPy virtual tensors are converted to retained
`SparseVirtualTensor` blocks before `FirstDegreeMPO.to_mpo()` invokes the
existing Symmray adapter. This preserves the sparse virtual topology and
avoids a dense Symmray virtual-index fallback. The supported symmetries are
`U1`, `Z2`, `U1U1`, and `Z2Z2`; fermionic string/sign histories are still not
implemented for graph clusters.

Streaming accepts `assembly_cutoff`, `assembly_cutoff_mode`, and
`assembly_form`. With no cutoff it keeps the backend-native fixed-rank SVD
policy. With a cutoff it performs an adaptive semantic TT-SVD after each
batch and reports the discarded singular-value weights. Rank selection is
discrete; compiled/JIT Torch/JAX users should leave the cutoff unset when
they need a fully traceable construction.

## Validation

- `tests/test_mpo_cluster.py`: 37 passed.
- MPO semantic and cluster regression set: 169 passed.
- Ruff checks passed for the modified source and test files.
- `git diff --check` passed.

The notebook-shaped 4x4 periodic square call with 424 graph clusters,
`assembly="streaming"`, and a working cap of 64 takes approximately 3 seconds
on the CPU Torch backend in this environment. The frontier planner reaches
its bounded state-work guard at 4,369 states (limit 4,096), selects the
bounded one-cluster fallback, and never materializes the large collection
list. Full exact collection plans remain explicitly opt-in for small graphs.
