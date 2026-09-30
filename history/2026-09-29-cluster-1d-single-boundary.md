# 2026-09-29 — One boundary per notebook run

- Scope: make the 1D notebook run either OBC or PBC, rather than both together.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Preserved unrelated changes.

## Changes

- Added `boundary = "OBC"` with validation and a derived singleton `boundaries`
  tuple. Users can change it to `"PBC"`, restart, and run all cells. Every
  numerical comparison and plot uses the selected boundary only.
- Routed the cluster galleries, four-site example, and disjoint examples through
  that same selection. Removed static OBC-only count captions; counts now come
  from the selected graph. Extended the drawing helpers to show periodic
  partition terms and classify the wrapping NN bond correctly as red/solid.
- Refreshed all 24 code-cell outputs from a successful OBC-only run. No saved
  output combines boundaries. Kept the concise colored Markdown, NN + NNN
  Hamiltonian, direct Pepsy constructor, and compression settings.
- Updated the [notebook guide](../../gaugy_examples/pauli_gaugy/README.md).

Primary files: [notebook](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb),
[drawing helper](../../gaugy_examples/pauli_gaugy/cluster_1d_diagrams.py), and
the existing `test_cluster_1d_diagrams.py` / `test_explicit_cluster.py` tests.
No package implementation was changed.

## Validation

- Full notebook execution passed for OBC and PBC in **separate sequential
  kernels**. Additional assertions checked boundary keys in every result table,
  diagram titles and wrap edges, a single plot column, and 15 compressed plus
  five fixed MPO comparisons per run. Temporary executed notebooks are under
  `/tmp/gaugy_examples_executed/cluster_1d_single_{obc,pbc}.ipynb`.
- Focused example tests: **32 passed**. The notebook setup is checked with each
  selector against independent four-/six-site Hamiltonians. The new periodic
  drawing regression checks all 15 four-site partitions, including the wrapping
  pair product, and the red solid wrapping NN arc.
- After visual review, separated coincident labels for nested PBC blocks.
  All **10** drawing tests passed again; regenerated the partition figures.
  The OBC image remained byte-identical. This label-only follow-up did not
  change numerical code or rerun the full numerical workflows.
- Ruff, notebook schema, all **65** strict KaTeX expressions, local links, and
  whitespace checks passed. Visually inspected the single-panel OBC bond plot
  and periodic four-site partition schematic.

No full Pepsy or Gaugy package suite was run; their implementations are unchanged.
