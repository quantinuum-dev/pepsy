# 2026-09-29 — Restore the requested ITF plus NNN notebook model

- Scope: the user clarified that `cluster_1d.ipynb` must use transverse-field
  Ising with next-nearest-neighbor ZZ interactions, and supplied the incorrect
  NN-only size-four gallery as evidence.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Pre-existing changes in both repositories were preserved.

## Correction to the previous repairs

The [presentation repair](2026-09-29-cluster-1d-presentation.md) and
[partition-schematic repair](2026-09-29-cluster-1d-partition-schematic.md)
retained an incorrect NN-only model. The first code cell forced `J2=0.0`,
so changing figure labels could not restore the requested NNN physics.
This handoff supersedes their model interpretation; their recorded tests
validated the earlier NN-only notebook, not the requested model.

## Changes

- Restored `J1=1`, `J2=0.5`, `h=1` and
  `H = J1 sum Z_i Z_(i+1) + J2 sum Z_i Z_(i+2) + h sum X_i` in the shared
  notebook constructor. Dense single/joint targets, residuals, MPOs, Pauli
  channels, and diagrams all receive the same weighted graph.
- Updated every model explanation, gallery count, plot label, and both
  examples READMEs. Kept colored LaTeX and validated its rendering.
- Drew NNN edges as green dashed arcs labeled `J2 Z_i Z_(i+2)`; NN edges
  remain solid red. The six-site OBC size-four gallery has 12 supports, seven
  cyclic; `{0,1,2,3}` has five bonds and two independent cycles.
- Generalized the four-site partition illustration to the actual NN/NNN
  graph: 13 OBC terms, including `B_02 tensor B_13`. Nonconsecutive blocks
  do not enclose skipped sites; the embedding notation restores site order.
- Recomputed and saved all 22 code-cell outputs after successful execution.
  No Pepsy/Gaugy package source or 2D notebook was changed by this task.

Primary files: [notebook](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb),
[drawing helper](../../gaugy_examples/pauli_gaugy/cluster_1d_diagrams.py),
[guide](../../gaugy_examples/pauli_gaugy/README.md).

## Validation performed now

Used the existing Python 3.12 environment and single-threaded CPU execution.

- `test_cluster_1d_diagrams.py` and `test_explicit_cluster.py`: **31 passed**.
  New regressions execute the actual notebook setup and compare the four-
  and six-site OBC/PBC Hamiltonians against an independent Pauli matrix sum.
  Drawing checks verify the 12-support gallery, its internal NNN arcs, and
  all 13 four-site terms. Four-site PBC NNN pairs are counted only once.
- Full notebook execution passed in
  `/tmp/gaugy_examples_executed/cluster_1d_nnn_restored.ipynb`.
  Single/joint full-order reconstructions and order-1 through order-5 scaling
  assertions passed. Thirty compact MPO matrix comparisons through p=5 had
  maximum relative construction error `3.09e-13`; Pauli comparisons through
  p=4 had maximum relative error `5.91e-14`.
- All **95** math expressions passed strict KaTeX rendering; Ruff on the
  modified Python files, notebook schema, README links, and whitespace checks
  passed.
- Visually inspected the regenerated size-four gallery and both partition
  illustrations. Preview: `/tmp/cluster_nnn_chain-gallery-4.png`.

No full package suite or larger-system validation was run. The notebook
defaults and all saved figures/results now describe the NN + NNN model.
