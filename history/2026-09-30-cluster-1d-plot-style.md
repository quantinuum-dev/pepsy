# 2026-09-30 — Improve cluster_1d plot styling

- Scope: user-requested colors, markers, and marker-edge improvements in
  [cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb).
- Branch / baseline: Gaugy Examples `main` at `dbabafa`; Pepsy `develop`
  at `c1ff0f8`.
- Commit status: working-tree edits only; nothing staged, committed, or
  published. The notebook was already untracked; existing changes preserved.

Updated the three numerical plotting cells with a shared palette, pale marker
fills, darker outlines, consistent method styles, lighter grids, cleaner axes,
and higher-resolution figures. Moved crowded legends below the MPO plots.
Coincident frontier/automaton curves retain distinct square/plus markers and
dashed/dotted lines. Refreshed only those three saved plot outputs.

Validation in the existing Python 3.12 environment:

- Complete default OBC notebook execution passed without cell errors, saved to
  `/tmp/gaugy_examples_executed/cluster_1d_plot_style.ipynb`.
- Visually inspected all three rendered figures.
- Notebook schema, plotting-cell syntax, and whitespace checks passed.
- Snapshot comparison verified every other cell, saved output, notebook
  metadata, and execution count was preserved exactly.

No numerical implementation changed. PBC and the full package test suite were
not rerun. The joint notebook and shared schematic helpers were unchanged.
