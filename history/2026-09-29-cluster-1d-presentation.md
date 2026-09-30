# 2026-09-29 — Restore the 1D notebook presentation

- Scope: investigate and repair the user's reported missing colored LaTeX and
  inconsistent schematics in the sibling Gaugy Examples notebook.
- Branch / baseline: Gaugy Examples `main`, `dbabafa`; Pepsy `develop`,
  `c1ff0f8`. The notebook and drawing helper were already untracked.
- Commit status: working-tree changes only; nothing staged, committed, or
  published. Existing changes in both repositories were preserved.

## Findings and changes

The earlier nearest-neighbor correction replaced formatted Markdown and
explicitly removed every LaTeX `\color{...}` command. This explains the lost
colors. The current equations already parsed in KaTeX; an editor-specific
failure to render all math was not reproduced.

- Restored colored inline and display equations in
  [cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb), with
  standalone display delimiters and blank lines. Kept the nearest-neighbor
  model and current dense/MPO numerical workflow.
- Matched gallery equations and labels to the highlighted support color;
  retained red for internal nearest-neighbor bonds and gray for context.
- Corrected the disjoint example to physical tensor-factor order
  `B_01 tensor B_2 tensor B_34 tensor B_5`. Added matching colored block labels
  and reduced whitespace in
  [cluster_1d_diagrams.py](../../gaugy_examples/pauli_gaugy/cluster_1d_diagrams.py).
- Refreshed only the seven schematic outputs. All 22 code-cell sources,
  notebook metadata, and other saved outputs were preserved.

## Validation performed in this session

Used the existing Python 3.12 environment and CPU execution with one BLAS
thread; no package implementation changed.

- All 78 math expressions rendered with VS Code's installed KaTeX under
  strict error checking.
- Complete notebook execution passed, writing the executed copy to
  `/tmp/gaugy_examples_executed/cluster_1d_presentation.ipynb`.
- `test_cluster_1d_diagrams.py`: **6 passed**.
- Ruff on the changed helper, notebook schema, local notebook link, and
  whitespace checks passed, including checks of the untracked task files.
- Visually inspected the regenerated disjoint example and size-two gallery.
  Compared notebook cells to a before-edit snapshot to verify preservation.

No full package suite was run for this presentation repair. The user's live
editor was not accessible; renderer validation used its installed KaTeX.
