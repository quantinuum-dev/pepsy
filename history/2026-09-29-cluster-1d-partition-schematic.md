# 2026-09-29 — Restore the missing four-site partition schematic

- Scope: continued repair after the user reported the schematic was still wrong.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; no staging, commit, or publication.

The [previous presentation repair](2026-09-29-cluster-1d-presentation.md)
corrected colors and labels but missed a deleted illustration. Comparison with
the committed `nlce_4x4_pbc.ipynb` showed that the original four-site OBC partition
diagram had been replaced with printed counts. The support galleries did not
explain the complete partition sum. A clarification was requested; absent a
reply, this confirmed omission was repaired. The user's intended schematic has
not been confirmed.

- Added `draw_four_site_partitions` in the examples drawing helper. It uses
  the existing independent set-partition enumeration and connectivity checks
  to show all eight terms, grouped by their first cutoff: 1, 4, 2, 1.
- Restored shaded blocks, color by block size, aligned physical site columns,
  explicit tensor-product labels, and the disjoint `B_01 tensor B_23` term.
- Restored the illustration and matching cutoff equations under **Four-site
  example** in
  [cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb).
  Only that section's two cells changed. All other cells, outputs, and notebook
  metadata were preserved exactly, including the six-site galleries.

Validation in the existing Python 3.12 environment:

- Visually inspected the rendered figure and checked eight unique formula
  rows, including the disjoint bond pair exactly once.
- Existing diagram tests: **6 passed**; Ruff on the helper passed.
- **79 math expressions** passed strict KaTeX rendering.
- Full notebook execution passed in
  `/tmp/gaugy_examples_executed/cluster_1d_partitions_restored.ipynb`.
- Notebook schema and whitespace checks passed. No full package suite run.

Preview: `/tmp/cluster_1d_restored_partitions.png`. The user's live editor was
not accessible; no claim is made that this resolves a different diagram or
editor-display problem.
