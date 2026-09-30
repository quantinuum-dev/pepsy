# 2026-09-29 — Separate single and joint 1D notebooks

- Scope: split the 1D example into single- and joint-exponential notebooks with
  matching Hamiltonian, cluster-schematic, and analysis sections. The user
  explicitly confirmed retaining the current NN + NNN interactions.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Existing unrelated changes were preserved.

## Changes

- [cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb) now
  focuses on `exp(-it H)` for the existing ITF + NNN model. Removed its embedded
  joint section and added navigation to the new notebook.
- [cluster_1d_joint.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d_joint.ipynb)
  is independently executable. It defines Hxx, Hyy, Hzz and the matrix product
  `exp(-it Hxx) @ exp(-it Hyy) @ exp(-it Hzz)`, in the order requested by the
  user. The earlier combined notebook's joint section used ZZ, XX, YY; its
  outputs were not reused for the new target. Hxx/Hyy retain onsite Z fields;
  Hzz retains onsite X fields. All factors use the same time and NN/NNN graph.
- Both notebooks follow Hamiltonian → joint/single residuals and schematics →
  full-matrix scaling and cutoff analysis → direct Pepsy MPO and Gaugy Pauli
  comparisons. Both retain one boundary per run, concise colored LaTeX, and
  the same six-site defaults and compression controls.
- Joint MPOs use `pepsy.operators.exp_mpo_cluster_product` directly, with three
  term lists in matrix order. Shared galleries accept a Pauli-axis label so
  the joint figures represent X/Y/Z interactions rather than only ZZ.
- Updated both examples READMEs and added independent factor-order and gallery
  label regressions. No package implementation or 2D notebook was changed.

## Factor-order finding

The first joint execution passed dense and Pepsy checks but failed the Pauli
comparison. `ExactClusterBasis` takes factors in action order, whereas Pepsy
takes matrix order. A one-site independent probe confirmed the convention.
The joint notebook now passes ZZ, YY, XX to Gaugy and XX, YY, ZZ to Pepsy,
producing the same requested target. The actual notebook Gaugy adapter is
covered by the independent four-site OBC/PBC tests. No tolerance was relaxed.

## Validation

- Full single OBC run passed; full joint OBC and PBC runs passed in separate
  sequential kernels. Each run checked 15 compressed and five fixed MPOs,
  explicit full-order reconstruction, scaling, bond-rank consistency, and
  Pauli matrices through p=4. Saved fresh OBC outputs in both notebooks:
  20 single and 21 joint code cells. Temporary validation cells were excluded.
- Maximum MPO construction errors against explicit `C_p`: single OBC
  `3.09e-13`, joint OBC `3.53e-13`, joint PBC `4.07e-13`. Joint Pauli matrix
  errors stayed below `6.11e-14` in the checked cases.
- Example numerical/drawing suite: **35 passed**. After correcting the Gaugy
  convention, both strengthened notebook order/adapter tests passed again.
- Three focused Pepsy ordered-product tests passed: the one-shot three-factor
  facade, recursive full-order product, and fixed disjoint/factor-order test.
- Ruff, notebook schemas, local links, whitespace, and strict KaTeX passed
  (58 expressions in single, 52 in joint). Visually inspected the joint OBC
  Pauli-axis gallery and MPO bond plot.
- Existing single PBC validation is recorded in the
  [single-boundary handoff](2026-09-29-cluster-1d-single-boundary.md); it was
  not rerun for this split. No single numerical code or parameters changed.

Environment and compression policy remain those of the earlier
[MPO audit](2026-09-29-cluster-mpo-dimension-audit.md). No full package suite,
new backend, or larger-system validation was run.
