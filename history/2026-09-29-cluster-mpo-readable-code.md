# 2026-09-29 — Make the notebook MPO calls readable

- Scope: clarify the MPO code in both 1D notebooks with commented inputs and
  a direct, runnable Pepsy function call before the parameter sweep.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: working-tree edits only; nothing staged, committed, or published.
  Existing unrelated changes were preserved.

## Changes

- In [cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb) and
  [cluster_1d_joint.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d_joint.ipynb),
  separated term/geometry preparation, storage settings, a one-MPO example,
  and the longer sweep. The example calls `exp_mpo_cluster` or
  `exp_mpo_cluster_product` directly and returns an MPO without a report tuple.
- Commented the local-term tuple format, physical inputs, storage options,
  return values, and diagnostic bookkeeping. Core constructor arguments are
  explicit keywords. The sweep requests diagnostics with `return_report=True`
  and has `jupyter.source_hidden=True`; support depends on the viewer.
- Updated the examples README and refreshed all default OBC notebook outputs.
  Kept the single boundary selector, NN + NNN model, joint XX/YY/ZZ matrix
  order, and existing assembly/compression settings. No package code changed.

## Validation

- Both notebooks executed fully in fresh kernels: 23 single and 24 joint code
  cells. All existing matrix, rank, scaling, and Pauli assertions passed.
  Each new one-MPO example matched the corresponding sweep result.
- Each complete run checked 15 compact and five fixed MPOs against independent
  cluster sums. Maximum relative matrix errors: single OBC compact `3.09e-13`,
  fixed `2.38e-14`; joint OBC compact `3.53e-13`, fixed `2.86e-14`.
- Separately executed each standalone example with PBC selected and compared
  with its independent cluster sum: single `2.40e-13`, joint `1.68e-13`.
  This turn did not rerun the full PBC sweep; earlier full runs are recorded
  in the [notebook split handoff](2026-09-29-cluster-1d-joint-split.md).
- Ruff, notebook schemas, strict KaTeX (60 single / 54 joint expressions),
  local links, and whitespace checks passed. Changes to cell sources were
  confined to the MPO section; unrelated sections retained their source.

Used the existing Python 3.12 environment and unchanged compression policy.
No full package suite or larger-system validation was run for this notebook
presentation refactor. No outstanding blocker.
