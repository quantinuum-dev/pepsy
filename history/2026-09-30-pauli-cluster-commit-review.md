# 2026-09-30 — Pauli cluster commit/integration review

- Scope: review latest Pepsy/Gaugy cluster commits and their local integration.
- Branch / baseline: Pepsy `develop` at `c1ff0f8`; Gaugy at `5da6a0b`.
- Commit status: this handoff is uncommitted. No implementation edits,
  staging, commits, pushes, or remote fetch. Existing local work preserved.

The shared-plan and square-selector commits (`25f2737`, `4e7d08e`) are in
Pepsy HEAD. New channel preparations, constructed-PEPO traces, compression,
and related fixes remain modified/untracked working-tree code. The latest
Gaugy commits require those changes; committed Pepsy lacks `trace_pepo`,
`prepare_cluster_channels`, and `partition_trace_exp`.

A clean committed source snapshot also reproduces a 2x2 periodic order-two
Pauli trace/materialization discrepancy of `0.05746138430952641`. The same
probe gives zero discrepancy in the local Pepsy tree. An order-four snapshot
probe hit NumPy's approximately 8 TiB allocation error and was not retried.
Full details, exact fixture, and dependency versions are recorded in the
[Gaugy review](../../gaugy/history/2026-09-30-pauli-cluster-commit-review.md).

Fresh checks in the shared Python 3.12 environment, one BLAS/OpenMP thread,
CPU JAX, using `python -m pytest -q -o addopts=''`:

- Pepsy: **114 passed**, 44.66 s, across `test_cluster_channel_algebra`,
  `test_cluster_channel_pepo`, `test_cluster_channel_automaton`,
  `test_pepo_operator_trace`, `test_square_cluster_plan`, and
  `test_cluster_api_review`.
- Gaugy integration: **167 passed**, six Quimb deprecation warnings, 27.83 s.

These validate selected current working-tree paths, not committed-only
reproducibility, full suites, notebooks, or larger systems. No new local
numerical defect was found in these selections. A reproducible matching
Pepsy revision is the main integration follow-up before distribution;
this review does not authorize publication.
