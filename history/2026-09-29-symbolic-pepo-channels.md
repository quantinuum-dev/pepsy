# 2026-09-29 — Symbolic PEPO channel preparation

- Scope: improve exact channel reduction and preparation cost, especially
  PEPOs, while preserving fixed autodiff replay and Pepsy ownership.
- Branch / baseline: `develop`, `c1ff0f8`.
- Commit status: all Pepsy changes remain unstaged and uncommitted, including
  earlier task work. Nothing pushed. User journals and other changes preserved.
- Gaugy's adapter, tests and documentation were committed locally as
  `41f88eb`; its three preexisting untracked study journals remain untouched.

## Implemented

`prepare_cluster_channels(..., preparation="symbolic")` compiles independent
residual atoms, removes formal identity zeros, shares full graph-edge slices
before routing, then shares square-edge slices. Reference/tangent samples
reuse linear gathers; expanded numerical PEPO builders are not called.
Graph quotient maps fold into replay constants, so replay retains its fused
matrix-contraction kernel and does no SVD/QR. Full channels add no projection
error. The old reference mode remains default; MPO frontier behavior is
unchanged. Square route labels use mixed-radix arithmetic rather than full
Cartesian lookup dictionaries.

Implementation: `_cluster_channel_pepo.py`, integration in
`cluster_channels.py` and `_cluster_channel_assembly.py`, and `pepo_routing.py`.
Gaugy only forwards the preparation/sharing flags through its bound adapter.
No fermionic or native PEPO scope was added.

## Evidence and remaining limits

The [API](../docs/api/operators/cluster_channels.md) and
[measurements](../docs/development/notes/2026-09-29-symbolic-pepo-channels.md)
include exactness, reference policy and all 48 benchmark rows. Branch exact
bonds fall `(289,17,1,17)` → `(25,5,1,5)`. At cap 4 with six references,
preparation is 124.81→42.13 ms and tracemalloc peak 11.614→0.658 MB. An exact
1 MiB allocation-guard regression succeeds where old preparation raises.

Small or independent-crossing examples can be slower. A changed exact gauge
can worsen tightly capped operator/gradient errors; full channels recover
the original target to roundoff. This remains conservative slice sharing,
not global symbolic minimization. Remaining routed blocks are enumerated
once, and local exponentials/routing width can still grow exponentially.

## Validation

Shared Python 3.12 environment; single BLAS/OpenMP thread, CPU JAX:

- Broad Pepsy selection: **219 passed**, 2 existing compatibility warnings,
  64.24 s. Files: `test_cluster_channel_pepo`, `test_cluster_channels`,
  `test_cluster_channel_assembly`, `test_cluster_channel_structure`,
  `test_cluster_channel_frontier`, `test_pepo_routing_compression_reports`,
  `test_graph_pepo_autodiff`, `test_graph_pepo_product`,
  `test_square_cluster_plan`, `test_public_api`, `test_package_layout`.
- Final PEPO-focused suite: **24 passed**, 16.37 s, including three additional
  allocation-budget/isolated-source checks. Twenty-one overlap the broad run;
  these results are not additive. Torch CPU/CUDA gradients, JAX trace JIT and
  Torch full-graph `aot_eager` assembly compilation passed.
- Gaugy downstream: **163 passed**, six existing Quimb warnings, 40.71 s,
  across channel, joint-parity, binding, materialization-option,
  exponential-API and package files.
- Ruff on `src`, `tests`, and new benchmark; whitespace and touched local
  documentation links pass. The full repository suite was not run.

Logs: `/tmp/symbolic-pepo-regressions.log`,
`/tmp/symbolic-pepo-final-focused.log`, `/tmp/gaugy-symbolic-pepo.log`.
Benchmark: `examples/symbolic_pepo_channels_benchmark.py --repeats 5`; final
run had no concurrent test workload. Tracemalloc is preparation allocation
measurement, not RSS or backward/device peak memory.

The same-task upstream audit was reused with unchanged dependencies. No
installed libraries, publishing destinations or runtime dependency ownership
were changed. Stronger symbolic linear dependencies and environment-aware
approximate basis selection remain unimplemented.
