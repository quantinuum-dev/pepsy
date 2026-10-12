# 2026-09-29 — Direct frontier preparation for exact compact MPOs

- Scope: improve preparation enumeration and explain exact channels versus
  capped approximation and full symbolic minimization.
- Branch / baseline: Pepsy `develop`, `c1ff0f8`.
- Commit status: all Pepsy work remains unstaged/uncommitted; nothing pushed.
  Prior code and unrelated user journals are preserved. Gaugy was not edited.

## Changes

New `_cluster_channel_frontier` compiles reachable unfinished-cluster states
and local transitions. Completed histories share continuation states before
allocation. `prepare_cluster_channels` adds explicit `preparation="frontier"`
and `frontier_state_budget`; the old preparation remains the default.
Exact interval/direct-graph and uncapped recursive sources are supported;
bounded/auto graph targets are rejected. The new path bypasses complete
collection enumeration but still obeys frontier and memory guards.

`ChannelSource`, symbolic sharing, and the fused assembler accept the new
source of transitions. Native charges, actual-MPO traces, local ordered
exponentials, and connected subtraction retain their contracts. The benchmark
and docs separate finite-cluster error, channel projection error, and exact
structural reuse. See [API](../docs/api/operators/cluster_channels.md) and
[proof, literature, measurements and remaining minimization work](../docs/development/notes/2026-09-29-frontier-cluster-channels.md).

## Validation

Shared Python 3.12, one BLAS/OpenMP thread, Torch CPU/CUDA and JAX CPU.
Commands use `python -m pytest -q -o addopts=''`.

- Broader affected Pepsy selection: **198 passed**, two existing deprecation
  warnings, 68.90 s. Files: `test_cluster_channel_frontier`,
  `test_cluster_channel_structure`, `test_cluster_channels`,
  `test_cluster_channel_assembly`, `test_graph_pepo_autodiff`,
  `test_graph_pepo_product`, `test_pepo_routing_compression_reports`,
  `test_square_cluster_plan`, `test_public_api`, `test_package_layout`.
- After the final plan-level budget override/validation addition, the four
  channel files above were rechecked: **85 passed**, 40.48 s. These overlap
  the broader selection; they are not 85 additional cases.
- Gaugy integration: **23 passed**, 8.14 s. Files: `test_cluster_channels`,
  `test_cluster_pepo_bindings`, `test_cluster_materialization_options`.
- Frontier coverage includes arbitrary independent residuals and their
  adjoints, zero/nonzero live parameters, CPU/CUDA, native U1/Z2/U1U1/Z2Z2
  with repeated physical charges, JAX trace JIT, Torch full-graph assembly,
  gapped higher-body ordered products, and hard refusal of budget/policy
  mismatches. The forty-site test makes all original source assemblers and
  collection enumerators raise, then compares the actual-MPO trace and its
  derivatives against an analytical oracle.
- An initial large-case value tolerance failure was traced to the existing
  Torch exponential error shared by the old assembler, documented in the
  evidence note. An initial compilation test incorrectly mocked the Torch
  namespace during Dynamo's own namespace inspection; eager no-decomposition
  and compiled replay checks are now separate and both pass.
- Ruff across source/tests and the new benchmark, whitespace and all eight
  edited documents' local links passed. No full-repository or optimized compiler
  throughput claim is made.

## Remaining limits

Full channels are exact for the chosen cluster expansion. A smaller cap still
approximates, and finite cluster order can differ from the global exponential.
Preparation no longer lists complete collections on the frontier path, but
reachable transitions and local dense residuals remain. Frontier width can
be exponential in graph cutwidth. Global symbolic linear-dependency reduction
and minimal parameter-family automata remain separate, unimplemented work.
