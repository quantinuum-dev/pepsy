# 2026-09-28 — JIT gradients for fixed clusters

- Scope: verify JAX JIT and Torch compile style gradients in fixed MPO/PEPO
  construction and correct narrow tracing failures.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or
  published. Existing unrelated changes were preserved.

## Changes and findings

Complete three-site p=3 and 2×2 square p=4 MPO and PEPO scalar losses pass
JAX `jit(value_and_grad)` against independent matrix exponentials at zero
and nonzero coefficient/time. Two-site complex-time gradients also pass.
Torch fullgraph capture passes through the local matrix exponential, fixed
identity and split, and the PEPO tree factorization, including backward,
after two native Torch conversion changes and one Python shape-product fix. Complete Torch builder capture still fails in installed Autoray and
Python/Quimb assembly. No whole-builder Torch compile claim is made.

See the [dated evidence](../docs/development/notes/2026-09-28-cluster-jit-gradients.md)
and owning [MPO](../docs/api/operators/mpo_cluster.md) and
[PEPO](../docs/api/operators/cluster_expansion.md) API guides.

## Validation

- `tests/test_cluster_jit_gradients.py`: 10 passed, CPU-only.
- Final affected cluster suites: 183 passed in 102.46 seconds, including all
  ten new JIT cases and the PEPO tree shape-product fix.
- Full `ruff check src tests`, `git diff --check`, and 22 relative
  documentation links passed.
- Full suite and GPU compilation were not run; earlier full-suite evidence
  predates this task.
