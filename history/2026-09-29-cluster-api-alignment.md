# 2026-09-29 — Common MPO, PEPO and Gaugy cluster API

- Branch / baseline: `develop`, `a233a40` (recent implementation `8673ef7`).
- Scope: requested API consistency in Pepsy and Gaugy; no numerical kernel,
  dependency, notebook, history-MPO algorithm or optimizer changes.
- Pepsy publication state: working-tree changes, not staged, committed or
  pushed. Gaugy follows its own repository commit/push instructions.

## Implemented

- `cluster_size` on Pauli PEPO bases, dense square/graph plans and their
  main one-shot builders; keep `order` and existing defaults, reject
  conflicting aliases and noninteger cutoffs. Dense positional arguments
  remain compatible. Expose resolved cutoff on products/compiled wrappers.
- Dense plans expose `compile_exp().exp(step)`, `evaluate` and callable forms:
  active blocks by default, `build(-step)` mathematics, fixed NumPy matrices.
- MPO products add `from_bases`, positional `parameters`, runtime term
  `coefficients`, and explicit `materialize=True`. One-shot products accept
  the same vectors. Factor-scale coefficients keep their separate meaning.
- Vector overrides build one additional structural MPO topology with distinct
  `MPOParameter` identities per term. Runtime tensors are never cached;
  graph/native-space/assembly/rank options carry through. Independent slots
  cannot silently acquire a shared gradient when current values coincide.
  Declared spatial symmetries must preserve these independent bindings.
- Parameter-required/callable factor scales use the `parameters` route;
  defaulted factor parameters still work with term vectors.
- Gaugy uses public Pepsy APIs and forwards its cutoff alias through `order`,
  preserving compatibility with the earlier published Pepsy API.

See [the shared API table](../docs/api/operators/exponentials.md#shared-connected-cluster-calls)
for constructor differences, cutoff defaults, physical term layouts,
materialization and connected-log versus partition trace semantics.

## Validation

Activated the shared Python 3.12 environment; CPU-only tests with one BLAS
thread, temporary Python/Numba/Matplotlib caches and `-o addopts=''`:

- Broad Pepsy cluster selection: **304 passed**, five existing warnings,
  125.27 s. Files: `test_cluster_api`, `test_cluster_expansion`,
  `test_mpo_cluster`, `test_cluster_trace`, `test_cluster_correctness_review`,
  `test_cluster_spatial_reuse`, `test_cluster_fixed_factorization`,
  `test_cluster_fixed_compile`, `test_cluster_jit_gradients`,
  `test_mpo_cluster_recursive`, `test_mpo_cluster_compression`,
  `test_pepo_active_storage`, `test_pepo_cutoff_policy`, `test_public_api`,
  `test_package_layout`.
- After the final materialization/factor-binding changes and two added
  regressions: **123 passed**, two existing deprecation warnings, 20.05 s:
  `test_cluster_api`, `test_mpo_cluster`, `test_cluster_trace`,
  `test_public_api`, `test_package_layout`.
- Gaugy focused selection: **196 passed**, six existing warnings, 45.49 s;
  final cross-package contract rerun: **3 passed**, 2.13 s.
- New tests compare dense noncommuting ordered products and normalized/raw
  traces across representations; repeated Torch matrix/trace gradients and
  JAX JIT gradients; graph and square dense sign conventions; old positional
  constructors and conflicting aliases; factor scales and invalid bindings.
- Pepsy `ruff check src tests` and both repositories' `git diff --check` pass.
  Gaugy changed Python files pass Ruff. No full repository suite was run.
- Initial new-test failures were test helpers: graph active blocks require
  their own dense contraction; Gaugy Pauli words are tuples of labels. The
  helpers were corrected without changing existing numerical code.

Installed versions inspected: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev7+g83fb22865`, Torch `2.6.0+cu124`, JAX `0.10.2`. The existing
contraction, backend dispatch and factorization implementations are reused;
no upstream compatibility adaptation was introduced.
