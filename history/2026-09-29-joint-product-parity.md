# 2026-09-29 — Joint MPO/PEPO/Pauli parity review

- Branch / baseline: `develop`, `b343ced`.
- Scope: user requested another single-versus-joint exponential review.
- Publication: Pepsy edits remain uncommitted and unpublished. Existing
  operator changes and staged unrelated optimizer/skill work are preserved.

## Outcome

Core dense joint products preserve complete local generators, union geometry,
algebraic factor order, bindings, backend derivatives, cluster cutoff,
construction, compression and reports in the tested CPU workflows. The API
distinguishes MPO scalar trace from actual-PEPO trace and Gaugy connected logs.
This is scoped validation, not certification of every backend/size/policy.

Fixed silent acceptance of Quimb compression options when `compress=False`.
Incompatible materialization also fails before coefficient callbacks. Gaugy
now exposes the existing Pepsy graph reuse switch without importing downstream
code into Pepsy. No numerical contraction/decomposition implementation changed.

The review exposed native U(1) failures for conserved hopping and retained
null channels at zero cutoff, shared by single and joint MPOs. These remain
unresolved and are documented explicitly, including four rejection checks.
Default-cutoff diagonal native construction is independently verified.
See the [capability matrix, references and upstream audit](../docs/development/notes/2026-09-29-joint-product-parity.md).

## Validation

Shared Python 3.12 environment; CPU, one BLAS/OpenMP thread; JAX CPU.

- New parity module: **24 passed** in 3.89s, including four explicit current
  native limitation/rejection cases. Their passing status is not support for
  native hopping or zero-cutoff null histories.
- Existing MPO cluster/recursive/compression, cluster API/review/correctness,
  factor reuse, fixed compiler/factorization/JIT, spatial reuse, trace,
  square/graph/routing/shared-plan, public API and package layout selection:
  **325 passed** in 137.57s, with two existing alias deprecation warnings.
- Actual-PEPO trace module: **18 passed** in 14.28s. Total for this review:
  **367 distinct passing Pepsy tests**, including the four explicit current
  unsupported-case checks described above.
- Gaugy joint/API selection: **75 passed** in 24.73s; package compatibility:
  **102 passed** in 16.62s, with six existing Quimb mode warnings.
- Pepsy full source/test Ruff passed; whitespace and current local-document
  targets checked before handoff. No full repository suite or GPU run.

Initial new native tests failed, leading to the explicit unsupported scope
above; none was hidden with xfail or a relaxed accuracy tolerance. One new
Gaugy report assertion initially counted uniform source-shape requests as
located occurrences and was corrected to the existing documented semantics.
