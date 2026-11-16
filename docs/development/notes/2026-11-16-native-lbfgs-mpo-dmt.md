# Native JAX L-BFGS and ordinary MPO protected SVD

**Implemented:** native Optax L-BFGS and ordinary ket/bra MPO compression
through the existing protected SVD kernel. Public contracts live in the
[solver guide](../../api/solvers/gradient.md#native-jax-l-bfgs) and
[ordinary MPO DMT guide](../../api/operators/mpo_dmt.md).

## Decisions and upstream audit

Installed environment: Python 3.12.14, CPU JAX 0.8.2, Optax 0.2.6, Quimb
1.15.0, Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c, NumPy 2.5.2. No packages were changed.
The installed Autoray is below the checkout's declared `>=0.9` requirement.

Reused the active task's review of the official
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The linked Symmray
Abelian-array documentation was unavailable; installed source supplied the
capability check. Inspected installed Optax `lbfgs`, zoom line-search state,
evaluation counters and cached-value/gradient behavior against its
[official example](https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html).

| Finding | Decision |
| --- | --- |
| Public Optax L-BFGS/zoom/cache diagnostics are available | **Adopt**; capability-check optional Optax without changing dependencies |
| Compiled scan can contain Optax's line-search while loop | **Adopt**; device-side stopping/best tracking, host reporting per chunk |
| Complex curvature conventions and dtype promotion are avoidable | **Adopt** real/imaginary JAX pytree coordinates, preserving leaf precision |
| Existing protected split is basis-independent | **Adopt** a matrix-unit trace covector and independent protection basis |
| Native Symmray or cyclic DMT needs distinct numerical design | **Defer**, reject explicitly rather than densifying |
| Constrained global variational MPO/PEPO fitting | **Defer**; a new unconstrained parameter solver does not implement this objective |

`jax-lbfgs` starts fresh numerical state on every `run()`. Its line searches,
updates and stopping run in compiled chunks; the public Python wrapper is not
outer-JIT-compatible or implicitly differentiable. Compilation is local to a
solve. Scalar/real-loss checks are functionalized with JAX `checkify`, including
rejected search trials. Counters include every evaluated trial and reuse the
terminal cached value/gradient. Bounds and clipping are rejected explicitly.
Existing SciPy `lbfgs` aliases and first-order JAX runners retain their paths.
The second review fixed a convergence-label error: a failed search's zero
fallback step could satisfy `ftol` despite a large gradient. It now remains
`linesearch_failed`; genuine gradient convergence still has its own check.
MPO writeback now prepares every backend reshape/transpose before modifying
any input tensor. Injected preparation failure verifies no partial mutation.

## Ordinary-basis protection

For matrix units $E_{st}$, use ${\color{blue}t_{st}}=\delta_{st}$ outside
protected regions. The Householder protection basis has $t/\|t\|$ first and
independent orthogonal remaining columns. Only protection vectors rotate;
stored MPO tensors retain their original ket/bra basis. This avoids appending
a redundant trace vector to a complete matrix-unit basis and deduplicates
regional identity strings. The Pauli slice-at-zero fast path is retained.

Both adapters share canonicalization, rank budgeting, residual SVD, trace and
marginal guards. Ordinary absolute cutoffs/report norms use physical
Hilbert–Schmidt units, including Quimb's external exponent. Exact semantic
construction attachments are invalidated after numerical compression.

An added trace-gradient oracle exposed NaNs with native Torch SVD backward.
Independent protection vectors remove one avoidable redundancy, but projected
residuals still contain exact zero singular values. The existing explicit
`TorchLinalgConfig(stabilized=True, svd_gap_rtol=1e-9)` policy gives the correct
analytic trace derivative through actual bond reduction. No global backend
registration or native simulation default was changed. Rank changes and
degenerate truncation boundaries remain outside a smooth-gradient guarantee.

**Measured limitation:** ordinary complex SVD truncation does not guarantee
global Hermiticity when a residual cutoff divides a degenerate subspace.
The seeded six-qubit density oracle at cap 12, using the independent protection
basis, produced a global Frobenius Hermiticity defect `1.33e-5` while all
protected marginals passed. This is a different issue from forward marginal
preservation or stabilized derivatives. The public contract explicitly states
that global Hermiticity and positivity are not guaranteed. Real Pauli
coefficients preserve global Hermiticity and remain a useful primary route.
Enforcing Hermiticity for arbitrary ordinary complex MPO truncation is deferred.

## Limited CPU timings

Three warmed runs per operation, median, one BLAS/OMP thread, complex128
storage in **both** bases. Inputs use `_mixture` from the existing
[`pauli_dmt_benchmark.py`](../../../examples/pauli_dmt_benchmark.py): seed 321,
positive product mixtures, Bloch length 0.5. Construction/conversion is outside
timing, cutoff 0, diagnostics off, all preservation checks active.

| Protected geometry | Sites / mixture rank / cap | Pauli | Ordinary MPO |
| --- | --- | ---: | ---: |
| Contiguous triples | 16 / 24 / 12 | 17.85 ms | 20.08 ms |
| Open 5×5 plaquettes, snake-row-major | 25 / 96 / 72 | 690.59 ms | 1100.92 ms |

These are illustrative CPU timings, not speedup or GPU scaling claims.
The ordinary adapter is slower on these workloads; protected SVD remains the
direct method in both bases. A smaller 3×3/rank-32/cap-24 probe explicitly
rejected a required protected budget of 29 rather than violating the cap.

## Validation

The complete focused rerun after the independent-basis correction passed
**408 tests with one pre-existing expected failure**. After the final
stopping/writeback review fixes, the four affected solver and ordinary-MPO
modules passed **187 tests**, including 47 tests in the two new files.
See the [session handoff](../../../history/2026-11-16-native-lbfgs-mpo-dmt.md)
for scope and remaining baseline failures.
Coverage includes Rosenbrock convergence, exact search evaluation counting,
complex/mixed precision, repeated-run state reset, rejected loss contracts,
failed searches, independent dense marginals, a single-bond Pauli reference,
qutrit truncation, explicit regions, external exponent/cutoff scaling,
transactional writes, backend storage and a stabilized Torch trace derivative.

API/layout checks: 54 passed, one known installed-metadata failure
(`0.5.0` installed versus checkout `0.6.0`). The full-suite attempt stopped at
the pre-existing BP `test_loopunion_selects_both_plaquettes_and_clips_at_edge`
NumPy-to-Torch dispatch failure: 102 passed, 7 skipped, 1 failed. No GPU or
native-Symmray numerical DMT support is claimed.
