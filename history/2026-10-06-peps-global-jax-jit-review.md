# 2026-10-06 — Global PEPS JAX/JIT investigation

- Scope: answer whether global mode works with JAX/JIT and cutoff zero.
- Branch / baseline commit: `develop` / `8f7c896`.
- Commit status: review notes only in this turn; existing implementation and
  test edits preserved. Nothing staged, committed, or pushed.

## Findings and validation

Real 3x3 complex128 boundary MPS probes show two JIT blockers: positive
cutoff selects a data-dependent rank, and stripped-exponent conversion in
GlobalOptimizer calls `.item()` on a tracer. Stripping also silently loses
JAX gradient contributions without JIT. With `cutoff=0` and
`strip_exponent=False`, JIT and non-JIT optimize successfully and directional
derivatives agree with finite differences within `2.22e-12`. The full
driver using JAX/JIT also improves fidelity and returns norm 1 for both
NumPy and JAX inputs, preserving their respective output backends.

See the [detailed evidence](../docs/development/notes/2026-10-06-peps-global-jax-jit-review.md)
for configurations, results, limitations, and proposed fixes. No numerical
implementation changed and no full suite was run for this investigation.
Local documentation links and `git diff --check` passed.
