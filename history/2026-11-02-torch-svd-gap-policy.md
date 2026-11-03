# Torch SVD gap policy

`TorchLinalgConfig.svd_gap_rtol` now controls the relative gap/sum threshold
of the stabilized Torch SVD VJP, with the existing `1e-6` default. Each SVD
captures the setting at forward time so a later configuration restoration does
not change its backward rule. Rectangular and complex-phase inverse singular
values retain their separate numerical-rank threshold. A smaller gap threshold
is an explicit numerical choice and does not resolve exact degeneracy.

The change was motivated by the downstream Gaugy χ=6 PEPO diagnostic, which
found a wrong-sign directional gradient under `1e-6` and close agreement with
finite differences under `1e-8` at one parameter state. See the sibling
`gaugy/history/2026-11-02-pepo-su6-gradient-diagnostic.md` for the fixture,
measurements, and limitations. This evidence does not establish general PEPO
gradient accuracy or JAX rank-growth support.

The JAX `svd_jax` function also accepts an explicit positive static
`gap_rtol`. Its default remains `1e-6`; the opt-in value is part of the JIT
program and uses the same bounded pullback. Near-gap real and complex JAX
gradient checks compare the configured rule to the native resolved reference.

Focused Torch and JAX SVD, backend, and prepared SU checks passed 215 tests,
including narrow-gap real/complex reference checks and invalid-setting cases.
`python -m ruff check src tests` and `git diff --check` passed. No full suite
or public release is claimed.
