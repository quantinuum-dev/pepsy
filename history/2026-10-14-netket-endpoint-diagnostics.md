# 2026-10-14 — Accepted-state NetKet TDVP diagnostics

- Scope: implement the first tVMC follow-up: accepted-state solve, physical
  tangent and sampling diagnostics. No new scaling solver or gradient rule.
- Branch / baseline: `develop` / `8f8695b`.
- Publication scope: the user authorized committing and pushing these NetKet
  TDVP changes to `develop`. This entry accompanies that scoped commit;
  unrelated compression, fitting, backend and Torch edits remain excluded.

## Changes and rationale

- `NetKetTDVPDriver.diagnostics()` synchronizes the accepted integrator state,
  draws a fresh endpoint MC batch, and recomputes native forces, QGT and solve.
  It leaves native stage velocity/force/QGT caches and integrator state intact;
  diagnostic sampling advances the sampler/RNG.
- Reports regularized and unregularized normal-equation residuals, squared
  unregularized residual, complex tangent mismatch and absolute tangent norms.
  Real/holomorphic coordinates and real/imaginary time have distinct equations.
- Sampling records preserve chains for energy standard error, split Rhat and
  autocorrelation. Unavailable/nonfinite mixing statistics become `None`.
  Full-sum checks use exact Born weights on nonzero-amplitude support.
- Reverse Jacobian storage has an explicit 1 GiB default allocation guard.
  Additional differentiation/QGT workspace is outside this single-array cap.
- The TFIM example adds opt-in `--tdvp-diagnostics` step JSON records.
  [API documentation](../docs/api/vmc.md#netket-peps-time-evolution) and
  [changelog](../CHANGELOG.md) describe the new endpoint contract.

## Validation

- Initial focused diagnostics: **6 passed**, 48 deselected, 16.67 s.
  Independent analytic log derivatives and NumPy equations cover both charts
  and both propagation types, sampled CG, stale-stage restoration, native
  cache preservation, continuation and compressed PEPS/budget handling.
- Added a zero-normal-force regression exposing missing physical motion and
  enabled diagnostics in the existing bounded TFIM snapshot/reference checks.
  Broader TDVP/backend-neutral VMC/public-API/package-layout selection:
  **155 passed, 1 failed**, 158.17 s. The new zero-force case exposed that
  NetKet rejects complex Jacobian mode for genuinely real model output.
  Diagnostics now detect the output dtype and choose the real mode there.
- Final endpoint/TFIM-example selection after that fix: **9 passed**,
  46 deselected, 29.57 s, including the formerly failing zero-force case.
  Counts overlap; the full four-file selection was not repeated after this
  localized mode fix. No unresolved failures remain in the final selection.
- Whole-source/test/example Ruff and local API Markdown links passed.
- Installed APIs inspected: NetKet 3.22.4, JAX/JAXlib 0.11.2, Flax 0.12.10,
  Quimb 1.15.1.dev91+gb3bf45a50, Symmray 0.4.1.dev18+g44fc5ae53.
  Checks use the existing Python 3.12 environment with CPU JAX and single
  thread limits. No dependency or installed-library changes.
- Reviewed public [TDVP](https://netket.readthedocs.io/en/stable/api/_generated/experimental/dynamics/netket.experimental.TDVP.html)
  and [Jacobian](https://netket.readthedocs.io/en/stable/api/_generated/jax/netket.jax.jacobian.html)
  APIs plus installed source. Adopt endpoint native solves and reverse
  Jacobians; do not copy private stage caches into endpoint diagnostics.

## Limits

Diagnostics are opt-in, use the solve's own batch and exclude directions
supported entirely at wavefunction zeros. They do not establish independent
sampling, contraction or long-time trajectory convergence. The degenerate-SVD
gradient follow-up, native sparse derivatives and sample-space dynamics remain
outside this task. No full repository suite or GPU validation is claimed.

## Follow-up review

- Re-read endpoint synchronization, chart equations, native solve/cache
  ownership, sampling statistics and CLI output. No blocking issue found
  within the reviewed scope; source implementation was unchanged.
- Fresh endpoint/example selection: **9 passed**, 46 deselected, 29.05 s.
- Additional frozen 3x3/D=2/chi=1 boundary-CG probe confirms actual
  truncation (sampled amplitude relative difference from exact: 0.2052).
  Regularized solve residual: 4.953e-11; unregularized: 0.04989; tangent:
  0.10730. This 16-sample probe checks execution and distinct diagnostics,
  not contraction or sampling accuracy. No trajectory was launched.
- Concurrent composed-projector/FIT gradient changes are visible but were
  not validated by this review. Do not treat the earlier gradient-gap note
  as proof that these newer changes are absent or complete.

## Publication validation

The later [basics audit](2026-10-14-netket-tdvp-basics-audit.md) supersedes the
earlier four-file rerun limitation: **224 passed** on the combined working
tree. It also records independent checks of the scoped publication snapshot.
