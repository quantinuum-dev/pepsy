# 2026-10-06 — Gradient optimizer corrections

Follow-up to the [read-only review](2026-10-06-gradient-optimizer-review.md),
implemented in the working tree based on `develop` at `2abb1e8`.
Unrelated MPS, instructions, notebook, and stored-data changes were preserved.
The [publication follow-up](../../../history/2026-10-06-gradient-publication.md)
records additional native-gradient, terminal-autograd and NLopt corrections,
Gaugy sweep integration, and final commit/push status.

## Implemented

1. Native JAX/Optax conjugates the gradient of a real loss before applying a
   complex update. Autoray dispatch supplies `conj` without host conversion.
   Mixed real/complex parameters retain their dtype and device.
2. Native Torch (including LBFGS) and FD Adam score the final update and count
   the evaluation. Best restoration considers that update, and invalid or
   worse updates preserve the finite best point. History remains the sequence
   of pre-update step losses. Native unbounded Torch best-state snapshots use
   the existing device-local clone/restore helpers instead of NumPy packing.
3. SciPy and FD SciPy report `invalid_objective` with the underlying numerical
   failure when the returned point has an invalid loss/gradient. A rejected
   bad trial at another point does not by itself invalidate the result.
   Scalar/real contract errors propagate as `ValueError`. Cached SciPy values
   replace uncounted fallback evaluations; terminal loss follows returned state.
4. SciPy aliases use precedence `maxiter > maxeval > its_max > n_steps`.
   `maxeval` remains a legacy iteration alias, not a strict loss-call cap.
   TNC uses `maxfun` and receives no default unsupported `maxiter`/`maxls`;
   SLSQP receives no unsupported default `gtol`. Explicit PEPS sweep limits
   survive option merging, while the default SciPy budget remains 30.
5. SciPy/FD SciPy accept both callback signatures, including `trust-constr`'s
   result argument; its accepted-point value supplies callback history.
6. Native JAX rejects unsupported bounds, step limits, and angle wrapping
   before evaluating the objective. This fixes silent constraint omission;
   it does not implement constrained Optax algorithms.
7. qMERA energy runs default to `assume_nonnegative=False` and preserve explicit
   overrides, so NLopt can retain negative ground-state energies.
8. Native JAX checks the imaginary component at each step and terminal
   evaluation, rejecting magnitudes above `1e-10` and nonfinite imaginary
   values. Near-real roundoff remains accepted.

The owning [gradient](../../api/solvers/gradient.md),
[finite-difference](../../api/solvers/finite_difference.md), and
[qMERA](../../api/optimizers/qmera.md) guides document the behavior.

## Dependency audit and decisions

The task's installed-source audit used Torch `2.6.0+cu124`, JAX `0.10.2`,
Optax `0.2.8`, SciPy `1.17.1`, NLopt `2.11.0`,
Autoray `0.11.1.dev9+g1291702f9`, Quimb `1.15.1.dev79+gb5e316200`,
Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray `0.4.1.dev11+g1a3481803`.
Inspected `minimize` and its callback wrapper, `jax.value_and_grad(has_aux)`,
Optax factories, Torch LBFGS, and Autoray dispatch capabilities.

- **Adopt:** [JAX's complex differentiation convention](https://docs.jax.dev/en/latest/complex-differentiation.html),
  [SciPy method/callback options](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
  and [Autoray](https://github.com/jcmgray/autoray) conjugation dispatch.
- **Defer:** a general backend abstraction, new constrained Optax algorithms,
  and moving explicit Torch bound clipping off the host. Backend-specific
  autodiff and ownership semantics remain explicit.
- No new upstream compatibility shim or installed-library edits.

Also checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The Symmray abelian-array
documentation page was unavailable to the browser; installed/official source
was available. These changes do not alter contraction or symmetry algorithms.

## Validation evidence

- Initial new regression selection: **42 failed, 5 passed** before corrections.
- Corrected solver selection: **173 passed**, including complex Optax descent,
  terminal state/loss agreement, invalid-final recovery, explicit constraint
  rejection, SciPy budgets/callbacks, real PEPS Torch/NumPy integration, and
  signed qMERA NLopt energy optimization.
- The full-suite attempt was interrupted after **24 failed, 125 passed,
  2 skipped**. All 24 failure node IDs matched failures reproduced from an
  isolated `git archive HEAD` copy of the two affected modules:
  `tests/test_bp_compression.py` and `tests/test_bp_open_series.py`
  (**24 failed, 42 passed**). Their failures require converged BP messages;
  they precede these optimizer edits. The complete suite is not validated.
- Final combined domain/API checks are recorded in the linked
  [session handoff](../../../history/2026-10-06-gradient-fixes.md).

Validation uses CPU Torch/JAX. Accelerator behavior and performance were not
measured. Native Symmray qMERA and PEPS coverage comes from the existing domain
tests, not a new approximation or dense replacement.
